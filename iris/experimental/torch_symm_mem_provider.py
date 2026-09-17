# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""PyTorch symmetric memory as an allocation provider for Iris device kernels.

Lets Iris device code operate on tensors from ``torch.distributed._symmetric_memory``
rather than from Iris's own symmetric heap. No Iris device code changes are
needed: iris.store, load and copy take ``heap_bases`` as a plain pointer argument
and translate with

    remote = peer_bases[to] + (ptr - peer_bases[local_rank])

so any table satisfying ``peer_bases[local_rank] == local allocation base``
drives them.

Torch hands that table over directly. Rendezvous returns a handle whose
``buffer_ptrs`` is a per-peer list of pointers into this process's address space,
with the local rank's entry equal to the tensor's own ``data_ptr()`` -- exactly
the invariant Iris needs. So this provider does no pointer arithmetic at all; it
reads the table off the handle per allocation.

That is worth preferring over the alternative even though the alternative also
works. Peer offsets happen to be constant across allocations, so one table could
be reused for all of them, but that only holds while every rank allocates in
lockstep: ``symm_mem.empty`` is a local call and only ``rendezvous`` is
collective, so ranks can diverge. Reading ``buffer_ptrs`` per allocation is
correct either way.

Scope is intra-node, set by the backend torch selects. The default backend
reports as ``'CUDA'`` and is the HIP IPC path on ROCm; it reaches peers sharing a
node. Do not call ``set_backend`` to try to change this -- the name
``get_backend`` returns is not one ``set_backend`` accepts, and forcing a
different backend selects one that fails at allocation.

``SymmetricAddressMap.direct`` records per-peer reachability for symmetry with
the other providers; a peer that is not directly addressable gets a base of 0,
which would translate to a wild pointer rather than an error, so callers are
expected to check it before launching.

This module is deliberately NOT imported by ``iris/experimental/__init__.py``.
Torch symmetric memory is a private torch API and its availability varies by
build, so importing it eagerly would make ``import iris`` fail on builds that
lack it.

The caller owns bootstrap and tensor lifetime; the process group must already be
initialised on a device backend:

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    provider = TorchSymmMemProvider()
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from iris.experimental.symmetric_memory import SymmetricAddressMap

__all__ = ["TorchSymmMemProvider", "SymmetricAddressMap"]


class TorchSymmMemProvider:
    """Allocates torch symmetric tensors and describes them for Iris kernels."""

    def __init__(self, group: dist.ProcessGroup | None = None, device: torch.device | str | None = None):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialised before constructing "
                "TorchSymmMemProvider; symmetric memory rendezvous is a collective."
            )
        self._group = group if group is not None else dist.group.WORLD
        self.cur_rank = dist.get_rank(self._group)
        self.num_ranks = dist.get_world_size(self._group)
        self._device = (
            torch.device(device) if device is not None else torch.device(f"cuda:{torch.cuda.current_device()}")
        )
        # data_ptr -> handle, so an allocation can be described again later
        # without a second rendezvous. Rendezvous is collective: calling it from
        # one rank alone would hang the others.
        self._handles: dict[int, object] = {}

    # ── table form ───────────────────────────────────────────────────────────

    def allocate_symmetric(self, *size, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate a symmetric tensor and return it with its peer-base table.

        Same signature and return shape as Iris.allocate_symmetric, so the same
        device kernels drive either provider.
        """
        tensor, amap = self.allocate_symmetric_map(*size, dtype=dtype)
        return tensor, amap.peer_bases

    # ── descriptor form ──────────────────────────────────────────────────────

    def allocate_symmetric_map(self, *size, dtype=None) -> tuple[torch.Tensor, SymmetricAddressMap]:
        """As allocate_symmetric, but returning the full address descriptor.

        Every rank must call this the same number of times and in the same
        order: the rendezvous inside is a collective.
        """
        shape = tuple(size[0]) if len(size) == 1 and hasattr(size[0], "__iter__") else tuple(size)
        dtype = dtype or torch.get_default_dtype()

        # State may be built from a forward running under inference_mode, which
        # would otherwise mark the allocation inference-only.
        with torch.inference_mode(False), torch.no_grad():
            tensor = symm_mem.empty(*shape, dtype=dtype, device=self._device)
        handle = symm_mem.rendezvous(tensor, group=self._group)
        self._handles[tensor.data_ptr()] = handle
        return tensor, self._map_from_handle(tensor, handle)

    def symmetric_address_map(self, tensor: torch.Tensor) -> SymmetricAddressMap:
        """Describe an already-allocated symmetric tensor.

        The tensor must have been allocated through this provider, whose handle
        is reused. Rendezvousing again here would be a collective call made from
        whichever rank happened to ask.
        """
        handle = self._handles.get(tensor.data_ptr())
        if handle is None:
            raise KeyError(
                "tensor was not allocated by this provider, so its rendezvous "
                "handle is unknown. Allocate through allocate_symmetric to have "
                "the handle recorded; rendezvous cannot be repeated here because "
                "it is collective."
            )
        return self._map_from_handle(tensor, handle)

    def _map_from_handle(self, tensor: torch.Tensor, handle) -> SymmetricAddressMap:
        """Build the descriptor from the handle's own peer-pointer table."""
        bases = [int(p) for p in handle.buffer_ptrs]

        # Length is checked, not assumed. Iris indexes this table by rank inside
        # the kernel, so a short table reads past the end of the allocation
        # rather than raising.
        if len(bases) != self.num_ranks:
            raise RuntimeError(
                f"handle.buffer_ptrs has {len(bases)} entries, expected "
                f"{self.num_ranks}; the table Iris indexes by rank would be short."
            )

        # The invariant every Iris translation depends on. Checked rather than
        # assumed: a silent mismatch here turns every remote address in a kernel
        # into a wild pointer.
        if bases[self.cur_rank] != tensor.data_ptr():
            raise RuntimeError(
                f"buffer_ptrs[{self.cur_rank}]={bases[self.cur_rank]:#x} does not "
                f"match the tensor base {tensor.data_ptr():#x}; Iris address "
                "translation would produce wild pointers."
            )

        return SymmetricAddressMap(
            peer_bases=torch.tensor(bases, dtype=torch.int64, device=tensor.device),
            local_rank=self.cur_rank,
            allocation_base=tensor.data_ptr(),
            allocation_bytes=tensor.numel() * tensor.element_size(),
            direct=tuple(b != 0 for b in bases),
        )

    # ── convenience ──────────────────────────────────────────────────────────

    def barrier(self):
        dist.barrier(self._group)

    def free(self, tensor: torch.Tensor):
        """Drop this provider's reference to the allocation.

        Torch symmetric memory is reference counted like any other tensor, so
        the storage goes away when the caller's reference does too. This only
        forgets the handle.
        """
        self._handles.pop(tensor.data_ptr(), None)

    def get_rank(self) -> int:
        return self.cur_rank

    def get_num_ranks(self) -> int:
        return self.num_ranks
