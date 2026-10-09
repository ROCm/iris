# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""PyTorch symmetric memory as an allocation provider for Iris device kernels.

Setup, measured torch builds and how this differs from the rocSHMEM provider are
in iris/experimental/README.md. What matters in the code:

- The rendezvous handle's ``buffer_ptrs`` already is the table Iris translates
  against, with the local entry equal to the tensor's ``data_ptr()``. The
  provider reads it per allocation and checks that invariant; it computes
  nothing.
- Use each table only for its own allocation. Whether peer offsets are shared
  between allocations depends on the backend, and on the default one they need
  not be.
- The provider never calls ``symm_mem.set_backend``; it uses whatever backend
  the caller selected, or torch's default.
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
