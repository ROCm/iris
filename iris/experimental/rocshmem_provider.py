# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""rocSHMEM as an allocation provider for Iris device kernels.

Lets Iris device code operate on tensors allocated by rocSHMEM rather than from
Iris's own symmetric heap. No Iris device code changes are needed: iris.store,
load and copy take ``heap_bases`` as a plain pointer argument and translate with

    remote = peer_bases[to] + (ptr - peer_bases[local_rank])

so any table satisfying ``peer_bases[local_rank] == local allocation base``
drives them. This module builds that table for rocSHMEM memory.

The table comes from ``rocshmem_ptr(base, peer)``, OpenSHMEM's ``shmem_ptr``: an
address in this process's own address space for the peer's counterpart of a
symmetric object, or NULL when that peer is not reachable by direct load/store.

The per-peer offsets are queried once. rocSHMEM's peer mapping is a single
linear translation of the whole symmetric heap, so the offset from a local
address to its counterpart on a given peer is the same constant everywhere in
the heap. Only those offsets are cached; each allocation's table is materialised
from its own base, so ``peer_bases[local_rank]`` is always that allocation's
base. rocSHMEM's heap base, which it does not expose publicly, is never needed.

Because the offsets are shared, a table built for one allocation still
translates pointers belonging to another. iris.copy relies on that: it takes one
``heap_bases`` and translates two pointers against it.

Scope is intra-node. A peer not reachable by direct load/store gets a base of 0,
which would translate to a wild pointer rather than an error, so
``SymmetricAddressMap.direct`` records reachability per peer and callers are
expected to check it before launching. Inter-node peers need a transport this
module does not provide.

TODO: settle where this belongs. It sits in Iris on the assumption that Iris
hosts provider adapters; the alternative is for it to live alongside rocSHMEM,
which owns the allocation and the tensor lifetime. It is one file either way.

This module is deliberately NOT imported by ``iris/experimental/__init__.py``,
so ``import iris`` does not require rocshmem4py. Keep it that way: adding it to
that package's eager imports would make a rocSHMEM install mandatory for every
Iris user.

The dependency is ``rocshmem4py``, a standalone Python package from the
ROCm/rocm-systems repository rather than something a ROCm install provides. It
does not link rocSHMEM at run time; it statically links it into its extension
module, and its version records which rocSHMEM that was (e.g.
``0.1.0+rocshmem3.7.0``). So installing it needs no separate rocSHMEM on the
system, and the rocSHMEM build options it was compiled with -- ``USE_IPC`` in
particular -- are fixed at its build time, not selectable later.

The caller owns bootstrap and tensor lifetime; rocSHMEM must already be
initialised:

    dist.init_process_group(backend="gloo")
    rocshmem4py.init_rocshmem_by_uniqueid(dist.group.WORLD)
    provider = RocshmemProvider()
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import rocshmem4py as rshmem
from rocshmem4py.interop import torch as rshmem_torch


@dataclass(frozen=True)
class SymmetricAddressMap:
    """Address metadata for one symmetric allocation.

    ``allocate_symmetric`` returns only ``(tensor, peer_bases)``; this carries
    what that pair cannot, notably ``direct``.
    """

    peer_bases: torch.Tensor  # int64[world_size], device-resident
    local_rank: int
    allocation_base: int
    allocation_bytes: int
    direct: tuple[bool, ...]  # per peer: reachable by load/store?

    def all_direct(self) -> bool:
        return all(self.direct)

    def indirect_peers(self) -> list[int]:
        return [r for r, d in enumerate(self.direct) if not d]


class RocshmemProvider:
    """Allocates rocSHMEM symmetric tensors and describes them for Iris kernels."""

    def __init__(self, device: str | None = None):
        self.cur_rank = rshmem.rocshmem_my_pe()
        self.num_ranks = rshmem.rocshmem_n_pes()
        self._device = device
        # peer -> byte offset from a local address to its counterpart on that
        # peer, or None when the peer is not reachable by load/store. Constant
        # across the heap, so it is computed once from the first allocation.
        self._deltas: list[int | None] | None = None

    # ── table form ───────────────────────────────────────────────────────────

    def allocate_symmetric(self, *size, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate a symmetric tensor and return it with its peer-base table.

        Same signature and return shape as Iris.allocate_symmetric, so the same
        device kernels drive either provider.

        Collective: rocSHMEM allocation is, so every PE must call this the same
        number of times and in the same order.

        Returns ``(tensor, peer_bases)`` where ``peer_bases`` is an
        ``int64[num_ranks]`` tensor on the same device as ``tensor``, holding for
        each peer the address of that peer's counterpart of this allocation, in
        this process's address space. Its ``local_rank`` entry is this
        allocation's own base, which is what Iris translation subtracts. A peer
        not reachable by direct load/store is 0; ``allocate_symmetric_map``
        returns the same thing plus the ``direct`` mask that says which, and
        callers that may run inter-node should check it rather than launching
        against a 0.

        Hold the returned table for as long as the allocation lives rather than
        re-deriving it per launch; it is built once here and does not change.
        """
        tensor, amap = self.allocate_symmetric_map(*size, dtype=dtype)
        return tensor, amap.peer_bases

    # ── descriptor form ──────────────────────────────────────────────────────

    def allocate_symmetric_map(self, *size, dtype=None) -> tuple[torch.Tensor, SymmetricAddressMap]:
        """As allocate_symmetric, but returning the full address descriptor."""
        shape = tuple(size[0]) if len(size) == 1 and hasattr(size[0], "__iter__") else tuple(size)
        dtype = dtype or torch.get_default_dtype()
        tensor = rshmem_torch.create_tensor(shape, dtype)
        return tensor, self.symmetric_address_map(tensor)

    def symmetric_address_map(self, tensor: torch.Tensor) -> SymmetricAddressMap:
        """Describe an already-allocated rocSHMEM tensor.

        Materialises a fresh ``int64[num_ranks]`` table on each call. That is one
        small device tensor per allocation on the normal path, since
        ``allocate_symmetric`` calls this once; it is not meant to be called per
        kernel launch. The table is not memoised on purpose: keying a cache by
        ``data_ptr()`` would alias once an allocation is freed and its address
        reused, and the result would be a silently wrong table rather than an
        error.
        """
        base = tensor.data_ptr()
        deltas = self._peer_deltas(tensor)
        bases = [0 if d is None else base + d for d in deltas]
        return SymmetricAddressMap(
            peer_bases=torch.tensor(bases, dtype=torch.int64, device=tensor.device),
            local_rank=self.cur_rank,
            allocation_base=base,
            allocation_bytes=tensor.numel() * tensor.element_size(),
            direct=tuple(d is not None for d in deltas),
        )

    def _peer_deltas(self, anchor: torch.Tensor) -> list[int | None]:
        """Per-peer byte offsets, queried once and reused.

        rocshmem_ptr is a linear translation of the whole symmetric heap, so the
        offset to a peer's counterpart is the same for every address in it. Only
        the offsets are cached; each allocation's table is materialised from its
        own base, which keeps peer_bases[local_rank] == that allocation's base.
        """
        if self._deltas is not None:
            return self._deltas

        base = anchor.data_ptr()
        deltas: list[int | None] = []
        for peer in range(self.num_ranks):
            if peer == self.cur_rank:
                deltas.append(0)
                continue
            p = int(rshmem.rocshmem_ptr(base, peer))
            deltas.append(p - base if p else None)

        # Every peer unreachable usually means rocSHMEM was built with
        # USE_IPC=OFF rather than that every peer is remote: with IPC compiled
        # out rocshmem_ptr returns NULL unconditionally. Upstream defaults it
        # ON. Failing here beats handing back a table whose zeros would
        # translate to wild pointers inside a kernel.
        peers = [r for r in range(self.num_ranks) if r != self.cur_rank]
        if peers and all(deltas[r] is None for r in peers):
            raise RuntimeError(
                "rocshmem_ptr returned NULL for every peer. If any peer shares "
                "this node, rocSHMEM was likely built with USE_IPC=OFF (upstream "
                "defaults ON); check the USE_IPC line in the rocSHMEM banner."
            )

        self._deltas = deltas
        return deltas

    # ── convenience ──────────────────────────────────────────────────────────

    def barrier(self):
        rshmem_torch.barrier_all()

    def free(self, tensor: torch.Tensor):
        """Release a symmetric allocation. Collective.

        Explicit by necessity, not by preference. rocshmem_free is documented as
        "a collective operation and must be called by all PEs", so it cannot be
        driven from ``__del__`` or a weakref finalizer: Python decides when to
        collect per process, and ranks that collect in different orders, or at
        different times, would diverge and hang instead of raising. Freeing has
        to stay where the caller can order it across ranks.
        """
        rshmem_torch.free_tensor(tensor)

    def get_rank(self) -> int:
        return self.cur_rank

    def get_num_ranks(self) -> int:
        return self.num_ranks
