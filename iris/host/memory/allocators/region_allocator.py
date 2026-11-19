# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Bump allocator over a fixed region another runtime provides and maps.

Used for ``engines=("rdma",)`` with a backend that provides the heap (RCCL
GIN): the backend allocates the region and maps it on the peers it can reach,
so Iris neither allocates nor exchanges handles; it only sub-allocates.
"""

import numpy as np
import torch

from .base import BaseAllocator
from .torch_allocator import TorchAllocator


class RegionAllocator(TorchAllocator):
    """Bump allocation (as :class:`TorchAllocator`) inside ``region.tensor``,
    with peer bases fixed by the region. The heap cannot grow."""

    def __init__(self, region, device_id: int, cur_rank: int, num_ranks: int):
        tensor = region.tensor
        BaseAllocator.__init__(self, tensor.numel() * tensor.element_size(), device_id, cur_rank, num_ranks)
        if len(region.peer_bases) != num_ranks:
            raise ValueError(f"region has {len(region.peer_bases)} peer bases for {num_ranks} ranks")
        if int(region.peer_bases[cur_rank]) != tensor.data_ptr():
            raise ValueError("region.peer_bases[cur_rank] must be the region's own address")
        self.device = f"cuda:{device_id}"
        self.rank_bools = None
        self.memory_pool = tensor.view(torch.int8)
        self.region = region
        self.heap_bases_array = np.array([int(b) for b in region.peer_bases], dtype=np.uint64)
        self._peer_ext_mem_handles = {}

    def get_shareable_handle(self) -> tuple:
        raise RuntimeError("a provided region is mapped on peers by its provider, not shared by Iris")

    def establish_peer_access(self, all_bases, connections=None):
        """Nothing to import: the provider mapped the region on every direct peer."""

    def close(self):
        """The provider owns the memory and frees it."""
