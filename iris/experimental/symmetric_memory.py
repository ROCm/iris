# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Address metadata shared by allocation providers.

Iris device code translates a pointer with

    remote = peer_bases[to] + (ptr - peer_bases[local_rank])

so any provider that can produce a table satisfying
``peer_bases[local_rank] == local allocation base`` drives iris.store, load and
copy unchanged. This module holds the descriptor that carries such a table plus
what the bare table cannot express, notably per-peer reachability.

It deliberately imports nothing but torch, so a provider for one runtime never
drags in another's dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
