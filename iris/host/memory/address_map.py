# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Allocator-agnostic symmetric address metadata.

Iris device-side translation only needs to answer one question: given a pointer
that is local to this rank, what address reaches the same location on a peer?
The answer is the same arithmetic regardless of which host allocator produced
the tensor::

    offset = local_pointer - local_allocation_base
    remote_pointer = peer_bases[target_rank] + offset

:class:`SymmetricAddressMap` is the normalized form of the metadata that
arithmetic requires. Providers (the Iris allocators today, rocSHMEM or Torch
Symmetric Memory later) own allocation, lifetime, handle exchange and peer-base
production; Iris consumes only this descriptor.

The descriptor is attached to a tensor's *backing allocation*, not to the Iris
context, so a single kernel can consume tensors from different providers with
different peer-base tables while running identical device code.
"""

from dataclasses import dataclass

import torch

#: The provider guarantees remote loads and stores through translated pointers.
CAP_REMOTE_LOAD_STORE = 1 << 0

#: The provider guarantees remote atomics through translated pointers.
CAP_REMOTE_ATOMICS = 1 << 1


@dataclass(frozen=True)
class SymmetricAddressMap:
    """
    Normalized peer address metadata for one backing allocation.

    Args:
        peer_bases (torch.Tensor): ``int64[world_size]`` device-resident tensor
            of backing-allocation base addresses, indexed by rank. This is the
            only field device code needs; pass it to kernels alongside the
            tensor pointer. Device-side translation casts the pointer to
            ``tl.uint64`` before subtracting, so signed storage is fine.
        local_rank (int): Rank of the calling process.
        allocation_base (int): Base address of the backing allocation on this
            rank. Views translate against this, not against the view pointer.
        allocation_bytes (int): Size of the backing allocation in bytes.
        capabilities (int): Bitmask of ``CAP_*`` flags the provider guarantees.

    Invariant:
        ``peer_bases[local_rank] == allocation_base``. Translation subtracts the
        local base and adds the peer base, so a descriptor that violates this
        produces silently wrong remote addresses.

    Example:
        >>> tensor, address_map = ctx.allocate_symmetric(1024, dtype=torch.float32)
        >>> kernel[grid](tensor, address_map.peer_bases, target_rank, ...)
    """

    peer_bases: torch.Tensor
    local_rank: int
    allocation_base: int
    allocation_bytes: int
    capabilities: int = CAP_REMOTE_LOAD_STORE | CAP_REMOTE_ATOMICS

    def __post_init__(self):
        if self.peer_bases.dtype not in (torch.int64, torch.uint64):
            raise ValueError(f"peer_bases must be int64 or uint64, got {self.peer_bases.dtype}")
        if not 0 <= self.local_rank < self.peer_bases.numel():
            raise ValueError(f"local_rank {self.local_rank} out of range for {self.peer_bases.numel()} ranks")

        local_base = int(self.peer_bases[self.local_rank].item())
        if local_base != self.allocation_base:
            raise ValueError(
                f"peer_bases[{self.local_rank}]={hex(local_base)} does not match "
                f"allocation_base={hex(self.allocation_base)}; translation would "
                "produce wrong remote addresses"
            )

    @property
    def world_size(self) -> int:
        """Number of ranks in the peer-base table."""
        return self.peer_bases.numel()

    def owns(self, tensor: torch.Tensor) -> bool:
        """
        Check that ``tensor`` lies inside the allocation this map describes.

        Passing a tensor and an address map as separate kernel arguments allows
        them to be paired by mistake -- a tensor from one provider with another
        provider's map. Such a pairing translates against the wrong base and
        corrupts memory silently. Different providers hand out disjoint address
        ranges, so a bounds check against the backing allocation catches it.

        This is a host-side check against metadata that is fully known before
        launch; it costs nothing at runtime. It cannot catch a mispairing
        between two allocations that share a backing range, which under the
        context-wide heap means two Iris tensors -- but those share a base, so
        translating one against the other's map is harmless today and becomes
        detectable once per-allocation bases replace the shared heap.

        Args:
            tensor (torch.Tensor): Tensor to check.

        Returns:
            bool: True if the tensor's storage lies within the backing allocation.
        """
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        return self.allocation_base <= start and end <= self.allocation_base + self.allocation_bytes

    def supports(self, capability: int) -> bool:
        """
        Check whether the provider guarantees a capability.

        Args:
            capability (int): One of the ``CAP_*`` flags.

        Returns:
            bool: True if the flag is set.
        """
        return bool(self.capabilities & capability)
