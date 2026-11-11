# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Utility functions and enums for iris-ccl collective operations.
"""

from enum import IntEnum
from typing import Tuple
import triton
import triton.language as tl
from iris.host.distributed.helpers import extract_group_info as _extract_group_info


@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups: tl.constexpr, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """
    Transform program ID for chiplet-aware workgroup distribution.

    This function redistributes workgroups across multiple XCDs (chiplets) in chunks
    to improve load balancing and memory access patterns.

    Args:
        pid: Program ID to transform
        num_workgroups: Total number of workgroups
        num_xcds: Number of XCDs (chiplets)
        chunk_size: Size of chunks for distribution

    Returns:
        Transformed program ID
    """
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


class ReduceOp(IntEnum):
    """
    Reduction operations for collective communications.
    Matches torch.distributed.ReduceOp semantics.

    Note: Currently only SUM is implemented. Other operations will be added in future releases.
    """

    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6


def extract_group_info(group, ctx) -> Tuple[int, int, int, int, int]:
    """
    Extract group information for collective operations.

    Args:
        group: ProcessGroup or None. If None, uses all ranks in ctx.
        ctx: Iris context

    Returns:
        Tuple of (rank_in_group, rank_global, world_size, rank_start, rank_stride)
        - rank_in_group: Rank within the group (0-indexed)
        - rank_global: Global rank of this process
        - world_size: Number of ranks in the group
        - rank_start: Starting global rank of the group
        - rank_stride: Stride between consecutive ranks in the group
    """

    return _extract_group_info(group, ctx.get_rank(), ctx.get_num_ranks())


def _ensure_symmetric(ctx, tensor, name="tensor"):
    """Return tensor on symmetric heap, importing if needed.

    For input tensors that are only read: the kernel reads from the heap
    copy, so the caller doesn't need the returned tensor back.

    Do NOT use this for output tensors — see _validate_output_symmetric.
    """
    if ctx.is_symmetric(tensor):
        return tensor
    return ctx.as_symmetric(tensor)


def _validate_output_symmetric(ctx, tensor, name="output_tensor"):
    """Raise if output tensor is not on symmetric heap.

    Output tensors cannot be auto-imported because the kernel writes results
    into the heap copy, but the caller's original tensor would never see
    those results (torch allocator makes an independent copy).
    """
    if not ctx.is_symmetric(tensor):
        raise ValueError(
            f"{name} must be on the symmetric heap. "
            f"Allocate with ctx.zeros() or import with ctx.as_symmetric() before calling."
        )
