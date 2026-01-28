# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Utility functions and enums for iris-ccl collective operations.
"""

from enum import IntEnum
from typing import Tuple
import triton
import triton.language as tl


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


def extract_group_info(group, shmem) -> Tuple[int, int, int, int, int]:
    """
    Extract group information for collective operations.

    Args:
        group: ProcessGroup or None. If None, uses all ranks in shmem context.
        shmem: Iris shmem context

    Returns:
        Tuple of (rank_in_group, rank_global, world_size, rank_start, rank_stride)
        - rank_in_group: Rank within the group (0-indexed), used for tile assignment and comparisons
        - rank_global: Global rank of this process, used for iris RMA operations (heap_bases indexing)
        - world_size: Number of ranks in the group
        - rank_start: Starting global rank of the group
        - rank_stride: Stride between consecutive ranks in the group

    Examples:
        >>> # group=None: all ranks [0,1,2,3], current global rank is 2
        >>> extract_group_info(None, shmem)
        (2, 2, 4, 0, 1)  # rank_in_group=2, rank_global=2, world_size=4, start=0, stride=1

        >>> # TP group: consecutive ranks [0,1,2,3], current global rank is 2
        >>> extract_group_info(tp_group, shmem)
        (2, 2, 4, 0, 1)  # rank_in_group=2, rank_global=2, world_size=4, start=0, stride=1

        >>> # DP group: strided ranks [0,4,8,12], current global rank is 8
        >>> extract_group_info(dp_group, shmem)
        (2, 8, 4, 0, 4)  # rank_in_group=2, rank_global=8, world_size=4, start=0, stride=4
    """
    if group is None:
        # Use all ranks in shmem context
        # When group is None, rank_in_group equals rank_global
        rank_global = shmem.get_rank()
        rank_in_group = rank_global
        world_size = shmem.get_num_ranks()
        rank_start = 0
        rank_stride = 1
        return rank_in_group, rank_global, world_size, rank_start, rank_stride

    # Extract from ProcessGroup
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized to use ProcessGroup. "
            "Call torch.distributed.init_process_group() first."
        )

    group_ranks = dist.get_process_group_ranks(group)
    world_size = len(group_ranks)
    rank_global = dist.get_rank()

    if rank_global not in group_ranks:
        raise RuntimeError(
            f"Current rank {rank_global} is not part of the specified process group. "
            f"Group contains ranks: {group_ranks}"
        )

    rank_in_group = group_ranks.index(rank_global)

    # Detect stride pattern
    if len(group_ranks) > 1:
        # Check if all consecutive pairs have the same stride
        strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
        is_strided = all(s == strides[0] for s in strides)

        if is_strided:
            rank_start = group_ranks[0]
            rank_stride = strides[0]

            # Validate rank_stride is not zero (would indicate duplicate ranks)
            if rank_stride == 0:
                raise ValueError(
                    f"Invalid process group: rank_stride is 0, indicating duplicate ranks. "
                    f"Group ranks: {group_ranks}. "
                    f"Each rank must appear exactly once in a process group."
                )
        else:
            # Non-strided group - not supported yet
            raise NotImplementedError(
                f"Non-strided process groups are not yet supported. "
                f"Group ranks: {group_ranks}. "
                f"Please use groups with uniform stride (e.g., [0,1,2,3] or [0,4,8,12])."
            )
    else:
        # Single rank group
        rank_start = group_ranks[0]
        rank_stride = 1

    return rank_in_group, rank_global, world_size, rank_start, rank_stride
