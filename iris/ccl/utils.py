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
from iris.host.distributed.helpers import _translate_ptr


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


@triton.jit()
def inline_device_barrier(
    pid,
    barrier_flags_ptr,
    wg_done_ptr,
    barrier_sense_ptr,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    """
    Inline device barrier — folds cross-rank sync into the collective kernel.

    All CTAs signal completion, pid 0 does cross-rank barrier, then signals
    all other CTAs to proceed. Eliminates the second kernel launch (~15us).

    Uses a sense-reversal barrier so wg_done doesn't need resetting between calls.
    """
    tl.debug_barrier()

    sense = tl.load(barrier_sense_ptr)
    next_sense = 1 - sense

    if pid == 0:
        # pid 0: wait for all CTAs to arrive
        tl.atomic_add(wg_done_ptr, 1, sem="acquire", scope="gpu")
        while tl.atomic_cas(wg_done_ptr, COMM_SMS, COMM_SMS, sem="acquire", scope="gpu") < COMM_SMS:
            pass

        # Reset wg_done for next use
        tl.atomic_xchg(wg_done_ptr, 0, sem="release", scope="gpu")

        # Cross-rank barrier: increment own flag, poll remotes
        own_flag_ptr = barrier_flags_ptr + iris_rank
        own_translated = _translate_ptr(own_flag_ptr, iris_rank, iris_rank, heap_bases)
        old = tl.atomic_add(own_translated, 1, sem="release", scope="sys")
        target = old + 1

        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            if remote_rank != iris_rank:
                remote_flag_ptr = barrier_flags_ptr + remote_rank
                remote_translated = _translate_ptr(remote_flag_ptr, iris_rank, remote_rank, heap_bases)
                while tl.atomic_cas(remote_translated, target, target, sem="acquire", scope="sys") < target:
                    pass

        # Signal all CTAs: flip sense
        tl.atomic_xchg(barrier_sense_ptr, next_sense, sem="release", scope="gpu")
    else:
        # Non-zero pids: signal arrival then wait for pid 0
        tl.atomic_add(wg_done_ptr, 1, sem="release", scope="gpu")
        while tl.atomic_cas(barrier_sense_ptr, next_sense, next_sense, sem="acquire", scope="gpu") != next_sense:
            pass
