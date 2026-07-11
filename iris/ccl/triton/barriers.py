# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
In-kernel barrier primitives for graph-capturable collectives.

_per_block_barrier uses monotonic atomic counters on the symmetric heap.
Each CTA signals all remote ranks and polls until all have signaled.
Counters never reset — safe across graph replays.
"""

import triton
import triton.language as tl
from iris.host.distributed.helpers import _translate_ptr


@triton.jit()
def per_block_barrier(
    pid,
    flags_ptr,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
):
    """Per-block cross-rank barrier using monotonic atomic counters.

    flags layout: (max_blocks, world_size) int32 on symmetric heap.
    Each CTA increments its own flag and all remote copies, then polls
    until all peers have reached the same count.
    """
    tl.debug_barrier()

    my_flag_ptr = flags_ptr + pid * world_size + group_rank
    my_local = _translate_ptr(my_flag_ptr, iris_rank, iris_rank, heap_bases)
    old = tl.atomic_add(my_local, 1, sem="release", scope="sys")
    target = old + 1

    for i in tl.static_range(0, world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            remote_translated = _translate_ptr(my_flag_ptr, iris_rank, remote_rank, heap_bases)
            tl.atomic_add(remote_translated, 1, sem="release", scope="sys")

    for i in tl.static_range(0, world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            poll_ptr = flags_ptr + pid * world_size + i
            poll_local = _translate_ptr(poll_ptr, iris_rank, iris_rank, heap_bases)
            while tl.atomic_cas(poll_local, target, target, sem="acquire", scope="sys") < target:
                pass
