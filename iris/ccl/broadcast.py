# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Broadcast collective communication primitive for Iris.

One-to-all data replication using all-pairs-read pattern:
non-root ranks read root's data via iris.load, root is a no-op.
"""

from typing import Optional

import triton
import triton.language as tl
import torch
import iris
from .config import Config
from .utils import extract_group_info


@triton.jit()
def _broadcast_kernel(
    tensor_ptr,
    numel,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    src_rank_in_group: tl.constexpr,
    src_rank_global: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Broadcast kernel: non-root ranks read root's data via iris.load.

    Grid: (cdiv(numel, BLOCK_SIZE),)

    Non-root ranks: iris.load from root, tl.store locally.
    Root: no-op (data already in place).
    """
    if group_rank != src_rank_in_group:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        data = iris.load(
            tensor_ptr + offsets,
            iris_rank,
            src_rank_global,
            heap_bases,
            mask=mask,
        )
        tl.store(tensor_ptr + offsets, data, mask=mask)


def broadcast(
    tensor,
    shmem,
    src=0,
    group=None,
    async_op=False,
    config: Optional[Config] = None,
):
    """
    In-place broadcast collective operation.

    The rank identified by ``src`` broadcasts its data to all other ranks.
    After the operation, all ranks hold a copy of the source rank's data.

    Args:
        tensor: Tensor on the symmetric heap. Modified in-place.
        shmem: Iris shmem context.
        src: Source rank within the group (default: 0).
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If False, performs a barrier at the end.
        config: Config instance with kernel parameters.
    """
    if config is None:
        config = Config()

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    if src < 0 or src >= world_size:
        raise ValueError(f"src rank {src} out of range [0, {world_size})")

    numel = tensor.numel()
    if numel == 0:
        if not async_op:
            shmem.barrier(group=group)
        return

    # Flatten to 1D for simplicity
    tensor_flat = tensor.view(-1)

    src_rank_global = rank_start + src * rank_stride
    heap_bases = shmem.get_heap_bases()

    block_size = config.block_size_m * config.block_size_n
    grid = ((numel + block_size - 1) // block_size,)

    # Barrier: ensure src has written its data before others read
    shmem.barrier(group=group)

    _broadcast_kernel[grid](
        tensor_flat,
        numel,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        src,
        src_rank_global,
        rank_start,
        rank_stride,
        block_size,
    )

    if not async_op:
        shmem.barrier(group=group)
