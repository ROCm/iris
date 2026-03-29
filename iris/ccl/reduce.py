# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Reduce collective communication primitive for Iris.

All-to-one reduction using all-pairs-read pattern:
root reads all remote ranks' data via iris.load and accumulates.
Non-root ranks are no-ops.
"""

from typing import Optional

import triton
import triton.language as tl
import iris
from .config import Config
from .utils import ReduceOp, extract_group_info


@triton.jit()
def _reduce_kernel(
    tensor_ptr,
    numel,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    dst_rank_in_group: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduce kernel: root reads all ranks' data and accumulates (SUM).

    Grid: (cdiv(numel, BLOCK_SIZE),)

    Root only:
    1. Load local tile
    2. For each remote rank: iris.load their tile, accumulate
    3. Store reduced result

    Non-root: no-op.
    """
    if group_rank == dst_rank_in_group:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        # Start with local data
        acc = tl.load(tensor_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Accumulate from all remote ranks
        for r in tl.static_range(world_size):
            if r != dst_rank_in_group:
                remote_global = rank_start + r * rank_stride
                remote_data = iris.load(
                    tensor_ptr + offsets,
                    iris_rank,
                    remote_global,
                    heap_bases,
                    mask=mask,
                )
                acc += remote_data.to(tl.float32)

        tl.store(tensor_ptr + offsets, acc.to(tensor_ptr.type.element_ty), mask=mask)


def reduce(
    tensor,
    shmem,
    dst=0,
    op=ReduceOp.SUM,
    group=None,
    async_op=False,
    config: Optional[Config] = None,
):
    """
    In-place reduce collective operation.

    All ranks contribute their data, and the rank identified by ``dst``
    receives the element-wise sum. After the operation, only the dst rank's
    tensor holds the reduced result; other ranks' tensors are unchanged.

    Args:
        tensor: Tensor on the symmetric heap. Modified in-place on dst rank.
        shmem: Iris shmem context.
        dst: Destination rank within the group (default: 0).
        op: Reduction operation. Currently only ReduceOp.SUM is supported.
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If False, performs a barrier at the end.
        config: Config instance with kernel parameters.
    """
    if op != ReduceOp.SUM:
        raise ValueError(
            f"Only ReduceOp.SUM is currently supported, got {op}. "
            "Support for other operations will be added in a future release."
        )

    if config is None:
        config = Config()

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, shmem)

    if dst < 0 or dst >= world_size:
        raise ValueError(f"dst rank {dst} out of range [0, {world_size})")

    numel = tensor.numel()
    if numel == 0:
        if not async_op:
            shmem.barrier(group=group)
        return

    # Flatten to 1D for simplicity
    tensor_flat = tensor.view(-1)

    heap_bases = shmem.get_heap_bases()

    block_size = config.block_size_m * config.block_size_n
    grid = ((numel + block_size - 1) // block_size,)

    # Device barrier: ensure all ranks have written their input before root reads.
    # Uses device-side barrier (CUDA-graph capturable, lower overhead than host barrier).
    shmem.device_barrier(group=group)

    _reduce_kernel[grid](
        tensor_flat,
        numel,
        heap_bases,
        rank_in_group,
        rank_global,
        world_size,
        dst,
        rank_start,
        rank_stride,
        block_size,
    )

    if not async_op:
        shmem.device_barrier(group=group)
