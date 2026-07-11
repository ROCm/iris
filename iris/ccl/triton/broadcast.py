# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernels for broadcast collective communication.

Pull-based broadcast with in-kernel barriers for graph capture safety.
"""

import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from iris.host.distributed.helpers import _translate_ptr
from iris.ccl.triton.all_reduce import _per_block_barrier


@triton.jit()
def broadcast_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    src_iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    start_flags_ptr,
    end_flags_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)

    _per_block_barrier(
        pid, start_flags_ptr, heap_bases,
        iris_rank, iris_rank, world_size, rank_start, rank_stride,
    )

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    total = num_m * num_n

    for tile in range(pid, total, COMM_SMS):
        pm = tile // num_n
        pn = tile % num_n
        rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * stride_m + rn[None, :] * stride_n

        if iris_rank == src_iris_rank:
            data = tl.load(input_ptr + off, mask=mask, other=0.0)
        else:
            data = iris.load(input_ptr + off, iris_rank, src_iris_rank, heap_bases, mask=mask)

        tl.store(output_ptr + off, data, mask=mask)

    _per_block_barrier(
        pid, end_flags_ptr, heap_bases,
        iris_rank, iris_rank, world_size, rank_start, rank_stride,
    )


class _BroadcastWorkspace:
    def __init__(self, ctx, world_size, max_blocks=16):
        needed = max_blocks * world_size
        self.start_flags = ctx.zeros((needed,), dtype=torch.int32)
        self.end_flags = ctx.zeros((needed,), dtype=torch.int32)


_workspace_cache = {}


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    src_rank,
    config,
):
    """Launch the Triton broadcast kernel with in-kernel barriers."""
    M, N = input_tensor.shape[:2]
    stride_m, stride_n = input_tensor.stride(0), input_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    src_iris_rank = rank_start + src_rank * rank_stride

    if "broadcast" not in _workspace_cache:
        _workspace_cache["broadcast"] = _BroadcastWorkspace(ctx, world_size, max_blocks=config.comm_sms)
    ws = _workspace_cache["broadcast"]

    iris_launch(
        broadcast_kernel,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
        stride_m,
        stride_n,
        heap_bases,
        rank_global,
        src_iris_rank,
        world_size,
        rank_start,
        rank_stride,
        ws.start_flags,
        ws.end_flags,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        num_stages=1,
        num_warps=4,
        waves_per_eu=0,
        algorithm="broadcast",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
