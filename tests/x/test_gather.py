#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for iris.x.gather primitive (single-rank gather)."""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x


@triton.jit
def gather_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor,
    cur_rank: tl.constexpr,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Test kernel that uses gather to pull a single tile from source_rank."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Create tile and views
        tile = iris.x.TileView(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        # Use gather to pull tile from source_rank
        data = iris.x.gather(tile, src_view, source_rank, ctx)

        # Store to output
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = rm < M
        mask_n = rn < N
        mask = mask_m[:, None] & mask_n[None, :]
        out_ptr = output_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        tl.store(out_ptr, data, mask=mask)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-3, 1e-3),
        (torch.float32, 1e-5, 1e-5),
    ],
)
@pytest.mark.parametrize("M, N, BLOCK_SIZE_M, BLOCK_SIZE_N", [(256, 128, 64, 64)])
def test_gather_from_specific_rank(dtype, atol, rtol, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test gather primitive pulling from a specific rank."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if world_size < 2:
        pytest.skip("Need at least 2 ranks")

    # Each rank creates unique input data
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    output_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")

    # Allocate in shmem
    shmem_input = shmem.zeros(M, N, dtype=dtype)
    shmem_output = shmem.zeros(M, N, dtype=dtype)
    shmem_input.copy_(input_tensor)

    shmem.barrier()

    # Each rank gathers from rank 0
    source_rank = 0
    grid = (64,)

    gather_kernel[grid](
        shmem_input,
        shmem_output,
        M,
        N,
        shmem_input.stride(0),
        shmem_input.stride(1),
        shmem_output.stride(0),
        shmem_output.stride(1),
        shmem.get_device_context(),
        rank,
        source_rank,
        world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    shmem.barrier()
    output_tensor.copy_(shmem_output)
    torch.cuda.synchronize()

    torch.manual_seed(42 + source_rank)
    expected = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")

    assert torch.allclose(output_tensor, expected, atol=atol, rtol=rtol), (
        f"Rank {rank}: gather from rank {source_rank} failed"
    )


@triton.jit
def gather_accumulate_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Test kernel that gathers from all ranks and accumulates (like all-reduce sum)."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        tile = iris.x.TileView(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        # Accumulate data from all ranks
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for source_rank in range(world_size):
            data = iris.x.gather(tile, src_view, source_rank, ctx)
            acc += data

        # Store accumulated result
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = rm < M
        mask_n = rn < N
        mask = mask_m[:, None] & mask_n[None, :]
        out_ptr = output_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        result = acc.to(output_ptr.type.element_ty)
        tl.store(out_ptr, result, mask=mask)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.float32, 1e-5, 1e-5),
    ],
)
@pytest.mark.parametrize("M, N, BLOCK_SIZE_M, BLOCK_SIZE_N", [(256, 128, 64, 64)])
def test_gather_accumulate_pattern(dtype, atol, rtol, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test gather used in accumulation pattern (like all-reduce sum)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank creates input with value = rank
    input_tensor = torch.full((M, N), float(rank), dtype=dtype, device=f"cuda:{rank}")
    output_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")

    # Allocate in shmem
    shmem_input = shmem.zeros(M, N, dtype=dtype)
    shmem_output = shmem.zeros(M, N, dtype=dtype)
    shmem_input.copy_(input_tensor)

    shmem.barrier()

    # Gather and accumulate from all ranks
    grid = (64,)
    gather_accumulate_kernel[grid](
        shmem_input,
        shmem_output,
        M,
        N,
        shmem_input.stride(0),
        shmem_input.stride(1),
        shmem_output.stride(0),
        shmem_output.stride(1),
        shmem.get_device_context(),
        rank,
        world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    shmem.barrier()
    output_tensor.copy_(shmem_output)
    torch.cuda.synchronize()

    expected_sum = sum(range(world_size))
    expected = torch.full((M, N), float(expected_sum), dtype=dtype, device=f"cuda:{rank}")

    assert torch.allclose(output_tensor, expected, atol=atol, rtol=rtol), (
        f"Rank {rank}: gather accumulate pattern failed"
    )
