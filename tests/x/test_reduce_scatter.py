# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for tile-level reduce-scatter primitive.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x


@triton.jit
def test_x_reduce_scatter_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    total_tiles: tl.constexpr,
):
    """Kernel that iterates over tiles assigned to this rank and calls reduce_scatter for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Each rank processes tiles assigned to it (striding distribution)
    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    start_tile = cur_rank
    stride = world_size
    remaining = total_tiles - start_tile
    remaining = tl.maximum(remaining, 0)
    max_tile_offset = tl.cdiv(remaining, stride)

    for tile_offset in range(pid, max_tile_offset, 1):
        tile_id = start_tile + tile_offset * stride
        if tile_id < total_tiles:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            # Create OOP objects for new API
            tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
            src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
            dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
            ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

            iris.x.reduce_scatter(tile, src_view, dst_view, ctx)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, BLOCK_SIZE_M, BLOCK_SIZE_N",
    [
        (128, 64, 64, 32),  # Small
        (1024, 256, 128, 128),  # Medium
        (2048, 2048, 256, 256),  # Large
        (100, 100, 64, 64),  # Non-aligned dimensions
        (256, 384, 128, 128),  # Non-square
        (64, 32, 128, 128),  # Block size larger than dimensions
    ],
)
def test_reduce_scatter(dtype, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level reduce-scatter primitive by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's reduce_scatter format: each rank has M x N data
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's reduce_scatter to get reference output
    pytorch_output_tensor = torch.empty_like(pytorch_input_tensor)
    shmem.barrier()
    dist.reduce_scatter(pytorch_output_tensor, [pytorch_input_tensor], op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Launch kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = (total_tiles + world_size - 1) // world_size
    grid = (tiles_per_rank,)

    test_x_reduce_scatter_kernel[grid](
        iris_input_tensor,
        iris_output_tensor,
        M,
        N,
        iris_input_tensor.stride(0),
        iris_input_tensor.stride(1),
        iris_output_tensor.stride(0),
        iris_output_tensor.stride(1),
        shmem.get_heap_bases(),
        rank,
        world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        total_tiles,
    )

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        # Verify overall correctness
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol, rtol=rtol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris x.reduce_scatter output doesn't match PyTorch's reduce_scatter"
        )

        # Verify the reduction is correct (sum of all ranks)
        expected_sum = sum(float(r + 1) for r in range(world_size))
        assert torch.allclose(iris_output_tensor, torch.full_like(iris_output_tensor, expected_sum), atol=atol), (
            f"Rank {rank}: Reduction result is incorrect, expected {expected_sum}"
        )

        if rank == 0:
            print(f"✓ Reduce-scatter test passed: {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

