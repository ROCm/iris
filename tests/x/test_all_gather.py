# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for tile-level all-gather primitive.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x


@triton.jit
def x_all_gather_kernel(
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
    gather_dim: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_gather for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Create OOP objects for new API
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(
            output_ptr,
            M * world_size if gather_dim == 0 else M,
            N if gather_dim == 0 else N * world_size,
            stride_out_m,
            stride_out_n,
        )
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        iris.x.all_gather(tile, src_view, dst_view, gather_dim, ctx)


@pytest.mark.parametrize(
    "gather_dim",
    [0, 1],
)
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
def test_all_gather(gather_dim, dtype, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level all-gather primitive by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_gather format: each rank has M x N data
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_gather to get reference output
    pytorch_output_list = [torch.empty_like(pytorch_input_tensor) for _ in range(world_size)]
    shmem.barrier()
    dist.all_gather(pytorch_output_list, pytorch_input_tensor)

    if gather_dim == 0:
        # Gather along rows (M dimension)
        pytorch_output_tensor = torch.cat(pytorch_output_list, dim=0)  # Concatenate along dim 0
    else:
        # Gather along columns (N dimension)
        pytorch_output_tensor = torch.cat(pytorch_output_list, dim=1)  # Concatenate along dim 1

    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    if gather_dim == 0:
        iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)
    else:
        iris_output_tensor = shmem.zeros((M, world_size * N), dtype=dtype)

    shmem.barrier()

    # Launch kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    x_all_gather_kernel[grid](
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
        gather_dim,
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
            f"Rank {rank}: Iris x.all_gather output doesn't match PyTorch's all_gather"
        )

        # Verify each rank's data is in the correct location
        if gather_dim == 0:
            # Gathered along rows
            for r in range(world_size):
                start_row = r * M
                end_row = (r + 1) * M
                rank_data = iris_output_tensor[start_row:end_row, :]
                expected_value = float(r + 1)
                assert torch.allclose(rank_data, torch.full_like(rank_data, expected_value), atol=atol), (
                    f"Rank {rank}: Data from rank {r} not in correct location or has wrong value"
                )
        else:
            # Gathered along columns
            for r in range(world_size):
                start_col = r * N
                end_col = (r + 1) * N
                rank_data = iris_output_tensor[:, start_col:end_col]
                expected_value = float(r + 1)
                assert torch.allclose(rank_data, torch.full_like(rank_data, expected_value), atol=atol), (
                    f"Rank {rank}: Data from rank {r} not in correct location or has wrong value"
                )

        if rank == 0:
            dim_str = "rows" if gather_dim == 0 else "cols"
            print(
                f"✓ All-gather test passed ({dim_str}): {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})"
            )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@triton.jit
def x_all_gather_ctx_api_kernel(
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
    gather_dim: tl.constexpr,
):
    """Kernel using new ctx.all_gather() API."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Create OOP objects for new API
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
        src_view = iris.x.TensorView(input_ptr, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(
            output_ptr,
            M * world_size if gather_dim == 0 else M,
            N if gather_dim == 0 else N * world_size,
            stride_out_m,
            stride_out_n,
        )
        ctx = iris.x.DeviceContext(cur_rank, world_size, heap_bases)

        # NEW: Call collective on ctx directly
        ctx.all_gather(tile, src_view, dst_view, gather_dim)


@pytest.mark.parametrize("gather_dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("M, N, BLOCK_SIZE_M, BLOCK_SIZE_N", [(256, 128, 64, 64)])
def test_all_gather_ctx_api(gather_dim, dtype, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level all-gather using new ctx.all_gather() API."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_gather format: each rank has M x N data
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_gather to get reference output
    pytorch_output_list = [torch.empty_like(pytorch_input_tensor) for _ in range(world_size)]
    shmem.barrier()
    dist.all_gather(pytorch_output_list, pytorch_input_tensor)

    if gather_dim == 0:
        pytorch_output_tensor = torch.cat(pytorch_output_list, dim=0)
    else:
        pytorch_output_tensor = torch.cat(pytorch_output_list, dim=1)

    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    if gather_dim == 0:
        iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)
    else:
        iris_output_tensor = shmem.zeros((M, world_size * N), dtype=dtype)

    shmem.barrier()

    # Launch kernel using NEW ctx API
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    x_all_gather_ctx_api_kernel[grid](
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
        gather_dim,
    )

    torch.cuda.synchronize()
    shmem.barrier()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol, rtol=rtol), (
            f"Rank {rank}: ctx.all_gather() output doesn't match PyTorch's all_gather"
        )

        if rank == 0:
            dim_str = "rows" if gather_dim == 0 else "cols"
            print(f"✓ ctx.all_gather() test passed ({dim_str}): {dtype}, M={M}, N={N}")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
