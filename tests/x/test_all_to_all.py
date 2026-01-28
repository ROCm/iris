# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for tile-level all-to-all primitive.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x


@triton.jit
def test_all_to_all_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_to_all for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, 1):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        iris.x.all_to_all(
            input_ptr,
            output_ptr,
            pid_m,
            pid_n,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            cur_rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
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
def test_all_to_all(dtype, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level all-to-all primitive by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_to_all format: input is (M, N * world_size), output is (M, N * world_size)
    # Each rank sends chunk [:, rank*N : (rank+1)*N] to all ranks
    pytorch_input_tensor = torch.randn(M, N * world_size, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values: rank value in each rank's chunk
    for r in range(world_size):
        pytorch_input_tensor[:, r * N : (r + 1) * N].fill_(float(r + 1))

    # Run PyTorch's all_to_all to get reference output
    pytorch_output_tensor = torch.empty_like(pytorch_input_tensor)
    shmem.barrier()
    # PyTorch all_to_all: split input into chunks, send chunk i to rank i
    input_chunks = list(torch.chunk(pytorch_input_tensor, world_size, dim=1))
    output_chunks = list(torch.chunk(pytorch_output_tensor, world_size, dim=1))
    dist.all_to_all(output_chunks, input_chunks)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N * world_size), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N * world_size), dtype=dtype)

    shmem.barrier()

    # Launch kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    test_all_to_all_kernel[grid](
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
            f"Rank {rank}: Iris x.all_to_all output doesn't match PyTorch's all_to_all"
        )
        
        # Verify each rank's received chunks contain correct data
        for r in range(world_size):
            start_col = r * N
            end_col = (r + 1) * N
            chunk_data = iris_output_tensor[:, start_col:end_col]
            # Data in this chunk should be from rank r, which is filled with value (r+1)
            expected_value = float(r + 1)
            assert torch.allclose(chunk_data, torch.full_like(chunk_data, expected_value), atol=atol), (
                f"Rank {rank}: Data from rank {r} not in correct location or has wrong value"
            )
        
        if rank == 0:
            print(f"✓ All-to-all test passed: {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

