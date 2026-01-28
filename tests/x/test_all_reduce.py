# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for tile-level all-reduce primitives.
"""

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris
import iris.x


@triton.jit
def test_all_reduce_atomic_kernel(
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
    """Kernel that iterates over tiles and calls all_reduce_atomic for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, 1):  # Process all tiles
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        iris.x.all_reduce_atomic(
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


@triton.jit
def test_all_reduce_one_shot_kernel(
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
    """Kernel that iterates over tiles and calls all_reduce_one_shot for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, 1):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        iris.x.all_reduce_one_shot(
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


@triton.jit
def test_all_reduce_two_shot_kernel(
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
    """Kernel that iterates over tiles and calls all_reduce_two_shot for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, 1):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        iris.x.all_reduce_two_shot(
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


@triton.jit
def test_all_reduce_spinlock_kernel(
    input_ptr,
    output_ptr,
    locks_ptr,
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
    """Kernel that iterates over tiles and calls all_reduce_spinlock for each."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, 1):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        iris.x.all_reduce_spinlock(
            input_ptr,
            output_ptr,
            locks_ptr,
            tile_id,
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
    "variant",
    [
        "atomic",
        "one_shot",
        "two_shot",
        "spinlock",
    ],
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
def test_all_reduce(variant, dtype, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level all-reduce primitives by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch's all_reduce format: each rank has M x N data
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_reduce to get reference output
    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Prepare workspace if needed
    locks = None
    if variant == "spinlock":
        num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        total_tiles = num_pid_m * num_pid_n
        locks = shmem.zeros((total_tiles,), dtype=torch.int32)

    shmem.barrier()

    # Select kernel based on variant
    if variant == "atomic":
        kernel = test_all_reduce_atomic_kernel
    elif variant == "one_shot":
        kernel = test_all_reduce_one_shot_kernel
    elif variant == "two_shot":
        kernel = test_all_reduce_two_shot_kernel
    elif variant == "spinlock":
        kernel = test_all_reduce_spinlock_kernel
    else:
        pytest.fail(f"Unknown variant: {variant}")

    # Launch kernel
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    if variant == "spinlock":
        kernel[grid](
            iris_input_tensor,
            iris_output_tensor,
            locks,
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
    else:
        kernel[grid](
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
            f"Rank {rank}: Iris x.all_reduce_{variant} output doesn't match PyTorch's all_reduce"
        )
        
        # Verify the reduction is correct (sum of all ranks)
        expected_sum = sum(float(r + 1) for r in range(world_size))
        assert torch.allclose(iris_output_tensor, torch.full_like(iris_output_tensor, expected_sum), atol=atol), (
            f"Rank {rank}: Reduction result is incorrect, expected {expected_sum}"
        )
        
        if rank == 0:
            print(f"✓ All-reduce {variant} test passed: {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()

