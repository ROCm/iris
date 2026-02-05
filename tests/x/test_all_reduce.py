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
def x_all_reduce_atomic_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_atomic for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):  # Stride by grid size to avoid overlap
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Create Tile with loaded data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.all_reduce_atomic(tile, dst_view, ctx)


@triton.jit
def x_all_reduce_one_shot_kernel(
    input_ptr,
    temp_buffer,
    output_ptr,
    locks,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_one_shot for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data from input
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Store to temp_buffer (avoid race condition) and signal ready
        temp_ptr = temp_buffer + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(temp_ptr, local_data, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg
        tl.atomic_xchg(locks + tile_id, 1, sem="release", scope="gpu")  # Release ensures prior stores visible

        # Create Tile with data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.all_reduce_one_shot(tile, src_view, dst_view, locks, ctx)


@triton.jit
def x_all_reduce_two_shot_kernel(
    input_ptr,
    temp_buffer,
    output_ptr,
    locks,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_two_shot for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data from input
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Store to temp_buffer (avoid race condition) and signal ready
        temp_ptr = temp_buffer + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(temp_ptr, local_data, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()  # Ensures all stores are visible before the atomic_xchg
        tl.atomic_xchg(locks + tile_id, 1, sem="release", scope="gpu")  # Release ensures prior stores visible

        # Create Tile with data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.all_reduce_two_shot(tile, src_view, dst_view, locks, cur_rank, world_size, ctx)


@triton.jit
def x_all_reduce_spinlock_kernel(
    input_ptr,
    output_ptr,
    locks_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls all_reduce_spinlock for each."""
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, grid_size):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Load local tile data
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        src_ptr = input_ptr + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        local_data = tl.load(src_ptr, mask=mask, other=0.0)

        # Create Tile with loaded data and views
        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.all_reduce_spinlock(tile, dst_view, locks_ptr, ctx)


@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        "one_shot",
        "two_shot",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-3, 1e-3),
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-3, 1e-3),
    ],
)
@pytest.mark.parametrize(
    "M, N, BLOCK_SIZE_M, BLOCK_SIZE_N",
    [
        (128, 64, 64, 32),  # Small
        (1024, 256, 128, 128),  # Medium
        (2048, 2048, 256, 256),  # Large
        # (100, 100, 64, 64),  # Non-aligned dimensions - DISABLED: other=0.0 not supported
        # (256, 384, 128, 128),  # Non-square - DISABLED: other=0.0 not supported
        # (64, 32, 128, 128),  # Block size larger than dimensions - DISABLED: other=0.0 not supported
    ],
)
def test_all_reduce(variant, dtype, atol, rtol, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
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

    # Prepare workspace if needed (locks + temp_buffer for one_shot/two_shot)
    locks = None
    temp_buffer = None
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n

    if variant in ["spinlock", "one_shot", "two_shot"]:
        locks = shmem.zeros((total_tiles,), dtype=torch.int32)

    if variant in ["one_shot", "two_shot"]:
        temp_buffer = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Select kernel based on variant
    if variant == "atomic":
        kernel = x_all_reduce_atomic_kernel
    elif variant == "one_shot":
        kernel = x_all_reduce_one_shot_kernel
    elif variant == "two_shot":
        kernel = x_all_reduce_two_shot_kernel
    elif variant == "spinlock":
        kernel = x_all_reduce_spinlock_kernel
    else:
        pytest.fail(f"Unknown variant: {variant}")

    # Launch kernel
    grid = (total_tiles,)

    if variant in ["one_shot", "two_shot"]:
        kernel[grid](
            iris_input_tensor,
            temp_buffer,
            iris_output_tensor,
            locks,
            M,
            N,
            iris_input_tensor.stride(0),
            iris_input_tensor.stride(1),
            iris_output_tensor.stride(0),
            iris_output_tensor.stride(1),
            shmem.get_device_context(),
            rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    elif variant == "spinlock":
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
            shmem.get_device_context(),
            rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    else:  # atomic
        kernel[grid](
            iris_input_tensor,
            iris_output_tensor,
            M,
            N,
            iris_input_tensor.stride(0),
            iris_input_tensor.stride(1),
            iris_output_tensor.stride(0),
            iris_output_tensor.stride(1),
            shmem.get_device_context(),
            rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
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
