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
def x_reduce_scatter_kernel(
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
    context_tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel that iterates over tiles and calls reduce_scatter for each."""
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

        # Store to temp_buffer and signal ready
        temp_ptr = temp_buffer + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        tl.store(temp_ptr, local_data, mask=mask, cache_modifier=".wt")
        tl.debug_barrier()
        tl.atomic_xchg(locks + tile_id, 1, sem="release", scope="gpu")

        tile = iris.x.Tile(pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, local_data)
        src_view = iris.x.TensorView(temp_buffer, M, N, stride_in_m, stride_in_n)
        dst_view = iris.x.TensorView(output_ptr, M, N, stride_out_m, stride_out_n)
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)

        iris.x.reduce_scatter(tile, src_view, dst_view, locks, ctx)


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
        (128, 64, 64, 32),
        (256, 128, 64, 64),
        (512, 512, 128, 128),
    ],
)
def test_reduce_scatter(dtype, atol, rtol, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test tile-level reduce-scatter primitive."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    pytorch_input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    tiles_per_rank = total_tiles // world_size
    start_tile = rank * tiles_per_rank
    if rank == world_size - 1:
        tiles_per_rank = total_tiles - start_tile

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_temp_buffer = shmem.zeros((M, N), dtype=dtype)
    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    locks_tensor = shmem.zeros(total_tiles, dtype=torch.int32)

    shmem.barrier()

    grid = (total_tiles,)

    x_reduce_scatter_kernel[grid](
        iris_input_tensor,
        iris_temp_buffer,
        iris_output_tensor,
        locks_tensor,
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

    expected_sum = sum(float(r + 1) for r in range(world_size))

    try:
        for local_tile_idx in range(tiles_per_rank):
            tile_id = start_tile + local_tile_idx
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n

            m_start = pid_m * BLOCK_SIZE_M
            m_end = min(m_start + BLOCK_SIZE_M, M)
            n_start = pid_n * BLOCK_SIZE_N
            n_end = min(n_start + BLOCK_SIZE_N, N)

            tile_data = iris_output_tensor[m_start:m_end, n_start:n_end]
            expected_tile = torch.full_like(tile_data, expected_sum)

            assert torch.allclose(tile_data, expected_tile, atol=atol, rtol=rtol), (
                f"Rank {rank}, tile {tile_id} ({pid_m},{pid_n}): "
                f"Expected {expected_sum}, got max {tile_data.max().item()}, "
                f"min {tile_data.min().item()}"
            )

        if rank == 0:
            print(f"Reduce-scatter test passed: {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
