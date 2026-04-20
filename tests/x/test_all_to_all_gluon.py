# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the Gluon tile-level all-to-all primitive (iris.x.all_to_all_gluon).
"""

import pytest
import torch
import torch.distributed as dist

# Try to import Gluon; skip all tests if not available.
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    import iris.experimental.iris_gluon as iris_gl
    import iris.x

    GLUON_AVAILABLE = hasattr(iris.x, "all_to_all_gluon")
except ImportError:
    GLUON_AVAILABLE = False


if GLUON_AVAILABLE:

    @gluon.jit
    def x_all_to_all_gluon_kernel(
        IrisDeviceCtx: gl.constexpr,
        context_tensor,
        input_ptr,
        output_ptr,
        M,
        N,
        N_per_rank: gl.constexpr,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        num_pid_n,
        cur_rank: gl.constexpr,
        world_size: gl.constexpr,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
    ):
        """Wrapper kernel that iterates over tiles and calls all_to_all_gluon."""
        pid = gl.program_id(0)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        iris.x.all_to_all_gluon(
            IrisDeviceCtx,
            context_tensor,
            input_ptr,
            output_ptr,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            pid_m,
            pid_n,
            N_per_rank,
            cur_rank,
            world_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
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
        (100, 100, 64, 64),  # Non-aligned dimensions
        (256, 384, 128, 128),  # Non-square
    ],
)
def test_all_to_all_gluon(dtype, atol, rtol, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """Test Gluon tile-level all-to-all by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8 GB
    shmem = iris_gl.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Build a reference tensor using PyTorch's dist.all_to_all.
    pytorch_input = torch.randn(M, N * world_size, dtype=dtype, device=f"cuda:{rank}")
    for r in range(world_size):
        pytorch_input[:, r * N : (r + 1) * N].fill_(float(r + 1))

    shmem.barrier()
    input_chunks = [chunk.contiguous() for chunk in torch.chunk(pytorch_input, world_size, dim=1)]
    output_chunks = [torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks)
    pytorch_output = torch.cat(output_chunks, dim=1)
    torch.cuda.synchronize()

    # Set up Iris Gluon tensors.
    iris_input = shmem.zeros((M, N * world_size), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = shmem.zeros((M, N * world_size), dtype=dtype)

    context_tensor = shmem.get_device_context()
    shmem.barrier()

    # Launch Gluon kernel — one program per tile.
    total_N = N * world_size
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (total_N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_tiles = num_pid_m * num_pid_n
    grid = (total_tiles,)

    x_all_to_all_gluon_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        iris_input,
        iris_output,
        M,
        total_N,
        N,  # N_per_rank
        iris_input.stride(0),
        iris_input.stride(1),
        iris_output.stride(0),
        iris_output.stride(1),
        num_pid_n,
        rank,
        world_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=4,
    )

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(iris_output - pytorch_output).max().item()

    try:
        assert torch.allclose(iris_output, pytorch_output, atol=atol, rtol=rtol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: iris.x.all_to_all_gluon output does not match PyTorch all_to_all"
        )

        # Verify each rank's received chunks contain the expected value.
        # After all-to-all, output[:, r*N:(r+1)*N] should hold rank r's data sent to
        # rank 'rank', which is rank r's chunk 'rank' filled with value (rank+1).
        for src_rank in range(world_size):
            chunk = iris_output[:, src_rank * N : (src_rank + 1) * N]
            expected_value = float(rank + 1)
            assert torch.allclose(chunk, torch.full_like(chunk, expected_value), atol=atol), (
                f"Rank {rank}: chunk from rank {src_rank} should have value {expected_value}"
            )

        if rank == 0:
            print(f"✓ all_to_all_gluon passed: {dtype}, M={M}, N={N}, blocks=({BLOCK_SIZE_M},{BLOCK_SIZE_N})")
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
