# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_gather API.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import tritonblas
import iris
import os


def _param_shapes():
    if "IRIS_TEST_M" in os.environ:
        return [
            (
                int(os.environ["IRIS_TEST_M"]),
                int(os.environ["IRIS_TEST_N"]),
                int(os.environ["IRIS_TEST_K"]),
            )
        ]
    return [
        (64, 64, 32),
        (512, 256, 512),
        (1024, 2048, 1024),
    ]


def _heap_size() -> int:
    return int(os.environ.get("IRIS_TEST_HEAP_SIZE", 1 << 34))


def _full_validation_threshold_bytes() -> int:
    return int(os.environ.get("IRIS_TEST_FULL_VALIDATION_THRESHOLD_BYTES", 2 << 30))


def _should_stream_validation(rows_per_rank: int, n: int, dtype: torch.dtype) -> bool:
    element_size = torch.tensor([], dtype=dtype).element_size()
    local_ref_bytes = rows_per_rank * n * element_size
    return local_ref_bytes > _full_validation_threshold_bytes()


def _validation_rows_per_chunk(rows_per_rank: int, n: int, dtype: torch.dtype) -> int:
    if "IRIS_TEST_VALIDATION_ROWS_PER_CHUNK" in os.environ:
        return max(1, min(rows_per_rank, int(os.environ["IRIS_TEST_VALIDATION_ROWS_PER_CHUNK"])))
    element_size = torch.tensor([], dtype=dtype).element_size()
    target_bytes = 2 << 30
    rows_per_chunk = max(1, target_bytes // max(1, n * element_size))
    return max(1, min(rows_per_rank, rows_per_chunk))


def _validation_cols_per_chunk(rows_per_chunk: int, n: int, dtype: torch.dtype) -> int:
    if "IRIS_TEST_VALIDATION_COLS_PER_CHUNK" in os.environ:
        return max(1, min(n, int(os.environ["IRIS_TEST_VALIDATION_COLS_PER_CHUNK"])))
    element_size = torch.tensor([], dtype=dtype).element_size()
    target_bytes = 128 << 20
    cols_per_chunk = max(1, target_bytes // max(1, rows_per_chunk * element_size * 4))
    return max(1, min(n, cols_per_chunk))


def _assert_close_chunked(output_chunk, ref_chunk, atol, rtol, src_rank, row_start):
    cols_per_chunk = _validation_cols_per_chunk(output_chunk.shape[0], output_chunk.shape[1], output_chunk.dtype)

    for col_start in range(0, output_chunk.shape[1], cols_per_chunk):
        col_end = min(col_start + cols_per_chunk, output_chunk.shape[1])
        output_slice = output_chunk[:, col_start:col_end]
        ref_slice = ref_chunk[:, col_start:col_end]
        abs_diff = torch.abs(output_slice - ref_slice)
        tolerance = atol + rtol * torch.abs(ref_slice)
        mismatch = abs_diff > tolerance

        if torch.any(mismatch):
            mismatch_idx = torch.nonzero(mismatch, as_tuple=False)[0]
            local_row = int(mismatch_idx[0].item())
            local_col = int(mismatch_idx[1].item())
            global_row = row_start + local_row
            global_col = col_start + local_col
            max_diff = torch.max(abs_diff).item()
            output_val = output_slice[local_row, local_col].item()
            ref_val = ref_slice[local_row, local_col].item()
            pytest.fail(
                f"Mismatch in gathered rows from src_rank={src_rank} at row={global_row}, col={global_col}: "
                f"output={output_val}, ref={ref_val}, max_diff={max_diff}, expected within atol={atol}, rtol={rtol}\n"
                f"Rank validation failed for shmem.ops.matmul_all_gather"
            )


def _assert_gathered_rows_match_dense(output, local_ref, rank, world_size, atol, rtol):
    recv_chunk = None
    rows_per_rank = local_ref.shape[0]

    for src_rank in range(world_size):
        if rank == src_rank:
            ref_chunk = local_ref
        else:
            if recv_chunk is None:
                recv_chunk = torch.empty_like(local_ref)
            ref_chunk = recv_chunk

        dist.broadcast(ref_chunk, src=src_rank)
        row_start = src_rank * rows_per_rank
        row_end = row_start + rows_per_rank
        output_chunk = output[row_start:row_end]

        if not torch.allclose(output_chunk, ref_chunk, atol=atol, rtol=rtol):
            max_diff = torch.max(torch.abs(output_chunk - ref_chunk)).item()
            pytest.fail(
                f"Max difference in gathered rows from src_rank={src_rank}: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: shmem.ops.matmul_all_gather output doesn't match reference"
            )


def _assert_gathered_rows_match_streamed(output, A_local, B, rank, world_size, atol, rtol):
    rows_per_rank = A_local.shape[0]
    rows_per_chunk = _validation_rows_per_chunk(rows_per_rank, output.shape[1], output.dtype)

    for src_rank in range(world_size):
        for local_row_start in range(0, rows_per_rank, rows_per_chunk):
            local_row_end = min(local_row_start + rows_per_chunk, rows_per_rank)
            chunk_rows = local_row_end - local_row_start

            if rank == src_rank:
                ref_chunk = torch.matmul(A_local[local_row_start:local_row_end], B)
            else:
                ref_chunk = torch.empty((chunk_rows, output.shape[1]), dtype=output.dtype, device=output.device)

            dist.broadcast(ref_chunk, src=src_rank)
            global_row_start = src_rank * rows_per_rank + local_row_start
            global_row_end = global_row_start + chunk_rows
            output_chunk = output[global_row_start:global_row_end]
            _assert_close_chunked(output_chunk, ref_chunk, atol, rtol, src_rank, global_row_start)


def _assert_gathered_rows_match(output, A_local, B, rank, world_size, atol, rtol):
    if _should_stream_validation(A_local.shape[0], output.shape[1], output.dtype):
        _assert_gathered_rows_match_streamed(output, A_local, B, rank, world_size, atol, rtol)
        return

    local_ref = torch.matmul(A_local, B)
    torch.cuda.synchronize()
    _assert_gathered_rows_match_dense(output, local_ref, rank, world_size, atol, rtol)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.5, 0.01),
        # (torch.float32, 0.5, 0.01),  # disabled: Triton AMD backend LLVM unrealized_conversion_cast
        (torch.bfloat16, 0.5, 0.01),
    ],
)
@pytest.mark.parametrize(
    "M, N, K",
    _param_shapes(),
)
def test_matmul_all_gather(dtype, atol, rtol, M, N, K):
    """Test matmul_all_gather using shmem.ops API with proper config."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # M must be divisible by world_size for row-wise sharding
    if M % world_size != 0:
        pytest.skip(f"M={M} not divisible by world_size={world_size}")

    M_local = M // world_size

    # Skip if problem size is too small for world_size
    # With default or custom configs, we need at least one tile per rank
    min_block_size = 32  # Smallest block size we use
    if M_local < min_block_size:
        pytest.skip(f"M_local={M_local} too small for world_size={world_size} (need >= {min_block_size})")
    if K < min_block_size:
        pytest.skip(f"K={K} too small (need >= {min_block_size})")
    if N < min_block_size:
        pytest.skip(f"N={N} too small (need >= {min_block_size})")

    # Create shmem tensors directly
    A_local = shmem.randn((M_local, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Use appropriate block sizes based on problem size
    from iris.ops.config import FusedConfig

    # Select config based on actual problem dimensions
    # Ensure block sizes don't exceed actual dimensions
    if M_local <= 64 or K <= 64 or N <= 64:
        # Small problems - use 32x32x32 blocks
        config = FusedConfig(block_size_m=32, block_size_n=32, block_size_k=32)
    elif M_local <= 128 or K <= 128 or N <= 128:
        # Medium problems - use 64x64x32 blocks
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    elif dtype == torch.float32:
        # Larger problems with fp32 - use 128x128x64 blocks
        config = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=64)
    else:
        # Larger problems with fp16/bf16 - use 128x128x64 blocks
        config = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=64)

    # Validate config against problem size
    if config is not None:
        assert M_local >= config.block_size_m, f"M_local ({M_local}) must be >= block_size_m ({config.block_size_m})"
        assert K >= config.block_size_k, f"K ({K}) must be >= block_size_k ({config.block_size_k})"
        assert N >= config.block_size_n, f"N ({N}) must be >= block_size_n ({config.block_size_n})"

    # Use shmem.ops API with proper config
    shmem.ops.matmul_all_gather(output, A_local, B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    _assert_gathered_rows_match(output, A_local, B, rank, world_size, atol, rtol)

    if rank == 0:
        print(f"✓ matmul_all_gather test passed: {dtype}, M={M}, N={N}, K={K}")

    shmem.barrier()
    del output
    del B
    del A_local
    del shmem
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.5, 0.01),
        (torch.bfloat16, 0.5, 0.01),
    ],
)
@pytest.mark.parametrize(
    "M, N, K",
    _param_shapes(),
)
def test_tritonblas_rccl_matmul_all_gather(dtype, atol, rtol, M, N, K):
    """Test tritonBLAS matmul + RCCL all_gather against a dense local reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if M % world_size != 0:
        pytest.skip(f"M={M} not divisible by world_size={world_size}")

    M_local = M // world_size
    min_block_size = 32
    if M_local < min_block_size:
        pytest.skip(f"M_local={M_local} too small for world_size={world_size} (need >= {min_block_size})")
    if K < min_block_size:
        pytest.skip(f"K={K} too small (need >= {min_block_size})")
    if N < min_block_size:
        pytest.skip(f"N={N} too small (need >= {min_block_size})")

    torch.manual_seed(123 + rank)
    A_local = torch.randn((M_local, K), device=f"cuda:{rank}", dtype=dtype)
    torch.manual_seed(456)
    B = torch.randn((K, N), device=f"cuda:{rank}", dtype=dtype)
    C_local = shmem.zeros((M_local, N), dtype=dtype)
    output = torch.empty((M, N), device=f"cuda:{rank}", dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(
        M_local,
        N,
        K,
        A_local.dtype,
        B.dtype,
        C_local.dtype,
        A_local.device,
    )
    config = tritonblas.matmul_preamble(selector)

    shmem.barrier()
    tritonblas.matmul_lt(A_local, B, C_local, selector, config)
    dist.all_gather_into_tensor(output, C_local)

    torch.cuda.synchronize()
    shmem.barrier()

    _assert_gathered_rows_match(output, A_local, B, rank, world_size, atol, rtol)
