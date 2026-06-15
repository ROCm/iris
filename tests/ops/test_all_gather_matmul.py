# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for fused all_gather + matmul operations.

Each rank has A_sharded (M x K_local), B is replicated.
The operation gathers A from all ranks and computes C = A_gathered @ B.
Covers both the baseline pull kernel and the HBM-buffered kernel.
"""

import pytest
import torch
import torch.distributed as dist
import tritonblas

import iris
from iris.ops.all_gather_matmul_hbm_buffer import (
    _auto_config,
    _CHAMPION_CONFIGS,
    all_gather_matmul_hbm_buffer,
    all_gather_matmul_hbm_buffer_preamble,
)
from iris.ops.config import FusedConfig


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Fixture to clean up GPU memory before and after each test."""
    # Cleanup before test starts
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    yield  # Run the test
    # Cleanup after test completes (pass or fail)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _param_shapes():
    if "IRIS_TEST_M" in os.environ:
        return [
            (
                int(os.environ["IRIS_TEST_M"]),
                int(os.environ["IRIS_TEST_K_LOCAL"]),
                int(os.environ["IRIS_TEST_N"]),
            )
        ]
    return [
        (128, 32, 64),
        (256, 64, 128),
    ]


def _heap_size() -> int:
    return int(os.environ.get("IRIS_TEST_HEAP_SIZE", 1 << 34))


def _full_validation_threshold_bytes() -> int:
    return int(os.environ.get("IRIS_TEST_FULL_VALIDATION_THRESHOLD_BYTES", 2 << 30))


def _validation_mode() -> str:
    return os.environ.get("IRIS_TEST_VALIDATION_MODE", "auto").lower()


def _validation_tile_rows(M: int) -> int:
    return max(1, min(M, int(os.environ.get("IRIS_TEST_VALIDATION_TILE_ROWS", 128))))


def _validation_tile_cols(N: int) -> int:
    return max(1, min(N, int(os.environ.get("IRIS_TEST_VALIDATION_TILE_COLS", 256))))


def _validation_num_row_tiles() -> int:
    return max(1, int(os.environ.get("IRIS_TEST_VALIDATION_NUM_ROW_TILES", 4)))


def _validation_num_col_tiles() -> int:
    return max(1, int(os.environ.get("IRIS_TEST_VALIDATION_NUM_COL_TILES", 4)))


def _sample_starts(size: int, tile_size: int, num_tiles: int) -> list[int]:
    if size <= tile_size:
        return [0]

    max_start = size - tile_size
    if num_tiles == 1:
        return [0]

    starts = {0, max_start}
    for tile_idx in range(1, num_tiles - 1):
        starts.add((max_start * tile_idx) // (num_tiles - 1))
    return sorted(starts)


def _should_sample_validation(M: int, N: int, dtype: torch.dtype) -> bool:
    mode = _validation_mode()
    if mode == "full":
        return False
    if mode == "sampled":
        return True
    if mode != "auto":
        raise ValueError(f"Unknown IRIS_TEST_VALIDATION_MODE={mode!r}; expected auto, full, or sampled")

    element_size = torch.tensor([], dtype=dtype).element_size()
    return M * N * element_size > _full_validation_threshold_bytes()


def _make_inputs(rank, world_size, M, K_local, N, dtype):
    device = f"cuda:{rank}"
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=device)

    torch.manual_seed(123)
    B = torch.randn(K, N, dtype=dtype, device=device)

    return A_sharded, B


def _make_full_reference(A_sharded, B, world_size):
    """Build a full torch reference output for small all_gather + matmul cases."""
    A_gathered_list = [torch.empty_like(A_sharded) for _ in range(world_size)]
    dist.all_gather(A_gathered_list, A_sharded)
    A_gathered_ref = torch.cat(A_gathered_list, dim=1)
    ref_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()
    return ref_output


def _assert_close_tile(output_tile, ref_tile, atol, rtol, rank, row_start, col_start, context):
    close = torch.isclose(output_tile, ref_tile, atol=atol, rtol=rtol)
    if torch.all(close):
        return

    mismatch_idx = torch.nonzero(~close, as_tuple=False)[0]
    local_row = int(mismatch_idx[0].item())
    local_col = int(mismatch_idx[1].item())
    global_row = row_start + local_row
    global_col = col_start + local_col
    abs_diff = torch.abs(output_tile - ref_tile)
    max_diff = torch.nan_to_num(abs_diff, nan=float("inf")).max().item()
    output_val = output_tile[local_row, local_col].item()
    ref_val = ref_tile[local_row, local_col].item()
    pytest.fail(
        f"Rank {rank}: sampled validation mismatch in {context} at row={global_row}, col={global_col}: "
        f"output={output_val}, ref={ref_val}, max_diff={max_diff}, expected within atol={atol}, rtol={rtol}"
    )


def _assert_full_output_matches(output, ref_output, atol, rtol, rank, context):
    if torch.allclose(output, ref_output, atol=atol, rtol=rtol):
        return

    max_diff = (output - ref_output).abs().max().item()
    pytest.fail(f"Rank {rank}: Max diff {max_diff}, expected < {atol} ({context})")


def _assert_sampled_output_matches(output, A_sharded, B, rank, world_size, atol, rtol, context, bias=None):
    M, N = output.shape
    rows_per_tile = _validation_tile_rows(M)
    cols_per_tile = _validation_tile_cols(N)
    row_starts = _sample_starts(M, rows_per_tile, _validation_num_row_tiles())
    col_starts = _sample_starts(N, cols_per_tile, _validation_num_col_tiles())

    for row_start in row_starts:
        row_end = min(row_start + rows_per_tile, M)
        local_a = A_sharded[row_start:row_end].contiguous()
        gathered_a_parts = [torch.empty_like(local_a) for _ in range(world_size)]
        dist.all_gather(gathered_a_parts, local_a)
        gathered_a_rows = torch.cat(gathered_a_parts, dim=1)

        for col_start in col_starts:
            col_end = min(col_start + cols_per_tile, N)
            ref_tile = torch.matmul(gathered_a_rows, B[:, col_start:col_end])
            if bias is not None:
                ref_tile = ref_tile + bias[row_start:row_end, None]
            output_tile = output[row_start:row_end, col_start:col_end]
            _assert_close_tile(output_tile, ref_tile, atol, rtol, rank, row_start, col_start, context)

        del gathered_a_rows, gathered_a_parts, local_a

    torch.cuda.synchronize()


def _assert_output_matches_reference(output, A_sharded, B, rank, world_size, atol, rtol, context, bias=None):
    if _should_sample_validation(output.shape[0], output.shape[1], output.dtype):
        _assert_sampled_output_matches(output, A_sharded, B, rank, world_size, atol, rtol, context, bias=bias)
        return

    ref_output = _make_full_reference(A_sharded, B, world_size)
    if bias is not None:
        ref_output = ref_output + bias[:, None]
    _assert_full_output_matches(output, ref_output, atol, rtol, rank, context)


def _hbm_buffer_test_config(M: int, K_local: int, N: int) -> FusedConfig | None:
    if M <= 256 or K_local <= 64 or N <= 128:
        return FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    return None


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    _param_shapes(),
)
def test_all_gather_matmul_baseline(dtype, atol, rtol, M, K_local, N):
    """Test baseline all_gather_matmul against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    min_block_size = 32
    if M < min_block_size or K_local < min_block_size or N < min_block_size:
        pytest.skip(f"Problem too small for min block size {min_block_size}")

    A_sharded, B = _make_inputs(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = (
        FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
        if M <= 256 or K_local <= 64 or N <= 128
        else FusedConfig()
    )

    assert M >= config.block_size_m
    assert K_local >= config.block_size_k
    assert N >= config.block_size_n

    ctx.ops.all_gather_matmul(output, A_sharded_shmem, B_shmem, config=config)

    torch.cuda.synchronize()
    ctx.barrier()

    _assert_output_matches_reference(
        output,
        A_sharded,
        B,
        rank,
        world_size,
        atol,
        rtol,
        "all_gather_matmul_baseline",
    )

    # Clean up to prevent OOM in subsequent tests
    del A_sharded, B, A_sharded_shmem, B_shmem, output
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    _param_shapes(),
)
def test_tritonblas_rccl_all_gather_matmul(dtype, atol, rtol, M, K_local, N):
    """Test RCCL all_gather + tritonBLAS matmul against torch reference."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")

    K = K_local * world_size
    A_sharded, B = _make_inputs(rank, world_size, M, K_local, N, dtype)

    A_gathered_parts = [torch.empty((M, K_local), dtype=dtype, device=device) for _ in range(world_size)]
    output = ctx.zeros((M, N), dtype=dtype)
    selector = tritonblas.OrigamiMatmulSelector(
        M,
        N,
        K,
        A_sharded.dtype,
        B.dtype,
        output.dtype,
        device,
    )
    config = tritonblas.matmul_preamble(selector)

    dist.all_gather(A_gathered_parts, A_sharded)
    A_gathered = torch.cat(A_gathered_parts, dim=1)
    tritonblas.matmul_lt(A_gathered, B, output, selector, config)

    torch.cuda.synchronize()

    _assert_output_matches_reference(
        output,
        A_sharded,
        B,
        rank,
        world_size,
        atol,
        rtol,
        f"tritonblas+rccl, M={M}, K_local={K_local}, N={N}",
    )

    # Clean up to prevent OOM in subsequent tests
    del A_sharded, B, output, A_gathered_parts, A_gathered
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    _param_shapes(),
)
@pytest.mark.parametrize(
    "staged_a_layout",
    [
        "k_contiguous",
        "m_contiguous",
    ],
)
def test_all_gather_matmul_hbm_buffer(dtype, atol, rtol, M, K_local, N, staged_a_layout):
    """Test all_gather_matmul_hbm_buffer against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B = _make_inputs(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = _hbm_buffer_test_config(M, K_local, N)

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, staged_a_layout=staged_a_layout
    )

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        staged_a_layout=staged_a_layout,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    _assert_output_matches_reference(
        output,
        A_sharded,
        B,
        rank,
        world_size,
        atol,
        rtol,
        f"staged_a_layout={staged_a_layout}, M={M}, K_local={K_local}, N={N}",
    )

    # Clean up to prevent OOM in subsequent tests
    del A_sharded, B, A_sharded_shmem, B_shmem, output, workspace
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    _param_shapes(),
)
def test_all_gather_matmul_hbm_buffer_with_bias(dtype, atol, rtol, M, K_local, N):
    """Test all_gather_matmul_hbm_buffer with a bias vector."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B = _make_inputs(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    torch.manual_seed(77)
    bias = torch.randn(M, dtype=dtype, device=device)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    bias_shmem = ctx.zeros((M,), dtype=dtype)
    bias_shmem.copy_(bias)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = _hbm_buffer_test_config(M, K_local, N)

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        bias=bias_shmem,
        config=config,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    _assert_output_matches_reference(
        output,
        A_sharded,
        B,
        rank,
        world_size,
        atol,
        rtol,
        "with bias",
        bias=bias,
    )

    # Clean up to prevent OOM in subsequent tests
    del A_sharded, B, bias, A_sharded_shmem, B_shmem, bias_shmem, output
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
        (256, 64, 128),
        (512, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "staged_a_layout",
    [
        "k_contiguous",
        "m_contiguous",
    ],
)
def test_all_gather_matmul_hbm_buffer(dtype, atol, rtol, M, K_local, N, staged_a_layout):
    """Test all_gather_matmul_hbm_buffer against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, staged_a_layout=staged_a_layout, k_per_flag=k_per_flag
    )

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        k_per_flag=k_per_flag,
        staged_a_layout=staged_a_layout,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} "
        f"(staged_a_layout={staged_a_layout}, M={M}, K_local={K_local}, N={N})"
    )


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M,K_local,N",
    [
        (128, 32, 64),
    ],
)
def test_all_gather_matmul_hbm_buffer_with_bias(dtype, atol, rtol, M, K_local, N):
    """Test all_gather_matmul_hbm_buffer with a bias vector."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    K = K_local * world_size

    A_sharded, B, ref_output_no_bias = _make_reference(rank, world_size, M, K_local, N, dtype)
    device = f"cuda:{rank}"

    torch.manual_seed(77)
    bias = torch.randn(M, dtype=dtype, device=device)
    ref_output = ref_output_no_bias + bias[:, None]

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    bias_shmem = ctx.zeros((M,), dtype=dtype)
    bias_shmem.copy_(bias)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)

    # k_per_flag must divide num_k_blocks = K // block_size_k; use 1 for small shapes
    num_k_blocks = K // config.block_size_k
    k_per_flag = 1
    while k_per_flag * 2 <= 8 and num_k_blocks % (k_per_flag * 2) == 0:
        k_per_flag *= 2

    all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        bias=bias_shmem,
        config=config,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (with bias)"
    )


def test_all_gather_matmul_hbm_buffer_auto_workspace():
    """Test all_gather_matmul_hbm_buffer with workspace=None (auto preamble)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    # workspace=None triggers automatic preamble inside the kernel function
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=None,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert ws is not None, "all_gather_matmul_hbm_buffer should return workspace"
    assert ws.aux_buffer is not None, "Workspace aux_buffer should be allocated"
    assert ws.locks is not None, "Workspace locks should be allocated"

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} (auto workspace)"
    )


def test_all_gather_matmul_hbm_buffer_workspace_reuse():
    """Test that workspace can be reused across multiple kernel calls."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2

    K = K_local * world_size
    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output1 = ctx.zeros((M, N), dtype=dtype)
    output2 = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    workspace = all_gather_matmul_hbm_buffer_preamble(
        ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag
    )

    # First call
    all_gather_matmul_hbm_buffer(
        ctx, output1, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    # Second call reusing workspace
    all_gather_matmul_hbm_buffer(
        ctx, output2, A_sharded_shmem, B_shmem, config=config, workspace=workspace, k_per_flag=k_per_flag, trace=False
    )
    torch.cuda.synchronize()
    ctx.barrier()

    max_diff1 = (output1 - ref_output).abs().max().item()
    max_diff2 = (output2 - ref_output).abs().max().item()
    assert torch.allclose(output1, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: First call max diff {max_diff1}, expected < {atol}"
    )
    assert torch.allclose(output2, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Second call (workspace reuse) max diff {max_diff2}, expected < {atol}"
    )
    assert torch.allclose(output1, output2), "Both calls should produce identical results"


def test_all_gather_matmul_hbm_buffer_trace():
    """Test that trace_data is None when trace=False (default)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    M, K_local, N = 128, 32, 64
    dtype = torch.float16

    K = K_local * world_size
    A_sharded, B, _ = _make_reference(rank, world_size, M, K_local, N, dtype)

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32)
    k_per_flag = 1

    ws = all_gather_matmul_hbm_buffer_preamble(ctx, A_sharded_shmem, B_shmem, config=config, k_per_flag=k_per_flag)

    # With trace=False, trace_data should not be populated
    ws = all_gather_matmul_hbm_buffer(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=ws,
        k_per_flag=k_per_flag,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    assert not hasattr(ws, "trace_data") or ws.trace_data is None, (
        # FusedWorkspace is a dataclass; trace_data is set only when trace=True.
        # Both conditions handle the case where the attribute is absent (fresh workspace)
        # or explicitly set to None (workspace reused from a previous trace=False call).
        "trace_data should not be populated when trace=False"
    )


# ──────────────────────────────────────────────────────────────────────
# Unit tests for _auto_config (no distributed context required)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K, world_size",
    [
        (1024, 256, 1024, 8),
        (4096, 3584, 8192, 8),
        (8192, 8192, 16384, 8),
        (16384, 3584, 8192, 4),
        (256, 256, 512, 2),
    ],
)
def test_auto_config_heuristic_validity(M, N, K, world_size):
    """Verify _auto_config returns valid configs where k_per_flag divides K//block_k."""
    config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size)

    assert config.block_size_m > 0
    assert config.block_size_n > 0
    assert config.block_size_k > 0

    num_k_blocks = K // config.block_size_k
    assert num_k_blocks % kpf == 0, (
        f"k_per_flag={kpf} does not divide num_k_blocks={num_k_blocks} for M={M},N={N},K={K}"
    )
    assert fs > 0, "num_fetch_sms must be positive"
    assert nfs > 0, "num_fetch_stages must be positive"
    assert fsf > 0, "first_stage_fetch_sms must be positive"


def test_auto_config_champion_shapes():
    """Verify that champion shapes are returned directly from _CHAMPION_CONFIGS."""
    for key in _CHAMPION_CONFIGS:
        M, N, K = key
        config, kpf, fs, nfs, fsf = _auto_config(M, N, K, world_size=8)
        c = _CHAMPION_CONFIGS[key]

        assert config.block_size_m == c["bm"]
        assert config.block_size_n == c["bn"]
        assert config.block_size_k == c["bk"]
        assert config.group_size_m == c["gm"]

        # kpf may be adjusted down by _auto_config when champion["kpf"] doesn't divide
        # num_k_blocks (e.g. different world_size changes K and therefore num_k_blocks).
        num_k_blocks = K // c["bk"]
        assert num_k_blocks % kpf == 0, f"Champion kpf={kpf} does not divide num_k_blocks={num_k_blocks} for {key}"


def test_auto_config_large_m_uses_block_256():
    """Verify _auto_config picks block_m=256 for large M (M >= 8192, M divisible by 256)."""
    config, *_ = _auto_config(8192, 3584, 8192, world_size=8)
    assert config.block_size_m == 256, f"Expected block_m=256 for large M, got {config.block_size_m}"


def test_auto_config_small_m_uses_block_128():
    """Verify _auto_config picks block_m=128 for small M (M < 8192)."""
    config, *_ = _auto_config(1024, 3584, 8192, world_size=8)
    assert config.block_size_m == 128, f"Expected block_m=128 for small M, got {config.block_size_m}"


def test_auto_config_block_n_always_256():
    """Verify _auto_config always selects block_n=256 (from sweep data)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_n == 256, f"Expected block_n=256 for M={M}, got {config.block_size_n}"


def test_auto_config_block_k_always_64():
    """Verify _auto_config always selects block_k=64 (exceeding LDS on MI300X with 128)."""
    for M in [1024, 4096, 16384]:
        config, *_ = _auto_config(M, 3584, 8192, world_size=8)
        assert config.block_size_k == 64, f"Expected block_k=64 for M={M}, got {config.block_size_k}"


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
