# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for all_gather_matmul_copy_engine.

Each rank owns A_sharded (M, K_local), gathers the K dimension across ranks,
and computes C = A_gathered @ B. This file exercises both the host-initiated
and device-initiated copy-engine paths against a torch reference.
"""

import gc
import pytest
import torch
import torch.distributed as dist

import iris
import os
from iris.ops.all_gather_matmul_copy_engine import (
    all_gather_matmul_copy_engine,
    all_gather_matmul_copy_engine_preamble,
)
from tritonblas.matmul import _make_matmul_selector


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Fixture to clean up GPU memory before and after each test."""
    # Cleanup before test starts
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    yield  # Run the test
    # Cleanup after test completes (pass or fail)
    gc.collect()
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
    return [(256, 128, 256)]


def _device_initiated_modes():
    mode = os.environ.get("IRIS_TEST_COPY_ENGINE_MODE")
    if mode == "host":
        return [False]
    if mode == "device":
        return [True]
    return [False, True]


def _host_transfer_backends():
    backend = os.environ.get("IRIS_TEST_HOST_TRANSFER_BACKEND")
    if backend:
        return [backend]
    return ["anvil"]


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


def _assert_sampled_output_matches(output, A_sharded, B, rank, world_size, atol, rtol, context):
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
            output_tile = output[row_start:row_end, col_start:col_end]
            _assert_close_tile(output_tile, ref_tile, atol, rtol, rank, row_start, col_start, context)

        del gathered_a_rows, gathered_a_parts, local_a

    torch.cuda.synchronize()


def _assert_output_matches_reference(output, A_sharded, B, rank, world_size, atol, rtol, context):
    if _should_sample_validation(output.shape[0], output.shape[1], output.dtype):
        _assert_sampled_output_matches(output, A_sharded, B, rank, world_size, atol, rtol, context)
        return

    ref_output = _make_full_reference(A_sharded, B, world_size)
    _assert_full_output_matches(output, ref_output, atol, rtol, rank, context)


def _make_selector(M, N, K, dtype, device):
    return _make_matmul_selector(
        M,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )


@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float16, 5e-2, 5e-2)])
@pytest.mark.parametrize("device_initiated", _device_initiated_modes())
@pytest.mark.parametrize("host_transfer_backend", _host_transfer_backends())
@pytest.mark.parametrize("M,K_local,N", _param_shapes())
def test_all_gather_matmul_copy_engine(dtype, atol, rtol, device_initiated, host_transfer_backend, M, K_local, N):
    """Test all_gather_matmul_copy_engine against torch all_gather + matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    K = K_local * world_size

    A_sharded, B = _make_inputs(rank, world_size, M, K_local, N, dtype)
    selector = _make_selector(M, N, K, dtype, B.device)

    if M % selector.block_m != 0:
        pytest.skip(f"M={M} must be divisible by block_m={selector.block_m}")
    if K % selector.block_k != 0:
        pytest.skip(f"K={K} must be divisible by block_k={selector.block_k}")
    if K_local % selector.block_k != 0:
        pytest.skip(f"K_local={K_local} must be divisible by block_k={selector.block_k}")

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    workspace = all_gather_matmul_copy_engine_preamble(
        ctx,
        A_sharded_shmem,
        B_shmem,
        selector=selector,
        k_per_flag=4,
    )

    ctx.barrier()

    all_gather_matmul_copy_engine(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        workspace=workspace,
        k_per_flag=4,
        device_initiated=device_initiated,
        host_transfer_backend=host_transfer_backend,
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
        (
            f"device_initiated={device_initiated}, host_transfer_backend={host_transfer_backend}, "
            f"M={M}, K_local={K_local}, N={N}"
        ),
    )

    # Clean up to prevent OOM in subsequent tests
    del A_sharded, B, A_sharded_shmem, B_shmem, output, workspace
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul_copy_engine.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
