# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for matmul_all_gather_copy_engine.

Each rank computes C_local = A_local @ B and the copy engine scatters the
result tiles so every rank observes the gathered output C.
"""

import pytest
import torch
import torch.distributed as dist

import iris
import os
from iris.ops.config import FusedConfig
from iris.ops.matmul_all_gather_host_copy_engine import (
    matmul_all_gather_host_copy_engine,
    matmul_all_gather_host_copy_engine_preamble,
)
from iris.ops.matmul_all_gather_copy_engine import (
    matmul_all_gather_copy_engine,
    matmul_all_gather_copy_engine_preamble,
)
from tritonblas.matmul import _make_matmul_selector


def _param_shapes():
    if "IRIS_TEST_M" in os.environ:
        return [
            (
                int(os.environ["IRIS_TEST_M"]),
                int(os.environ["IRIS_TEST_N"]),
                int(os.environ["IRIS_TEST_K"]),
            )
        ]
    return [(1024, 256, 256)]


def _copy_engine_modes():
    mode = os.environ.get("IRIS_TEST_COPY_ENGINE_MODE")
    if mode in {"host", "device"}:
        return [mode]
    return ["host", "device"]


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


def _assert_close_chunked(output_chunk, ref_chunk, atol, rtol, copy_engine_mode, src_rank, row_start):
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
                f"Rank validation failed for matmul_all_gather_copy_engine (mode={copy_engine_mode})"
            )


def _assert_gathered_rows_match_dense(output, local_ref, rank, world_size, atol, rtol, copy_engine_mode):
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
                f"Rank validation failed for matmul_all_gather_copy_engine (mode={copy_engine_mode})"
            )


def _assert_gathered_rows_match_streamed(output, A_local, B, rank, world_size, atol, rtol, copy_engine_mode):
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
            _assert_close_chunked(
                output_chunk,
                ref_chunk,
                atol,
                rtol,
                copy_engine_mode,
                src_rank,
                global_row_start,
            )


def _assert_gathered_rows_match(output, A_local, B, rank, world_size, atol, rtol, copy_engine_mode):
    if _should_stream_validation(A_local.shape[0], output.shape[1], output.dtype):
        _assert_gathered_rows_match_streamed(output, A_local, B, rank, world_size, atol, rtol, copy_engine_mode)
        return

    local_ref = torch.matmul(A_local, B)
    torch.cuda.synchronize()
    _assert_gathered_rows_match_dense(output, local_ref, rank, world_size, atol, rtol, copy_engine_mode)


def _make_selector_config(M_local, N, K, dtype, device):
    selector = _make_matmul_selector(
        M_local,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )
    config = FusedConfig(
        block_size_m=selector.block_m,
        block_size_n=selector.block_n,
        block_size_k=selector.block_k,
        group_size_m=selector.group_m,
        num_xcds=max(1, int(getattr(selector, "num_sms", 1))),
    )
    return selector, config


@pytest.mark.parametrize("dtype, atol, rtol", [(torch.float16, 5e-2, 5e-2)])
@pytest.mark.parametrize("copy_engine_mode", _copy_engine_modes())
@pytest.mark.parametrize("M,N,K", _param_shapes())
def test_matmul_all_gather_copy_engine(dtype, atol, rtol, copy_engine_mode, M, N, K):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    device = torch.device(f"cuda:{rank}")

    if M % world_size != 0:
        pytest.skip(f"M={M} not divisible by world_size={world_size}")

    M_local = M // world_size
    selector, config = _make_selector_config(M_local, N, K, dtype, device)

    if M_local % config.block_size_m != 0:
        pytest.skip(f"M_local={M_local} must be divisible by block_size_m={config.block_size_m}")
    if K % config.block_size_k != 0:
        pytest.skip(f"K={K} must be divisible by block_size_k={config.block_size_k}")

    A_local = shmem.randn((M_local, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    m_tiles_per_batch = 1
    if copy_engine_mode == "device":
        workspace = matmul_all_gather_copy_engine_preamble(
            shmem,
            A_local,
            B,
            config=config,
            m_tiles_per_batch=m_tiles_per_batch,
        )
        workspace.selector = selector
    else:
        workspace = matmul_all_gather_host_copy_engine_preamble(
            shmem,
            A_local,
            B,
            config=config,
            m_tiles_per_batch=m_tiles_per_batch,
            trace=False,
            use_tritonblas=True,
        )

    shmem.barrier()

    if copy_engine_mode == "device":
        matmul_all_gather_copy_engine(
            shmem,
            output,
            A_local,
            B,
            config=config,
            workspace=workspace,
            use_copy_engine=True,
            m_tiles_per_batch=m_tiles_per_batch,
        )
    else:
        matmul_all_gather_host_copy_engine(
            shmem,
            output,
            A_local,
            B,
            config=config,
            workspace=workspace,
            m_tiles_per_batch=m_tiles_per_batch,
            trace=False,
            use_tritonblas=True,
        )

    torch.cuda.synchronize()
    shmem.barrier()

    _assert_gathered_rows_match(output, A_local, B, rank, world_size, atol, rtol, copy_engine_mode)

    shmem.barrier()
    del output
    del B
    del A_local
    del shmem
    import gc

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_matmul_all_gather_copy_engine.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
