# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for matmul_all_reduce_copy_engine.

Each rank computes A @ B locally, then reduces across all ranks using
copy-engine for communication. Supports one_shot and two_shot variants.
"""

import gc
import os
import pytest
import torch
import torch.distributed as dist

import iris
from iris.ops.matmul_all_reduce_copy_engine import (
    matmul_all_reduce_copy_engine,
    matmul_all_reduce_copy_engine_preamble,
)
from tritonblas.matmul import _make_matmul_selector, persistent_matmul_lt


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Fixture to clean up GPU memory before and after each test."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    yield
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _param_shapes():
    if "IRIS_TEST_M" in os.environ:
        return [
            (
                int(os.environ["IRIS_TEST_M"]),
                int(os.environ["IRIS_TEST_N"]),
                int(os.environ["IRIS_TEST_K"]),
            )
        ]
    return [(256, 128, 256)]


def _heap_size() -> int:
    return int(os.environ.get("IRIS_TEST_HEAP_SIZE", 1 << 34))


def _make_selector(M: int, N: int, K: int, dtype: torch.dtype, device: torch.device):
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


def _reference_tolerances(dtype: torch.dtype):
    # The fused kernel and reference below use the same tritonBLAS GEMM. The
    # remaining drift is only from the all-reduce accumulation order and final
    # cast back to the output dtype.
    if dtype == torch.float16:
        return 0.5, 0.01
    if dtype == torch.bfloat16:
        return 4.0, 0.02
    return 0.3, 0.01


@pytest.mark.parametrize("M,N,K", _param_shapes())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("variant", ["one_shot", "two_shot"])
def test_matmul_all_reduce_copy_engine(M, N, K, dtype, variant):
    """Test matmul_all_reduce_copy_engine against tritonBLAS matmul + fp32 all-reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    heap_size = _heap_size()
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    device = torch.device(f"cuda:{rank}")
    selector = _make_selector(M, N, K, dtype, device)

    # Create input matrices
    torch.manual_seed(123 + rank)
    A_local = torch.randn(M, K, dtype=dtype, device=device)
    torch.manual_seed(456)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Compute a reference with the same local GEMM implementation used by the
    # fused kernel, so this test validates the copy-engine all-reduce rather
    # than rocBLAS-vs-tritonBLAS matmul rounding at large K.
    C_local_ref = torch.empty((M, N), dtype=dtype, device=device)
    persistent_matmul_lt(
        A_local,
        B,
        C_local_ref,
        selector,
        config=None,
        bias=None,
        work_stealing=False,
    )
    torch.cuda.synchronize()
    pytorch_output = C_local_ref.to(torch.float32)
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    pytorch_output = pytorch_output.to(dtype)

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Create workspace with config
    from iris.ops import FusedConfig

    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_copy_engine_preamble(
        shmem,
        iris_C,
        iris_A,
        iris_B,
        config=config,
        selector=selector,
    )

    # Use the API
    matmul_all_reduce_copy_engine(
        shmem,
        iris_C,
        iris_A,
        iris_B,
        async_op=False,
        config=config,
        workspace=workspace,
    )

    torch.cuda.synchronize()
    shmem.barrier()

    # Validate
    atol, rtol = _reference_tolerances(dtype)

    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected within atol={atol}, rtol={rtol}\n"
        f"Rank {rank}: matmul_all_reduce_copy_engine output doesn't match reference\n"
        f"variant={variant}"
    )

    if rank == 0:
        print(
            f"✓ matmul_all_reduce_copy_engine test passed: {dtype}, M={M}, N={N}, K={K}, "
            f"variant={variant}"
        )

    shmem.barrier()
    del shmem
    gc.collect()
