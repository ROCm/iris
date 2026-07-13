# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_reduce API.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import os
import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops
from iris.ops.config import FusedConfig


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
        (128, 64, 32),
        (1024, 256, 512),
    ]


def _heap_size() -> int:
    return int(os.environ.get("IRIS_TEST_HEAP_SIZE", 2**33))


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 0.2, 0.01),
        (torch.float32, 0.3, 0.01),
        (torch.bfloat16, 2.5, 0.02),  # Increased from 1.5 to 2.5 for 8-rank tests
    ],
)
@pytest.mark.parametrize(
    "M, N, K",
    _param_shapes(),
)
@pytest.mark.parametrize(
    "variant",
    [
        "one_shot",
        "two_shot",
    ],
)
def test_matmul_all_reduce(dtype, atol, rtol, M, N, K, variant):
    """Test matmul_all_reduce against torch.matmul plus fp32 all-reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Match the benchmark inputs: per-rank A, replicated B.
    torch.manual_seed(123 + rank)
    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    torch.manual_seed(456)
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: local matmul rounded to output dtype, fp32 all-reduce,
    # then final cast back to the output dtype.
    C_local_ref = torch.matmul(A_local, B)
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

    config = FusedConfig(all_reduce_variant=variant)

    # Use high-level API
    ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected within atol={atol}, rtol={rtol}\n"
        f"Rank {rank}: iris.ops.matmul_all_reduce output doesn't match reference"
    )

    if rank == 0:
        print(f"✓ matmul_all_reduce test passed: {dtype}, M={M}, N={N}, K={K}, variant={variant}")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_all_reduce_via_shmem_ops():
    """Test accessing matmul_all_reduce via shmem.ops namespace."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    M, N, K = 256, 128, 64
    dtype = torch.float16

    A = shmem.randn((M, K), dtype=dtype)
    B = shmem.randn((K, N), dtype=dtype)
    output = shmem.zeros((M, N), dtype=dtype)

    # Reference using PyTorch with the same fp32 all-reduce contract as the op.
    A_ref = A.clone()
    B_ref = B.clone()
    C_ref = torch.matmul(A_ref, B_ref)
    pytorch_output = C_ref.to(torch.float32)
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    pytorch_output = pytorch_output.to(dtype)

    # Use shmem.ops interface
    shmem.ops.matmul_all_reduce(output, A, B)

    torch.cuda.synchronize()
    shmem.barrier()

    atol = 0.2
    rtol = 0.01
    assert torch.allclose(output, pytorch_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: shmem.ops.matmul_all_reduce doesn't match reference"
    )

    if rank == 0:
        print("✓ shmem.ops.matmul_all_reduce test passed")

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_matmul_all_reduce_unsupported_variant():
    """matmul_all_reduce only supports the direct one_shot/two_shot paths."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    from iris.ops.config import FusedConfig

    M, N, K = 128, 128, 64
    dtype = torch.float16

    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant="atomic")
    with pytest.raises(ValueError, match="supports only"):
        ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B, config=config)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
