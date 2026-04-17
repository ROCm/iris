# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for high-level matmul_all_reduce API.

Note: This test requires tritonBLAS to be installed.
Install with: pip install git+https://github.com/ROCm/tritonBLAS.git
"""

import pytest
import torch
import torch.distributed as dist
import iris
import iris.ops as ops


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
    [
        (128, 64, 32),
        (1024, 256, 512),
    ],
)
@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
        "one_shot",
        "two_shot",
    ],
)
def test_matmul_all_reduce(dtype, atol, rtol, M, N, K, variant):
    """Test matmul_all_reduce by comparing against torch.matmul + dist.all_reduce."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Create input matrices
    A_local = torch.randn(M, K, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K, N, dtype=dtype, device=f"cuda:{rank}")

    # Compute reference: torch.matmul + dist.all_reduce
    C_local_ref = torch.matmul(A_local, B)
    pytorch_output = C_local_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Set up Iris tensors
    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_A.copy_(A_local)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_B.copy_(B)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Select appropriate config based on problem size
    from iris.ops.config import FusedConfig

    if M <= 128 or K <= 64 or N <= 128:
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant=variant)
    elif dtype == torch.float32:
        config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant=variant)
    else:
        config = FusedConfig(all_reduce_variant=variant)

    # Use high-level API
    ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B, config=config)

    torch.cuda.synchronize()
    shmem.barrier()

    max_diff = torch.abs(iris_C - pytorch_output).max().item()

    assert torch.allclose(iris_C, pytorch_output, atol=atol, rtol=rtol), (
        f"Max difference: {max_diff}, expected < {atol}\n"
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

    # Reference using PyTorch
    A_ref = A.clone()
    B_ref = B.clone()
    C_ref = torch.matmul(A_ref, B_ref)
    pytorch_output = C_ref.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

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


def test_matmul_all_reduce_lock_too_small():
    """Test that ValueError is raised when the lock array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then reused
    with smaller block sizes (more tiles). The workspace.matches() call skips the preamble,
    leaving the undersized lock array in place.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    from iris.ops.config import FusedConfig

    M, N, K = 512, 512, 64
    dtype = torch.float16
    variant = "one_shot"

    iris_A = shmem.zeros((M, K), dtype=dtype)
    iris_B = shmem.zeros((K, N), dtype=dtype)
    iris_C = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller lock array
    config_large = FusedConfig(block_size_m=128, block_size_n=128, block_size_k=32, all_reduce_variant=variant)
    workspace = ops.matmul_all_reduce_preamble(shmem, iris_C, iris_A, iris_B, config=config_large)

    # Step 2: manually mark workspace as prepared so matmul_all_reduce skips the preamble
    workspace.prepared = True

    # Step 3: call matmul_all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized lock array is detected.
    config_small = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant=variant)
    with pytest.raises(ValueError, match="Lock array too small"):
        ops.matmul_all_reduce(shmem, iris_C, iris_A, iris_B, config=config_small, workspace=workspace)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
