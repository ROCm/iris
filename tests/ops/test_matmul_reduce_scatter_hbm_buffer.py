# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for fused GEMM+ReduceScatter with HBM buffer staging.

Reference: torch.mm(A, B) on each rank (partial sum), then
dist.reduce_scatter_tensor to get each rank's M-shard of the reduced result.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ops import FusedConfig
from iris.ops.matmul_reduce_scatter_hbm_buffer import (
    matmul_reduce_scatter_hbm_buffer,
    matmul_reduce_scatter_hbm_buffer_preamble,
)


def _reference_gemm_reduce_scatter(A, B, world_size, rank):
    """Compute reference: GEMM partial sum then reduce-scatter along M."""
    C_partial = torch.mm(A, B)
    M, N = C_partial.shape
    M_local = M // world_size
    C_local = torch.empty((M_local, N), device=A.device, dtype=A.dtype)
    dist.reduce_scatter_tensor(C_local, C_partial, op=dist.ReduceOp.SUM)
    return C_local


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 5e-1, 1e-2),
        (torch.bfloat16, 5e-1, 1e-2),
    ],
)
@pytest.mark.parametrize(
    "M, N, K_local",
    [
        (256, 128, 64),
        (512, 256, 128),
        (2048, 2880, 512),
    ],
)
def test_matmul_reduce_scatter_hbm_buffer(dtype, atol, rtol, M, N, K_local):
    """Test fused GEMM+RS HBM buffer against torch.mm + dist.reduce_scatter_tensor."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if M % (world_size * 128) != 0:
        pytest.skip(f"M={M} not divisible by world_size*block_size_m={world_size * 128}")

    M_local = M // world_size

    A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

    ref = _reference_gemm_reduce_scatter(A, B, world_size, rank)
    torch.cuda.synchronize()

    iris_A = shmem.zeros((M, K_local), dtype=dtype)
    iris_A.copy_(A)
    iris_B = B.clone()
    iris_C = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

    config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)
    shmem.barrier()

    matmul_reduce_scatter_hbm_buffer(
        shmem, iris_C, iris_A, iris_B, config=config, num_scatter_sms=32,
    )

    torch.cuda.synchronize()

    max_diff = torch.abs(iris_C - ref).max().item()
    if rank == 0:
        print(f"GEMM+RS HBM buffer: {dtype}, M={M}, N={N}, K_local={K_local}, max_diff={max_diff:.6f}")

    assert torch.allclose(iris_C, ref, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol}"
    )

    shmem.barrier()
    del shmem
    import gc
    gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16])
def test_gpt_oss_120b_shapes(dtype):
    """Test with GPT-OSS-120B MoE shapes from aporva's PR #513."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    K_global = 4096
    N = 2880
    K_local = K_global // world_size

    for M in [2048]:
        if M % (world_size * 128) != 0:
            continue

        M_local = M // world_size

        A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
        B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

        ref = _reference_gemm_reduce_scatter(A, B, world_size, rank)
        torch.cuda.synchronize()

        iris_A = shmem.zeros((M, K_local), dtype=dtype)
        iris_A.copy_(A)
        iris_B = B.clone()
        iris_C = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

        config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)
        shmem.barrier()

        matmul_reduce_scatter_hbm_buffer(
            shmem, iris_C, iris_A, iris_B, config=config, num_scatter_sms=32,
        )

        torch.cuda.synchronize()

        max_diff = torch.abs(iris_C - ref).max().item()
        atol = 5e-1

        if rank == 0:
            print(f"GPT-OSS-120B shape: M={M}, N={N}, K_local={K_local}, max_diff={max_diff:.6f}")

        assert torch.allclose(iris_C, ref, atol=atol, rtol=1e-2), (
            f"Rank {rank}, M={M}: Max diff {max_diff}"
        )

    shmem.barrier()
    del shmem
    import gc
    gc.collect()


def test_workspace_reuse(dtype=torch.float16):
    """Test that workspace can be reused across calls."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N, K_local = 256, 128, 64
    if M % (world_size * 128) != 0:
        pytest.skip("Shape not compatible")

    M_local = M // world_size

    A = shmem.zeros((M, K_local), dtype=dtype)
    A.fill_(0.01 * (rank + 1))
    B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")
    C = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

    config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)

    ws = matmul_reduce_scatter_hbm_buffer_preamble(shmem, A, B, config)

    for _ in range(3):
        C.zero_()
        matmul_reduce_scatter_hbm_buffer(
            shmem, C, A, B, config=config, workspace=ws, num_scatter_sms=32,
        )
        torch.cuda.synchronize()

    if rank == 0:
        print("Workspace reuse: 3 iterations passed")

    shmem.barrier()
    del shmem
    import gc
    gc.collect()
