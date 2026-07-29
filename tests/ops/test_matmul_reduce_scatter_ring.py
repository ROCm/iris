#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for fused GEMM + ring reduce-scatter.

Reference: torch.mm(A, B) on each rank (partial sum), then
dist.reduce_scatter_tensor to get each rank's M-shard.
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ops import FusedConfig
from iris.ops.matmul_reduce_scatter_ring import (
    matmul_reduce_scatter_ring,
)


def _reference(A, B, world_size, rank):
    C_partial = torch.mm(A, B)
    M, N = C_partial.shape
    M_local = M // world_size
    C_local = torch.empty((M_local, N), device=A.device, dtype=A.dtype)
    dist.reduce_scatter_tensor(C_local, C_partial, op=dist.ReduceOp.SUM)
    return C_local


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (torch.float16, 5e-1),
        (torch.bfloat16, 5e-1),
    ],
)
@pytest.mark.parametrize(
    "M, N, K_local",
    [
        (1024, 128, 64),
        (2048, 256, 128),
        (2048, 2880, 512),
    ],
)
def test_ring_reduce_scatter(dtype, atol, M, N, K_local):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    block_m = 128
    if M % (world_size * block_m) != 0:
        pytest.skip(f"M={M} not divisible by ws*bm={world_size * block_m}")

    M_local = M // world_size

    A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

    ref = _reference(A, B, world_size, rank)
    torch.cuda.synchronize()

    iris_A = shmem.zeros((M, K_local), dtype=dtype)
    iris_A.copy_(A)
    iris_C = torch.zeros((M_local, N), dtype=dtype, device=f"cuda:{rank}")

    config = FusedConfig(block_size_m=128, block_size_n=64, block_size_k=64, group_size_m=4)
    shmem.barrier()

    matmul_reduce_scatter_ring(
        shmem,
        iris_C,
        iris_A,
        B,
        config=config,
        num_scatter_sms=32,
    )
    torch.cuda.synchronize()

    max_diff = torch.abs(iris_C - ref).max().item()
    if rank == 0:
        print(f"Ring RS: {dtype}, M={M}, N={N}, K_local={K_local}, max_diff={max_diff:.6f}")

    assert torch.allclose(iris_C, ref, atol=atol, rtol=1e-2), f"Rank {rank}: Max diff {max_diff}, expected < {atol}"

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-x"]))
