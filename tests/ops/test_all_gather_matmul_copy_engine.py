# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for all_gather_matmul_copy_engine.

Each rank owns A_sharded (M, K_local), gathers the K dimension across ranks,
and computes C = A_gathered @ B. This file exercises both the host-initiated
and device-initiated copy-engine paths against a torch reference.
"""

import pytest
import torch
import torch.distributed as dist

import iris
import os
from iris.ops.all_gather_matmul_copy_engine import (
    all_gather_matmul_copy_engine,
    all_gather_matmul_copy_engine_preamble,
)
from iris.ops.config import FusedConfig
from tritonblas.matmul import _make_matmul_selector


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


def _heap_size() -> int:
    return int(os.environ.get("IRIS_TEST_HEAP_SIZE", 1 << 34))


def _make_reference(rank, world_size, M, K_local, N, dtype):
    device = f"cuda:{rank}"
    K = K_local * world_size

    torch.manual_seed(42 + rank)
    A_sharded = torch.randn(M, K_local, dtype=dtype, device=device)

    torch.manual_seed(123)
    B = torch.randn(K, N, dtype=dtype, device=device)

    gathered = [torch.zeros(M, K_local, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, A_sharded)
    A_gathered_ref = torch.cat(gathered, dim=1)
    ref_output = torch.matmul(A_gathered_ref, B)
    torch.cuda.synchronize()
    return A_sharded, B, ref_output


def _make_selector_config(M, N, K, dtype, device):
    selector = _make_matmul_selector(
        M,
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
@pytest.mark.parametrize("device_initiated", _device_initiated_modes())
@pytest.mark.parametrize("M,K_local,N", _param_shapes())
def test_all_gather_matmul_copy_engine(dtype, atol, rtol, device_initiated, M, K_local, N):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = _heap_size()
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")
    K = K_local * world_size

    A_sharded, B, ref_output = _make_reference(rank, world_size, M, K_local, N, dtype)
    selector, config = _make_selector_config(M, N, K, dtype, device)

    if M % config.block_size_m != 0:
        pytest.skip(f"M={M} must be divisible by block_size_m={config.block_size_m}")
    if K % config.block_size_k != 0:
        pytest.skip(f"K={K} must be divisible by block_size_k={config.block_size_k}")
    if K_local % config.block_size_k != 0:
        pytest.skip(f"K_local={K_local} must be divisible by block_size_k={config.block_size_k}")

    A_sharded_shmem = ctx.zeros((M, K_local), dtype=dtype)
    A_sharded_shmem.copy_(A_sharded)
    B_shmem = ctx.zeros((K, N), dtype=dtype)
    B_shmem.copy_(B)
    output = ctx.zeros((M, N), dtype=dtype)

    m_tiles_per_batch = 1
    workspace = all_gather_matmul_copy_engine_preamble(
        ctx,
        A_sharded_shmem,
        B_shmem,
        config=config,
        k_per_flag=4,
        m_tiles_per_batch=m_tiles_per_batch,
    )
    workspace.selector = selector

    ctx.barrier()

    all_gather_matmul_copy_engine(
        ctx,
        output,
        A_sharded_shmem,
        B_shmem,
        config=config,
        workspace=workspace,
        k_per_flag=4,
        m_tiles_per_batch=m_tiles_per_batch,
        device_initiated=device_initiated,
        trace=False,
    )

    torch.cuda.synchronize()
    ctx.barrier()

    max_diff = (output - ref_output).abs().max().item()
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol), (
        f"Rank {rank}: Max diff {max_diff}, expected < {atol} "
        f"(device_initiated={device_initiated}, M={M}, K_local={K_local}, N={N})"
    )


if __name__ == "__main__":
    import sys

    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=2 tests/ops/test_all_gather_matmul_copy_engine.py")
        sys.exit(1)

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Tests in this file require pytest + torchrun. See tests/run_tests_distributed.py")
