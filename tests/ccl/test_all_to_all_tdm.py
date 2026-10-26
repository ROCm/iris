# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the gfx1250 TDM (tensor data movement) all-to-all.

The output is compared bit for bit: all-to-all only moves data. Every (source,
destination) chunk has its own seed, so a chunk landing in the wrong place is
caught, and the expected output is rebuilt locally without another collective.
"""

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


def _is_gfx1250():
    # The device this rank runs on: the test runner sets it before collecting tests.
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx1250")


try:
    from triton.experimental.gluon.language.amd.gfx1250 import tdm  # noqa: F401

    TDM_AVAILABLE = _is_gfx1250()
except ImportError:
    TDM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TDM_AVAILABLE, reason="needs a gfx1250 GPU and Gluon TDM support")


def _chunk(src, dst, M, N, dtype):
    """What rank src sends to rank dst."""
    g = torch.Generator(device="cuda").manual_seed(1000 + 97 * src + dst)
    return torch.randn(M, N, device="cuda", generator=g, dtype=torch.float32).to(dtype)


def _run(shmem, M, N, dtype, config, calls=1):
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    src = shmem.zeros((M, world_size * N), dtype=dtype)
    for d in range(world_size):
        src[:, d * N : (d + 1) * N] = _chunk(rank, d, M, N, dtype)
    dst = shmem.zeros((M, world_size * N), dtype=dtype)
    dst.fill_(float("nan"))
    shmem.barrier()
    for _ in range(calls):
        shmem.ccl.all_to_all(dst, src, config=config)
    torch.cuda.synchronize()
    return dst


def _expected(rank, world_size, M, N, dtype):
    return torch.cat([_chunk(s, rank, M, N, dtype) for s in range(world_size)], dim=1)


def _check(shmem, M, N, dtype, config, calls=1):
    out = _run(shmem, M, N, dtype, config, calls)
    expected = _expected(shmem.get_rank(), shmem.get_num_ranks(), M, N, dtype)
    wrong = (out != expected).sum().item()
    assert wrong == 0, f"rank {shmem.get_rank()}: {wrong} of {expected.numel()} elements differ"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, N",
    [
        (1024, 1024),  # whole tiles
        (100, 1024),  # ragged rows
        (513, 1030),  # ragged rows and columns: the last tile is clipped at this rank's chunk
        (7, 300),  # smaller than one tile
    ],
)
@pytest.mark.parametrize(
    "block_size_m, block_size_n, num_warps, comm_sms, num_stages",
    [
        (32, 256, 4, 64, 1),
        (64, 256, 16, 256, 1),
        # Pipelined, with few enough programs that the LDS ring wraps.
        (32, 256, 4, 8, 2),
        (64, 128, 16, 8, 3),
        (128, 128, 16, 4, 6),
        # LOOKAHEAD - 1 = 9 is more than world_size: the peel spans several tiles.
        (16, 128, 4, 4, 20),
    ],
)
def test_all_to_all_tdm(dtype, M, N, block_size_m, block_size_n, num_warps, comm_sms, num_stages):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ring_bytes = num_stages * block_size_m * block_size_n * dtype.itemsize
    max_lds = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block
    if ring_bytes > max_lds:
        # e.g. 6 x 128 x 128 fp32 is 384 KiB; gfx1250 has 320 KiB. The launcher rejects it,
        # which test_all_to_all_tdm_rejects_oversized_ring covers.
        pytest.skip(f"a {ring_bytes}-byte LDS ring does not fit in {max_lds} bytes")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            all_to_all_variant="tdm",
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_warps=num_warps,
            comm_sms=comm_sms,
            num_stages=num_stages,
        )
        _check(shmem, M, N, dtype, config)
    finally:
        shmem.barrier()
        del shmem


@pytest.mark.parametrize("num_stages", [1, 6])
def test_all_to_all_tdm_repeated_calls(num_stages):
    """Back-to-back calls on the same buffers stay correct."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            all_to_all_variant="tdm",
            block_size_m=128,
            block_size_n=128,
            num_warps=16,
            comm_sms=16,
            num_stages=num_stages,
        )
        _check(shmem, 256, 1024, torch.bfloat16, config, calls=10)
    finally:
        shmem.barrier()
        del shmem


def test_all_to_all_tdm_rejects_oversized_ring():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        max_lds = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block
        stages = max_lds // (128 * 128 * 2) + 1
        config = Config(use_gluon=True, all_to_all_variant="tdm", block_size_m=128, block_size_n=128, num_stages=stages)
        with pytest.raises(ValueError, match="bytes of LDS"):
            _run(shmem, 128, 128, torch.bfloat16, config)
    finally:
        shmem.barrier()
        del shmem


def test_all_to_all_tdm_requires_gluon():
    with pytest.raises(ValueError, match="use_gluon=True"):
        Config(all_to_all_variant="tdm")
