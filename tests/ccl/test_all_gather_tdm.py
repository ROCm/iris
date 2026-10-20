# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the gfx1250 TDM (tensor data movement) all-gather.

The output is compared bit for bit: all-gather only moves data. The expected output
is rebuilt locally from each rank's seed, so the check does not depend on another
collective implementation.
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


def _shard(rank, M, N, dtype):
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    return torch.randn(M, N, device="cuda", generator=g, dtype=torch.float32).to(dtype)


def _expected(world_size, M, N, dtype):
    return torch.cat([_shard(r, M, N, dtype) for r in range(world_size)])


def _run(shmem, M, N, dtype, config, calls=1):
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    src = shmem.zeros((M, N), dtype=dtype)
    src.copy_(_shard(rank, M, N, dtype))
    dst = shmem.zeros((world_size * M, N), dtype=dtype)
    dst.fill_(float("nan"))
    shmem.barrier()
    for _ in range(calls):
        shmem.ccl.all_gather(dst, src, config=config)
    torch.cuda.synchronize()
    return dst


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, N",
    [
        (1024, 4096),  # whole tiles
        (100, 4096),  # ragged rows: the last tile is clipped at this rank's shard
        (513, 2050),  # ragged rows and columns
        (7, 300),  # smaller than one tile
    ],
)
@pytest.mark.parametrize(
    "block_size_m, block_size_n, num_warps, comm_sms, num_stages",
    [
        (32, 256, 4, 64, 1),
        (64, 512, 16, 256, 1),
        # Pipelined: several tiles per program, so the LDS ring wraps.
        (32, 128, 4, 4, 2),
        (128, 128, 16, 4, 3),
        (16, 256, 4, 2, 4),
    ],
)
def test_all_gather_tdm(dtype, M, N, block_size_m, block_size_n, num_warps, comm_sms, num_stages):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            all_gather_variant="tdm",
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_warps=num_warps,
            comm_sms=comm_sms,
            num_stages=num_stages,
        )
        out = _run(shmem, M, N, dtype, config)
        expected = _expected(shmem.get_num_ranks(), M, N, dtype)
        wrong = (out != expected).sum().item()
        assert wrong == 0, f"rank {shmem.get_rank()}: {wrong} of {expected.numel()} elements differ"
    finally:
        shmem.barrier()
        del shmem


@pytest.mark.parametrize("num_stages", [1, 3])
def test_all_gather_tdm_repeated_calls(num_stages):
    """Back-to-back calls on the same buffers stay correct."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            all_gather_variant="tdm",
            block_size_m=64,
            block_size_n=512,
            num_warps=16,
            num_stages=num_stages,
        )
        out = _run(shmem, 256, 4096, torch.bfloat16, config, calls=10)
        expected = _expected(shmem.get_num_ranks(), 256, 4096, torch.bfloat16)
        assert torch.equal(out, expected)
    finally:
        shmem.barrier()
        del shmem


@pytest.mark.parametrize("M, N", [(0, 4096), (64, 0)])
def test_all_gather_tdm_empty(M, N):
    """An empty shard launches nothing and leaves the (empty) output alone."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(use_gluon=True, all_gather_variant="tdm", block_size_m=64, block_size_n=256, num_stages=3)
        out = _run(shmem, M, N, torch.bfloat16, config)
        assert out.shape == (shmem.get_num_ranks() * M, N)
    finally:
        shmem.barrier()
        del shmem


def test_all_gather_tdm_requires_gluon():
    with pytest.raises(ValueError, match="use_gluon=True"):
        Config(all_gather_variant="tdm")


def test_num_stages_must_be_positive():
    with pytest.raises(ValueError, match="num_stages must be positive"):
        Config(use_gluon=True, all_gather_variant="tdm", num_stages=0)


def test_all_gather_tdm_rejects_ring_larger_than_lds():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        # 8 stages of a 256 x 256 fp16 tile is 1 MiB, more LDS than any workgroup has.
        config = Config(
            use_gluon=True, all_gather_variant="tdm", block_size_m=256, block_size_n=256, num_warps=16, num_stages=8
        )
        src = shmem.zeros((256, 256), dtype=torch.float16)
        dst = shmem.zeros((shmem.get_num_ranks() * 256, 256), dtype=torch.float16)
        with pytest.raises(ValueError, match="LDS"):
            shmem.ccl.all_gather(dst, src, config=config)
    finally:
        shmem.barrier()
        del shmem
