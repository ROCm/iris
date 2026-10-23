# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the gfx1250 TDM (tensor data movement) reduce-scatter.

Each rank owns a set of output tiles (Config.all_reduce_distribution) and must
write the sum over all ranks there, leaving every other element untouched. The
expected sum is rebuilt locally from each rank's seed, so the check does not
depend on another collective implementation. Floating-point sums are compared
with a tolerance, since the ranks are summed in a rank-dependent order; int8 is
compared exactly.
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

GROUP_SIZE_M = 4  # Config.swizzle_size default


def _shard(rank, M, N, dtype):
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(M, N, device="cuda", generator=g, dtype=torch.float32)
    if dtype == torch.int8:
        return (x * 30).round().clamp(-127, 127).to(dtype)
    return x.to(dtype)


def _owned(rank, world_size, M, N, block_size_m, block_size_n, distribution):
    """Mask of the output elements this rank owns, mirroring the kernel's tile assignment."""
    num_pid_m, num_pid_n = -(-M // block_size_m), -(-N // block_size_n)
    total = num_pid_m * num_pid_n
    if distribution == 0:
        tiles = range(rank, total, world_size)
    else:
        per_rank = -(-total // world_size)
        tiles = range(rank * per_rank, min(total, (rank + 1) * per_rank))
    mask = torch.zeros(M, N, dtype=torch.bool, device="cuda")
    for t in tiles:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        first_pid_m = (t // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (t % num_pid_in_group) % group_size_m
        pid_n = (t % num_pid_in_group) // group_size_m
        mask[pid_m * block_size_m : (pid_m + 1) * block_size_m, pid_n * block_size_n : (pid_n + 1) * block_size_n] = (
            True
        )
    return mask


def _check(shmem, M, N, dtype, config, calls=1):
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    src = shmem.zeros((M, N), dtype=dtype)
    src.copy_(_shard(rank, M, N, dtype))
    dst = shmem.zeros((M, N), dtype=dtype)
    sentinel = 77 if dtype == torch.int8 else float("nan")
    dst.fill_(sentinel)
    shmem.barrier()
    for _ in range(calls):
        shmem.ccl.reduce_scatter(dst, src, config=config)
    torch.cuda.synchronize()

    acc_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    expected = sum(_shard(r, M, N, dtype).to(acc_dtype) for r in range(world_size)).to(dtype)
    mask = _owned(rank, world_size, M, N, config.block_size_m, config.block_size_n, config.all_reduce_distribution)
    if dtype == torch.int8:
        assert torch.equal(dst[mask], expected[mask]), f"rank {rank}: owned tiles differ"
        assert (dst[~mask] == sentinel).all(), f"rank {rank}: wrote outside its tiles"
    else:
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(dst[mask], expected[mask], rtol=tol, atol=tol)
        assert torch.isnan(dst[~mask]).all(), f"rank {rank}: wrote outside its tiles"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, N",
    [
        (1024, 1024),  # whole tiles
        (100, 1024),  # ragged rows
        (513, 1030),  # ragged rows and columns
        (7, 300),  # smaller than one tile: most ranks own nothing
    ],
)
@pytest.mark.parametrize(
    "block_size_m, block_size_n, num_warps, comm_sms, num_stages",
    [
        (32, 128, 4, 64, 1),
        (64, 128, 16, 256, 1),
        # Pipelined, with few enough programs that the LDS ring wraps.
        (32, 128, 8, 4, 2),
        (16, 128, 4, 2, 4),
    ],
)
@pytest.mark.parametrize("distribution", [0, 1])
def test_reduce_scatter_tdm(dtype, M, N, block_size_m, block_size_n, num_warps, comm_sms, num_stages, distribution):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            reduce_scatter_variant="tdm",
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            num_warps=num_warps,
            comm_sms=comm_sms,
            num_stages=num_stages,
            all_reduce_distribution=distribution,
        )
        _check(shmem, M, N, dtype, config)
    finally:
        shmem.barrier()
        del shmem


def test_reduce_scatter_tdm_int8():
    """int8 accumulates in int32, so the sum is exact."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True, reduce_scatter_variant="tdm", block_size_m=32, block_size_n=128, comm_sms=8, num_stages=2
        )
        _check(shmem, 512, 1024, torch.int8, config)
    finally:
        shmem.barrier()
        del shmem


@pytest.mark.parametrize("num_stages", [1, 2])
def test_reduce_scatter_tdm_repeated_calls(num_stages):
    """Back-to-back calls on the same buffers stay correct."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        config = Config(
            use_gluon=True,
            reduce_scatter_variant="tdm",
            block_size_m=64,
            block_size_n=128,
            num_warps=16,
            comm_sms=16,
            num_stages=num_stages,
        )
        _check(shmem, 256, 1024, torch.bfloat16, config, calls=10)
    finally:
        shmem.barrier()
        del shmem


def test_reduce_scatter_tdm_rejects_oversized_ring():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        tile_bytes = 128 * 128 * 2
        max_lds = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block
        stages = max_lds // ((shmem.get_num_ranks() + 1) * tile_bytes) + 1
        config = Config(
            use_gluon=True, reduce_scatter_variant="tdm", block_size_m=128, block_size_n=128, num_stages=stages
        )
        src = shmem.zeros((128, 128), dtype=torch.bfloat16)
        dst = shmem.zeros((128, 128), dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="bytes of LDS"):
            shmem.ccl.reduce_scatter(dst, src, config=config)
    finally:
        shmem.barrier()
        del shmem


def test_reduce_scatter_tdm_requires_gluon():
    with pytest.raises(ValueError, match="use_gluon=True"):
        Config(reduce_scatter_variant="tdm")
