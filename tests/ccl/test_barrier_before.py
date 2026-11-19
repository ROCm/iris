# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Each collective must wait for every rank's earlier work before it starts.

The loop never calls a host barrier between iterations. In each one a different
rank is slow: it delays writing its input, which a pull kernel would read stale
without the barrier, and delays reading its output, which a push kernel would
overwrite with the next iteration's data.
"""

import pytest
import torch
import torch.distributed as dist
import iris

# Long enough that the other ranks' kernels finish while the slow rank sleeps
SLEEP_CYCLES = 20_000_000
# N splits into whole 128-column chunks at up to 8 ranks for all-to-all
M, N = 256, 1024


def _value(rank, i, world_size):
    return float(rank + world_size * i + 1)


def _run(shmem, collective, make_output, expected, iters):
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    src = shmem.zeros((M, N), dtype=torch.float32)
    dst = make_output(shmem)
    seen = []
    shmem.barrier()
    for i in range(iters):
        if rank == i % world_size:
            torch.cuda._sleep(SLEEP_CYCLES)
        src.fill_(_value(rank, i, world_size))
        collective(shmem, dst, src)
        if rank == (i + 1) % world_size:
            torch.cuda._sleep(SLEEP_CYCLES)
        seen.append(dst.clone())
    torch.cuda.synchronize()
    shmem.barrier()
    for i, out in enumerate(seen):
        expected(out, i, rank, world_size)


def _all_gather(shmem, dst, src):
    shmem.ccl.all_gather(dst, src)


def _all_to_all(shmem, dst, src):
    shmem.ccl.all_to_all(dst, src)


def _all_reduce(shmem, dst, src):
    shmem.ccl.all_reduce(dst, src)


def _reduce_scatter(shmem, dst, src):
    # Only the owned tiles are written, so mark the rest. Reduce-scatter pulls:
    # peers read this rank's input but never write its output, so only this
    # rank's own kernel can race with this fill, and the stream orders that
    dst.fill_(float("nan"))
    shmem.ccl.reduce_scatter(dst, src)


def _expect_all_gather(out, i, rank, world_size):
    for r in range(world_size):
        block = out[r * M : (r + 1) * M]
        assert (block == _value(r, i, world_size)).all(), f"rank {rank}, iteration {i}: block from rank {r} is stale"


def _expect_all_to_all(out, i, rank, world_size):
    chunk = N // world_size
    for r in range(world_size):
        block = out[:, r * chunk : (r + 1) * chunk]
        assert (block == _value(r, i, world_size)).all(), f"rank {rank}, iteration {i}: chunk from rank {r} is stale"


def _expect_sum(out, i, rank, world_size, owned_only=False):
    total = sum(_value(r, i, world_size) for r in range(world_size))
    if owned_only:
        out = out[~torch.isnan(out)]
        assert out.numel() > 0, f"rank {rank}, iteration {i}: wrote nothing"
    assert (out == total).all(), f"rank {rank}, iteration {i}: sum is stale"


CASES = {
    "all_gather": (_all_gather, lambda s: s.zeros((s.get_num_ranks() * M, N), dtype=torch.float32), _expect_all_gather),
    "all_to_all": (_all_to_all, lambda s: s.zeros((M, N), dtype=torch.float32), _expect_all_to_all),
    "all_reduce": (_all_reduce, lambda s: s.zeros((M, N), dtype=torch.float32), _expect_sum),
    "reduce_scatter": (
        _reduce_scatter,
        lambda s: s.zeros((M, N), dtype=torch.float32),
        lambda out, i, rank, ws: _expect_sum(out, i, rank, ws, owned_only=True),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_collective_waits_for_earlier_work(name):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(2**30)
    try:
        collective, make_output, expected = CASES[name]
        # Two rounds of the slow rank moving through every rank
        _run(shmem, collective, make_output, expected, iters=2 * shmem.get_num_ranks())
    finally:
        shmem.barrier()
        del shmem
