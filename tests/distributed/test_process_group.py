# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the iris torch.distributed backend.

Run with torchrun::

    torchrun --nproc_per_node=4 tests/run_tests_distributed.py \\
        tests/distributed/test_process_group.py -v --tb=short -x

These tests initialise ``dist.init_process_group(backend="nccl")``
(handled by the test runner), then create an ``IrisProcessGroup``
directly and exercise the collective operations.  A separate
"integration" test section verifies that the ``backend="iris"``
registration path works end-to-end.
"""

import pytest
import torch
import torch.distributed as dist


def _skip_if_not_distributed():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _world():
    """Return (rank, world_size)."""
    _skip_if_not_distributed()
    return dist.get_rank(), dist.get_world_size()


# ---------------------------------------------------------------------------
# Helpers -- create an IrisProcessGroup on top of the already-initialised PG
# ---------------------------------------------------------------------------


def _make_iris_pg():
    """Construct an IrisProcessGroup wrapping the current distributed state."""
    from iris.distributed.process_group import IrisProcessGroup

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    store = dist.distributed_c10d._get_default_store()
    timeout = dist.distributed_c10d._get_default_timeout()
    return IrisProcessGroup(store, rank, world_size, timeout)


# ===================================================================
# All-Reduce
# ===================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_allreduce(dtype):
    """Each rank contributes (rank+1); result should be sum(1..world_size)."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    tensor = torch.full((64, 32), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")

    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    work = pg.allreduce([tensor], opts)
    work.wait()

    expected = sum(range(1, world_size + 1))
    assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-2), (
        f"rank {rank}: expected {expected}, got {tensor[0, 0].item()}"
    )


# ===================================================================
# All-Gather (list variant)
# ===================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_allgather(dtype):
    """Each rank sends (rank+1); verify gathered chunks."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M, N = 32, 16
    input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    work = pg.allgather([output_list], [input_tensor])
    work.wait()

    for i in range(world_size):
        expected_val = float(i + 1)
        assert torch.allclose(output_list[i], torch.full_like(output_list[i], expected_val), atol=1e-2), (
            f"rank {rank}: chunk {i} expected {expected_val}, got {output_list[i][0, 0].item()}"
        )


# ===================================================================
# All-Gather (flat / into_tensor variant)
# ===================================================================


def test_allgather_into_tensor():
    """all_gather_into_tensor style -- flat output."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M, N = 32, 16
    dtype = torch.float32
    input_tensor = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    work = pg._allgather_base(output_tensor, input_tensor)
    work.wait()

    for i in range(world_size):
        chunk = output_tensor[i * M : (i + 1) * M]
        expected_val = float(i + 1)
        assert torch.allclose(chunk, torch.full_like(chunk, expected_val), atol=1e-2), (
            f"rank {rank}: chunk {i} expected {expected_val}, got {chunk[0, 0].item()}"
        )


# ===================================================================
# Reduce-Scatter (list variant)
# ===================================================================


def test_reduce_scatter():
    """Reduce then scatter -- each rank gets the sum of its chunk."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M, N = 32, 16
    dtype = torch.float32

    # Each rank contributes world_size chunks, each filled with (rank+1)
    input_list = [torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    output_tensor = torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}")

    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    work = pg.reduce_scatter([output_tensor], [input_list], opts)
    work.wait()

    expected = sum(range(1, world_size + 1))
    assert torch.allclose(output_tensor, torch.full_like(output_tensor, expected), atol=1e-2), (
        f"rank {rank}: expected {expected}, got {output_tensor[0, 0].item()}"
    )


# ===================================================================
# Reduce-Scatter (flat / into_tensor variant)
# ===================================================================


def test_reduce_scatter_tensor():
    """reduce_scatter_tensor -- flat input/output."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M_per_rank, N = 32, 16
    dtype = torch.float32

    input_tensor = torch.full((world_size * M_per_rank, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
    output_tensor = torch.zeros(M_per_rank, N, dtype=dtype, device=f"cuda:{rank}")

    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    work = pg._reduce_scatter_base(output_tensor, input_tensor, opts)
    work.wait()

    expected = sum(range(1, world_size + 1))
    assert torch.allclose(output_tensor, torch.full_like(output_tensor, expected), atol=1e-2), (
        f"rank {rank}: expected {expected}, got {output_tensor[0, 0].item()}"
    )


# ===================================================================
# All-to-All (list variant)
# ===================================================================


def test_alltoall():
    """Each rank sends its rank value to all; rank i receives values from all ranks."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M, N = 16, 8
    dtype = torch.float32

    input_list = [torch.full((M, N), float(rank), dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    opts = dist.AllToAllOptions()
    work = pg.alltoall(output_list, input_list, opts)
    work.wait()

    # Each output_list[i] should contain the value i (sent by rank i)
    for i in range(world_size):
        expected_val = float(i)
        assert torch.allclose(output_list[i], torch.full_like(output_list[i], expected_val), atol=1e-2), (
            f"rank {rank}: from rank {i} expected {expected_val}, got {output_list[i][0, 0].item()}"
        )


# ===================================================================
# All-to-All (flat / base variant)
# ===================================================================


def test_alltoall_base():
    """all_to_all with single concatenated tensors."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    M, N = 16, 8
    dtype = torch.float32

    # Concatenated input: (M, N * world_size)
    input_tensor = torch.full((M, N * world_size), float(rank), dtype=dtype, device=f"cuda:{rank}")
    output_tensor = torch.zeros(M, N * world_size, dtype=dtype, device=f"cuda:{rank}")

    opts = dist.AllToAllOptions()
    work = pg.alltoall_base(output_tensor, input_tensor, [], [], opts)
    work.wait()

    # output[:, i*N:(i+1)*N] should contain value i (from rank i)
    for i in range(world_size):
        chunk = output_tensor[:, i * N : (i + 1) * N]
        expected_val = float(i)
        assert torch.allclose(chunk, torch.full_like(chunk, expected_val), atol=1e-2), (
            f"rank {rank}: chunk {i} expected {expected_val}, got {chunk[0, 0].item()}"
        )


# ===================================================================
# Barrier
# ===================================================================


def test_barrier():
    """Barrier should complete without error."""
    _world()
    pg = _make_iris_pg()
    work = pg.barrier()
    work.wait()


# ===================================================================
# Broadcast
# ===================================================================


def test_broadcast():
    """Broadcast from rank 0."""
    rank, world_size = _world()
    pg = _make_iris_pg()

    dtype = torch.float32
    if rank == 0:
        tensor = torch.full((32, 16), 42.0, dtype=dtype, device=f"cuda:{rank}")
    else:
        tensor = torch.zeros(32, 16, dtype=dtype, device=f"cuda:{rank}")

    opts = dist.BroadcastOptions()
    opts.rootRank = 0
    work = pg.broadcast([tensor], opts)
    work.wait()

    assert torch.allclose(tensor, torch.full_like(tensor, 42.0), atol=1e-2), (
        f"rank {rank}: expected 42.0, got {tensor[0, 0].item()}"
    )


# ===================================================================
# Backend name
# ===================================================================


def test_backend_name():
    """Verify getBackendName returns 'iris'."""
    _world()
    pg = _make_iris_pg()
    assert pg.getBackendName() == "iris"
