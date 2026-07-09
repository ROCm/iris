# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for barrier collective operation.

Tests the iris CCL barrier (ported from RCCL's zero-byte AllReduce)
against torch.distributed for correctness.
"""

import gc

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris


@triton.jit
def _read_remote_kernel(
    buf_ptr,
    result_ptr,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    """Read data from a remote rank's buffer."""
    offsets = tl.arange(0, BLOCK_SIZE)
    data = iris.load(buf_ptr + offsets, cur_rank, remote_rank, heap_bases)
    tl.store(result_ptr + offsets, data)


@triton.jit
def _write_kernel(
    buf_ptr,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    """Write a value to a local buffer."""
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.full([BLOCK_SIZE], value, dtype=tl.float32)
    tl.store(buf_ptr + offsets, data)


def test_barrier_basic():
    """Test that barrier completes without error."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    try:
        shmem.barrier()
        shmem.ccl.barrier()
        shmem.ccl.barrier()
        shmem.ccl.barrier()
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_multiple():
    """Test multiple consecutive barriers."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    try:
        for _ in range(10):
            shmem.ccl.barrier()
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_async():
    """Test async barrier (no trailing host barrier)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    try:
        shmem.ccl.barrier(async_op=True)
        torch.cuda.synchronize()
        # Still need to sync ranks for test cleanup
        shmem.barrier()
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("N", [1, 64, 256, 1024])
@pytest.mark.parametrize("rounds", [1, 3, 5])
def test_barrier_data_visibility(N, rounds):
    """
    Verify that data written before a barrier is visible after the barrier.

    This is the core correctness test: each rank writes to its own buffer,
    then after barrier, reads from a neighbor to verify visibility.

    This matches RCCL's barrier semantics where all prior writes from
    all ranks are guaranteed to be visible after barrier completion.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    neighbor = (rank + 1) % num_ranks

    buf = shmem.zeros((N,), dtype=torch.float32)
    result = shmem.zeros((N,), dtype=torch.float32)

    try:
        for i in range(rounds):
            # Each rank writes a unique value to its buffer
            val = float(rank + i * 100)
            buf.fill_(val)

            # Barrier ensures all writes are visible
            shmem.ccl.barrier()

            # Read from neighbor — should see neighbor's value
            _read_remote_kernel[(1,)](
                buf, result, rank, neighbor, N, heap_bases
            )
            torch.cuda.synchronize()

            expected_val = float(neighbor + i * 100)
            expected = torch.full((N,), expected_val, dtype=torch.float32, device="cuda")
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

            # Barrier before next iteration
            shmem.ccl.barrier()
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 64),
        (128, 128),
        (256, 256),
        (1024, 1024),
    ],
)
def test_barrier_with_allreduce_correctness(dtype, M, N):
    """
    Test barrier correctness by running allreduce before and after barrier.

    Since RCCL's barrier is a zero-byte AllReduce, we verify that a
    barrier between two allreduce operations correctly synchronizes
    the ranks and produces correct results.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    from iris.ccl import Config

    heap_size = 2 ** 33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    try:
        # Setup tensors
        inp = shmem.zeros((M, N), dtype=dtype)
        out = shmem.zeros((M, N), dtype=dtype)
        inp.fill_(float(rank + 1))

        # PyTorch reference
        ref_inp = torch.full((M, N), float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
        ref_out = ref_inp.clone()
        dist.all_reduce(ref_out, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Iris: allreduce + barrier + verify
        config = Config(all_reduce_variant="two_shot", block_size_m=32, block_size_n=64)
        workspace = shmem.ccl.all_reduce_preamble(out, inp, config=config)
        shmem.barrier()
        shmem.ccl.all_reduce(out, inp, config=config, workspace=workspace)
        torch.cuda.synchronize()

        # Barrier to ensure all ranks have completed allreduce
        shmem.ccl.barrier()

        # Compare
        atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        max_diff = torch.abs(out - ref_out).max().item()
        assert torch.allclose(out, ref_out, atol=atol), (
            f"Barrier+AllReduce mismatch: max_diff={max_diff}, "
            f"rank={rank}, dtype={dtype}, shape=({M},{N})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_state_reuse():
    """Verify that CCL barrier reuses the same flags tensor across calls."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    try:
        # First call creates the state
        shmem.ccl.barrier()
        assert hasattr(shmem, '_ccl_barrier_state')
        assert None in shmem._ccl_barrier_state
        flags = shmem._ccl_barrier_state[None]
        flags_ptr = flags.data_ptr()

        # Subsequent calls reuse the same tensor
        for _ in range(5):
            shmem.ccl.barrier()
            assert shmem._ccl_barrier_state[None].data_ptr() == flags_ptr
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_barrier_graph_capturable():
    """Test that the CCL barrier kernel is CUDA graph capturable."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    neighbor = (rank + 1) % num_ranks
    N = 64

    buf = shmem.zeros((N,), dtype=torch.float32)
    result = shmem.zeros((N,), dtype=torch.float32)

    try:
        capture_stream = torch.cuda.Stream()

        # Warmup
        buf.fill_(float(rank))
        with torch.cuda.stream(capture_stream):
            shmem.ccl.barrier(async_op=True)
            _read_remote_kernel[(1,)](buf, result, rank, neighbor, N, heap_bases)
            shmem.ccl.barrier(async_op=True)
        capture_stream.synchronize()
        shmem.barrier()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            shmem.ccl.barrier(async_op=True)
            _read_remote_kernel[(1,)](buf, result, rank, neighbor, N, heap_bases)
            shmem.ccl.barrier(async_op=True)

        # Replay with fresh data
        for i in range(3):
            val = float(rank + (i + 1) * 10)
            with torch.cuda.stream(capture_stream):
                buf.fill_(val)
                shmem.device_barrier()
                graph.replay()
            capture_stream.synchronize()
            shmem.barrier()

            expected = torch.full(
                (N,), float(neighbor + (i + 1) * 10),
                dtype=torch.float32, device="cuda"
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
