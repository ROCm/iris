# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import gc
from typing import Literal

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris


BarrierType = Literal["host", "device"]
BARRIER_TYPES: list[BarrierType] = ["host", "device"]


def _call_barrier(shmem: iris.Iris, barrier_type: BarrierType) -> None:
    if barrier_type == "host":
        shmem.barrier()
    else:
        shmem.device_barrier()


@triton.jit
def _read_remote_kernel(
    buf_ptr,
    result_ptr,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    data = iris.load(buf_ptr + offsets, cur_rank, remote_rank, heap_bases)
    tl.store(result_ptr + offsets, data)


@triton.jit
def _write_remote_kernel(
    buf_ptr,
    value,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.full([BLOCK_SIZE], value, dtype=tl.float32)
    iris.store(buf_ptr + offsets, data, cur_rank, remote_rank, heap_bases)


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_basic(barrier_type, n):
    shmem = iris.iris(1 << 20)
    _call_barrier(shmem, barrier_type)

    try:
        for _ in range(n):
            _call_barrier(shmem, barrier_type)
    finally:
        _call_barrier(shmem, barrier_type)
        del shmem
        gc.collect()


@pytest.mark.parametrize("n", [1, 2, 5, 10])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_state_reuse(barrier_type, n):
    """Verify device barrier uses the flags allocated in __init__ and never allocates."""
    shmem = iris.iris(1 << 20)
    _call_barrier(shmem, barrier_type)

    try:
        flags = shmem._device_barrier_flags
        flags_ptr = flags.data_ptr()
        assert flags.shape == (shmem.get_num_ranks(), shmem.get_num_ranks())
        before = shmem.zeros((1,), dtype=torch.int32).data_ptr()
        torch.cuda.synchronize()
        own_row = flags[shmem.get_rank()].clone()

        for _ in range(n):
            shmem.device_barrier()
            assert shmem._device_barrier_flags.data_ptr() == flags_ptr

        # Each barrier advances this rank's counter for every peer exactly once.
        torch.cuda.synchronize()
        expected = torch.full_like(own_row, n)
        expected[shmem.get_rank()] = 0
        torch.testing.assert_close(flags[shmem.get_rank()] - own_row, expected)

        # The gap between allocations is the same with or without barriers in between.
        after = shmem.zeros((1,), dtype=torch.int32).data_ptr()
        next_ = shmem.zeros((1,), dtype=torch.int32).data_ptr()
        assert after - before == next_ - after
    finally:
        _call_barrier(shmem, barrier_type)
        del shmem
        gc.collect()


def test_device_barrier_flags_do_not_use_heap_size():
    """The barrier flags Iris allocates for itself must not come out of the caller's heap_size."""
    heap_size = 1 << 20
    shmem = iris.iris(heap_size, allocator_type="torch")
    try:
        whole_heap = shmem.zeros((heap_size,), dtype=torch.int8)
        assert whole_heap.numel() == heap_size
        shmem.device_barrier()
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def _heap_offset(shmem, tensor):
    return tensor.data_ptr() - int(shmem.get_heap_bases()[shmem.get_rank()].item())


def _assert_heap_offsets_match(shmem):
    """Allocate collectively and check every rank got the same heap offset."""
    offset = _heap_offset(shmem, shmem.zeros((1,), dtype=torch.int32))
    offsets = [None] * shmem.get_num_ranks()
    dist.all_gather_object(offsets, offset)
    assert len(set(offsets)) == 1, f"heap offsets diverged across ranks: {offsets}"


def _halves(num_ranks):
    """Collectively create the groups [0, n/2) and [n/2, n); return this rank's."""
    half = num_ranks // 2
    lo = dist.new_group(list(range(half)))
    hi = dist.new_group(list(range(half, num_ranks)))
    return lo if dist.get_rank() < half else hi


def test_device_barrier_group_object_mismatch():
    """Ranks may pass different ProcessGroup objects for groups with the same members.

    The lower half reuses one group object; the upper half passes a freshly created,
    equal group on every call. Neither may allocate on the heap or wait on the
    other half.
    """
    if dist.get_world_size() < 4:
        pytest.skip("needs at least 4 ranks")
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    rank = shmem.get_rank()
    half = num_ranks // 2

    try:
        cached = _halves(num_ranks)
        for _ in range(3):
            fresh = dist.new_group(list(range(half, num_ranks)))
            shmem.device_barrier(group=cached if rank < half else fresh)
            torch.cuda.synchronize()
        shmem.device_barrier()
        torch.cuda.synchronize()
        _assert_heap_offsets_match(shmem)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("N", [1, 1024])
def test_device_barrier_overlapping_groups(N, rounds=4):
    """Interleave barriers over the world, contiguous halves and strided halves.

    After each barrier a rank reads a group peer's buffer, which that peer
    filled before entering the barrier. A barrier over one group must not let
    through a rank waiting on a barrier over another.
    """
    if dist.get_world_size() < 4 or dist.get_world_size() % 2:
        pytest.skip("needs an even number of ranks, at least 4")
    shmem = iris.iris(1 << 20)
    num_ranks = shmem.get_num_ranks()
    rank = shmem.get_rank()
    heap_bases = shmem.get_heap_bases()

    halves = _halves(num_ranks)
    even = dist.new_group(list(range(0, num_ranks, 2)))
    odd = dist.new_group(list(range(1, num_ranks, 2)))
    strided = even if rank % 2 == 0 else odd

    buf = shmem.zeros((N,), dtype=torch.float32)
    result = shmem.zeros((N,), dtype=torch.float32)
    shmem.barrier()

    try:
        for i in range(rounds):
            for group in (None, halves, strided):
                members = list(range(num_ranks)) if group is None else dist.get_process_group_ranks(group)
                peer = members[(members.index(rank) + 1) % len(members)]
                val = float(rank + i * 100)
                buf.fill_(val)
                shmem.device_barrier(group=group)
                _read_remote_kernel[(1,)](buf, result, rank, peer, N, heap_bases)
                # Keep the peer from refilling buf while this rank is still reading it.
                shmem.device_barrier(group=group)
                expected = torch.full((N,), float(peer + i * 100), dtype=torch.float32, device="cuda")
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        _assert_heap_offsets_match(shmem)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def _cross_rank_eager(
    shmem,
    barrier_type,
    op,
    num_barriers,
    rounds,
    N,
    rank,
    neighbor,
    writer,
    heap_bases,
    buf,
    result,
):
    if op == "load":
        for i in range(rounds):
            buf.fill_(float(rank + i * 100))

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            _read_remote_kernel[(1,)](
                buf,
                result,
                rank,
                neighbor,
                N,
                heap_bases,
            )

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            expected_val = float(neighbor + i * 100)
            expected = torch.full((N,), expected_val, dtype=torch.float32, device="cuda")
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    else:
        for i in range(rounds):
            buf.fill_(0.0)

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            write_val = float(rank + i * 100)
            _write_remote_kernel[(1,)](
                buf,
                write_val,
                rank,
                neighbor,
                N,
                heap_bases,
            )

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            expected_val = float(writer + i * 100)
            expected = torch.full((N,), expected_val, dtype=torch.float32, device="cuda")
            torch.testing.assert_close(buf, expected, rtol=0, atol=0)


def _cross_rank_graph(
    shmem,
    op,
    num_barriers,
    rounds,
    N,
    rank,
    neighbor,
    writer,
    heap_bases,
    buf,
    result,
):
    capture_stream = torch.cuda.Stream()

    if op == "load":
        buf.fill_(float(rank))

        # Warmup on capture stream.
        with torch.cuda.stream(capture_stream):
            for _ in range(num_barriers):
                shmem.device_barrier()
            _read_remote_kernel[(1,)](
                buf,
                result,
                rank,
                neighbor,
                N,
                heap_bases,
            )
            for _ in range(num_barriers):
                shmem.device_barrier()
        capture_stream.synchronize()

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(num_barriers):
                shmem.device_barrier()
            _read_remote_kernel[(1,)](
                buf,
                result,
                rank,
                neighbor,
                N,
                heap_bases,
            )
            for _ in range(num_barriers):
                shmem.device_barrier()

        # Replay with fresh data.
        for i in range(rounds):
            val = float(rank + (i + 1) * 10)
            with torch.cuda.stream(capture_stream):
                buf.fill_(val)
                shmem.device_barrier()
                graph.replay()
            capture_stream.synchronize()

            expected = torch.full(
                (N,),
                float(neighbor + (i + 1) * 10),
                dtype=torch.float32,
                device="cuda",
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    else:
        buf.fill_(0.0)

        # Warmup on capture stream.
        with torch.cuda.stream(capture_stream):
            for _ in range(num_barriers):
                shmem.device_barrier()
            _write_remote_kernel[(1,)](
                buf,
                float(rank),
                rank,
                neighbor,
                N,
                heap_bases,
            )
            for _ in range(num_barriers):
                shmem.device_barrier()
        capture_stream.synchronize()

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(num_barriers):
                shmem.device_barrier()
            _write_remote_kernel[(1,)](
                buf,
                float(rank),
                rank,
                neighbor,
                N,
                heap_bases,
            )
            for _ in range(num_barriers):
                shmem.device_barrier()

        # Replay and verify.
        for _ in range(rounds):
            with torch.cuda.stream(capture_stream):
                buf.fill_(0.0)
                shmem.device_barrier()
                graph.replay()
            capture_stream.synchronize()

            with torch.cuda.stream(capture_stream):
                shmem.device_barrier()
            capture_stream.synchronize()
            expected = torch.full((N,), float(writer), dtype=torch.float32, device="cuda")
            torch.testing.assert_close(buf, expected, rtol=0, atol=0)


# Host barrier is not graph-capturable (uses NCCL which crashes with
# hipErrorStreamCaptureUnsupported on ROCm). Skip host+graph combos.
@pytest.mark.parametrize("N", [1, 64, 256, 1024])
@pytest.mark.parametrize("num_barriers", [1, 2, 4])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("op", ["load", "store", "both"])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_cross_rank(barrier_type, op, mode, num_barriers, N, rounds=3):
    """Verify cross-rank data visibility after barrier.

    - op: load (iris.load from neighbor), store (iris.store to neighbor), or both
    - mode: eager (direct calls) or graph (CUDA graph capture + replay)
    - num_barriers: consecutive barriers to test idempotency
    - N: number of elements (must be power of 2 for Triton BLOCK_SIZE)
    - rounds: number of iterations with changing data (default 3)

    Each mode runs multiple rounds with changing data to stress correctness.
    Graph mode captures barrier + kernel into a CUDA graph, then replays
    with fresh data to verify correctness through the captured graph.
    """
    if mode == "graph" and barrier_type == "host":
        pytest.skip(
            "Host barrier uses NCCL which is not graph-capturable on ROCm. See https://github.com/ROCm/HIP/issues/3876"
        )

    shmem = iris.iris(1 << 20)
    _call_barrier(shmem, barrier_type)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    neighbor = (rank + 1) % num_ranks
    writer = (rank - 1 + num_ranks) % num_ranks

    buf = shmem.zeros((N,), dtype=torch.float32)
    result = shmem.zeros((N,), dtype=torch.float32)

    ops = ["load", "store"] if op == "both" else [op]

    try:
        for single_op in ops:
            if mode == "eager":
                _cross_rank_eager(
                    shmem,
                    barrier_type,
                    single_op,
                    num_barriers,
                    rounds,
                    N,
                    rank,
                    neighbor,
                    writer,
                    heap_bases,
                    buf,
                    result,
                )
            else:
                _cross_rank_graph(
                    shmem,
                    single_op,
                    num_barriers,
                    rounds,
                    N,
                    rank,
                    neighbor,
                    writer,
                    heap_bases,
                    buf,
                    result,
                )
    finally:
        _call_barrier(shmem, barrier_type)
        del shmem
        gc.collect()
