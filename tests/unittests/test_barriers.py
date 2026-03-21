# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import gc
from typing import Literal

import pytest
import torch
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
    """Verify device barrier reuses the same flags tensor across calls."""
    shmem = iris.iris(1 << 20)
    _call_barrier(shmem, barrier_type)

    try:
        shmem.device_barrier()
        assert None in shmem._device_barrier_state
        flags = shmem._device_barrier_state[None]
        flags_ptr = flags.data_ptr()

        for _ in range(n):
            shmem.device_barrier()
            assert shmem._device_barrier_state[None].data_ptr() == flags_ptr
    finally:
        _call_barrier(shmem, barrier_type)
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
