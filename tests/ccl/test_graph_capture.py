# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Graph capture correctness tests for all iris CCL collectives.

Each test: warmup → graph capture → 50 replays with vary=True
(different input data each replay). Compares iris output against
torch.distributed reference on every replay.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


NUM_REPLAYS = 50


@pytest.fixture(scope="module")
def ctx():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    shmem = iris.iris(heap_size=2**31)
    yield shmem
    del shmem


def _graph_capture_test(ctx, collective_fn, ref_fn, input_shape, output_shape, dtype=torch.bfloat16):
    """Shared graph capture test pattern for all collectives."""
    rank = ctx.get_rank()
    ws = ctx.get_num_ranks()
    device = f"cuda:{rank}"

    user_input = torch.randn(*input_shape, dtype=dtype, device=device)
    user_output = torch.zeros(*output_shape, dtype=dtype, device=device)

    # Eager warmup
    collective_fn(user_output, user_input)
    torch.cuda.synchronize()

    # Graph capture
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.stream(stream):
        collective_fn(user_output, user_input)

    torch.cuda.synchronize()

    with torch.cuda.graph(graph, stream=stream):
        collective_fn(user_output, user_input)

    torch.cuda.synchronize()

    # Replay with varying data
    for i in range(NUM_REPLAYS):
        new_data = torch.ones(*input_shape, dtype=dtype, device=device) * (rank + 1) * (i + 1)
        user_input.copy_(new_data)

        graph.replay()
        torch.cuda.synchronize()

        ref_output = torch.zeros(*output_shape, dtype=dtype, device=device)
        ref_fn(ref_output, new_data)
        torch.cuda.synchronize()

        if not torch.allclose(user_output, ref_output, atol=1e-2, rtol=1e-2):
            max_diff = (user_output - ref_output).abs().max().item()
            pytest.fail(
                f"Rank {rank}: graph replay {i} mismatch, max_diff={max_diff}"
            )


class TestAllReduceGraphCapture:
    def test_graph_capture(self, ctx):
        M, N = 32, 64
        config = Config(all_reduce_variant="one_shot")

        def collective(out, inp):
            ctx.ccl.all_reduce(out, inp, config=config)

        def reference(out, inp):
            out.copy_(inp)
            dist.all_reduce(out)

        _graph_capture_test(ctx, collective, reference, (M, N), (M, N))


class TestReduceScatterGraphCapture:
    def test_graph_capture(self, ctx):
        ws = ctx.get_num_ranks()
        M, N = 32, 64

        def collective(out, inp):
            ctx.ccl.reduce_scatter(out, inp)

        def reference(out, inp):
            dist.reduce_scatter_tensor(out, inp)

        _graph_capture_test(ctx, collective, reference, (M, N), (M, N))


class TestBroadcastGraphCapture:
    def test_graph_capture(self, ctx):
        M, N = 32, 64

        def collective(out, inp):
            ctx.ccl.broadcast(out, inp, src=0)

        def reference(out, inp):
            out.copy_(inp)
            dist.broadcast(out, src=0)

        _graph_capture_test(ctx, collective, reference, (M, N), (M, N))


class TestAllGatherGraphCapture:
    def test_graph_capture(self, ctx):
        ws = ctx.get_num_ranks()
        M, N = 32, 64

        def collective(out, inp):
            ctx.ccl.all_gather(out, inp)

        def reference(out, inp):
            dist.all_gather_into_tensor(out, inp)

        _graph_capture_test(ctx, collective, reference, (M, N), (ws * M, N))


class TestAllToAllGraphCapture:
    def test_graph_capture(self, ctx):
        ws = ctx.get_num_ranks()
        M, N = 32, 64

        def collective(out, inp):
            ctx.ccl.all_to_all(out, inp)

        def reference(out, inp):
            dist.all_to_all_single(out, inp)

        _graph_capture_test(ctx, collective, reference, (M, N * ws), (M, N * ws))


class TestReduceGraphCapture:
    def test_graph_capture(self, ctx):
        M, N = 32, 64

        def collective(out, inp):
            ctx.ccl.reduce(out, inp, root=0)

        def reference(out, inp):
            out.copy_(inp)
            dist.reduce(out, dst=0)

        _graph_capture_test(ctx, collective, reference, (M, N), (M, N))


class TestBarrierGraphCapture:
    def test_graph_capture(self, ctx):
        rank = ctx.get_rank()
        device = f"cuda:{rank}"

        flag = torch.zeros(1, dtype=torch.int32, device=device)

        def barrier_fn():
            ctx.ccl.barrier()

        # Warmup
        for _ in range(5):
            barrier_fn()
        torch.cuda.synchronize()

        # Graph capture
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(stream):
            barrier_fn()
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, stream=stream):
            barrier_fn()
        torch.cuda.synchronize()

        # Replay
        for i in range(NUM_REPLAYS):
            graph.replay()
            torch.cuda.synchronize()
