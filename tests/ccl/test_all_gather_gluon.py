# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation using Gluon.
"""

import os

import pytest
import torch
import torch.distributed as dist

try:
    import iris
    from iris.ccl import Config
    from triton.experimental import gluon  # noqa: F401

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


NUM_REPLAYS = 200


def _all_gather(impl, src, stage_buf, result, shmem, config, async_op):
    """Stage src into the input buffer, then all-gather. Module-level (no closure
    over shmem) so the test can ``del shmem`` for IPC cleanup."""
    stage_buf.copy_(src)
    if impl == "torch":
        dist.all_gather_into_tensor(result, stage_buf)
    else:
        shmem.ccl.all_gather(result, stage_buf, config=config, async_op=async_op)


def _make_buffers(impl, shmem, rank, world_size, M, N, dtype, block_size_m, block_size_n):
    """Resolve impl -> (stage_buf, result, config) in one place: torch uses plain
    device tensors and no config; the iris backends use symmetric-heap buffers and
    a use_gluon config. Output is (world_size * M, N) — block r holds rank r's input."""
    if impl == "torch":
        stage = torch.empty((M, N), dtype=dtype, device=f"cuda:{rank}")
        result = torch.empty((world_size * M, N), dtype=dtype, device=f"cuda:{rank}")
        return stage, result, None
    stage = shmem.zeros((M, N), dtype=dtype)
    result = shmem.zeros((world_size * M, N), dtype=dtype)
    config = Config(use_gluon=(impl == "gluon"), block_size_m=block_size_m, block_size_n=block_size_n)
    return stage, result, config


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
@pytest.mark.parametrize("impl", ["torch", "triton", "gluon"])
@pytest.mark.parametrize("mode", ["eager_barrier", "eager_nobarrier", "graph"])
@pytest.mark.parametrize("vary", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
# block_size_n must be a multiple of threads_per_warp * num_warps (256 at defaults).
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [(64, 8192, 32, 1024), (256, 8192, 32, 1024)],
)
def test_all_gather_gluon(impl, mode, vary, dtype, M, N, block_size_m, block_size_n):
    """Drive all-gather across impl x mode x vary and check the gathered output.

    mode: eager_barrier (async_op=False, trailing ctx.barrier()), eager_nobarrier
    (async_op=True, no barrier), graph (HIP-graph capture+replay, async_op=True —
    the host barrier can't be captured). vary=False replays identical input;
    vary=True feeds a fresh input each step, surfacing stale cross-rank reads.

    Rank r fills its whole input with 1 + r + replay%16 (exact integers), so output
    block r must equal 1 + r + replay%16 — any >=1 mismatch is a real drop. torch
    and eager_barrier are the references; per-peer-slice fail tallies show which
    peers' slices dropped."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    if impl == "torch" and mode == "eager_nobarrier":
        pytest.skip("torch has no barrier knob; eager_barrier already covers eager torch")

    # Resolve (impl, mode) up front; the body runs straight-line off these.
    async_op = mode != "eager_barrier"
    capture = mode == "graph"

    # Size heap to fit input (M*N) + output (max_ranks*M*N) with headroom
    max_ranks = int(os.environ.get("WORLD_SIZE", 8))
    elem_size = torch.tensor([], dtype=dtype).element_size()
    needed = (1 + max_ranks) * M * N * elem_size
    heap_size = max(2**30, int(needed * 2))  # 2x headroom, minimum 1GB
    shmem = iris.iris(heap_size)
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    torch.cuda.set_device(rank)
    src = torch.empty((M, N), dtype=dtype, device=f"cuda:{rank}")
    stage_buf, result, config = _make_buffers(impl, shmem, rank, world_size, M, N, dtype, block_size_m, block_size_n)
    shmem.barrier()

    def fill_src(replay):
        src.fill_(float(1 + rank + (replay % 16)))

    # Warmup (runs lazy JIT/setup), then capture the step once if in graph mode.
    fill_src(0)
    _all_gather(impl, src, stage_buf, result, shmem, config, async_op)
    torch.cuda.synchronize()
    shmem.barrier()

    graph = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            _all_gather(impl, src, stage_buf, result, shmem, config, async_op)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)

    atol = 0.5  # exact integer inputs; >=1 mismatch is a real drop
    failures = []  # (step, max|diff|, bad_slices)
    block_fail = [0] * world_size  # steps each peer slice dropped
    try:
        for i in range(NUM_REPLAYS):
            replay = i if vary else 0
            fill_src(replay)
            if capture:
                graph.replay()
            else:
                _all_gather(impl, src, stage_buf, result, shmem, config, async_op)
            torch.cuda.synchronize()
            diffs = [
                torch.abs(result[r * M : (r + 1) * M] - float(1 + r + (replay % 16))).max().item()
                for r in range(world_size)
            ]
            bad = [r for r in range(world_size) if diffs[r] > atol]
            for r in bad:
                block_fail[r] += 1
            if bad:
                failures.append((i, round(max(diffs[r] for r in bad), 4), bad))
        print(
            f"[rank {rank}] all_gather impl={impl} mode={mode} vary={vary} dtype={dtype} "
            f"{M}x{N}: {NUM_REPLAYS - len(failures)}/{NUM_REPLAYS} ok; "
            f"per-peer-slice fail counts={block_fail}" + (f"; first FAIL={failures[0]}" if failures else ""),
            flush=True,
        )
        assert not failures, (
            f"impl={impl} mode={mode} vary={vary} dtype={dtype} {M}x{N}: "
            f"{len(failures)}/{NUM_REPLAYS} steps wrong (first {failures[0]}; per-peer-slice "
            f"fail counts={block_fail})."
        )
    finally:
        if graph is not None:
            del graph
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()
