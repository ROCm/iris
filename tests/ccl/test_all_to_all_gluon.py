# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-to-all collective operation using Gluon with traffic shaping.
"""

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


def _all_to_all(src, stage_buf, result, shmem, config, async_op):
    """Stage src into the input buffer, then all-to-all. Module-level (no closure
    over shmem) so the test can ``del shmem`` for IPC cleanup. triton
    (use_gluon=False) and gluon (use_gluon=True) both dispatch through
    iris.ccl.all_to_all; async_op=True skips the capture-illegal trailing barrier."""
    stage_buf.copy_(src)
    shmem.ccl.all_to_all(result, stage_buf, config=config, async_op=async_op)


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
@pytest.mark.parametrize("impl", ["triton", "gluon"])
@pytest.mark.parametrize("mode", ["eager_barrier", "eager_nobarrier", "graph"])
@pytest.mark.parametrize("vary", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("M, N", [(128, 1024), (256, 1024)])
def test_all_to_all_gluon(impl, mode, vary, dtype, M, N):
    """Drive all-to-all across impl x mode x vary and check the exchanged output.
    mode/vary as in test_all_gather_gluon. No torch arm — torch.distributed
    all_to_all uses a different (row-split) layout and can't share this harness;
    eager_barrier is the reference (correct when properly synced).

    Layout is iris's (M, N*world_size) column-chunks: rank r fills its whole input
    with 1 + r + replay%16, so output chunk c (columns [c*N:(c+1)*N]) must equal
    1 + c + replay%16 (chunk c is rank c's data) — any >=1 mismatch is a real drop.
    Per-source-chunk fail tallies show which chunks dropped."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    # Resolve mode up front; the body runs straight-line off these.
    async_op = mode != "eager_barrier"
    capture = mode == "graph"

    shmem = iris.iris(2**33)  # 8 GB
    rank, world_size = shmem.get_rank(), shmem.get_num_ranks()
    torch.cuda.set_device(rank)
    width = N * world_size
    src = torch.empty((M, width), dtype=dtype, device=f"cuda:{rank}")
    stage_buf = shmem.zeros((M, width), dtype=dtype)
    result = shmem.zeros((M, width), dtype=dtype)
    config = Config(use_gluon=(impl == "gluon"))
    shmem.barrier()

    def fill_src(replay):
        src.fill_(float(1 + rank + (replay % 16)))

    # Warmup (runs lazy JIT/setup), then capture the step once if in graph mode.
    fill_src(0)
    _all_to_all(src, stage_buf, result, shmem, config, async_op)
    torch.cuda.synchronize()
    shmem.barrier()

    graph = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            _all_to_all(src, stage_buf, result, shmem, config, async_op)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)

    atol = 0.5  # exact integer inputs; >=1 mismatch is a real drop
    failures = []  # (step, max|diff|, bad_chunks)
    chunk_fail = [0] * world_size  # steps each source chunk dropped
    try:
        for i in range(NUM_REPLAYS):
            replay = i if vary else 0
            fill_src(replay)
            if capture:
                graph.replay()
            else:
                _all_to_all(src, stage_buf, result, shmem, config, async_op)
            torch.cuda.synchronize()
            diffs = [
                torch.abs(result[:, c * N : (c + 1) * N] - float(1 + c + (replay % 16))).max().item()
                for c in range(world_size)
            ]
            bad = [c for c in range(world_size) if diffs[c] > atol]
            for c in bad:
                chunk_fail[c] += 1
            if bad:
                failures.append((i, round(max(diffs[c] for c in bad), 4), bad))
        print(
            f"[rank {rank}] all_to_all impl={impl} mode={mode} vary={vary} dtype={dtype} "
            f"{M}x{width}: {NUM_REPLAYS - len(failures)}/{NUM_REPLAYS} ok; "
            f"per-source-chunk fail counts={chunk_fail}" + (f"; first FAIL={failures[0]}" if failures else ""),
            flush=True,
        )
        assert not failures, (
            f"impl={impl} mode={mode} vary={vary} dtype={dtype} {M}x{width}: "
            f"{len(failures)}/{NUM_REPLAYS} steps wrong (first {failures[0]}; per-source-chunk "
            f"fail counts={chunk_fail})."
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
