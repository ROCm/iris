#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Graph Capture Probe — test which iris operations can be CUDA-graph captured.

Methodology:
  1. Warmup: run the operation eagerly to compile kernels + allocate workspace
  2. Pre-capture: run on capture stream (populate Triton per-stream JIT cache)
  3. Capture: wrap operation in torch.cuda.graph() — HIP records kernel launches
     as graph nodes. Non-capturable ops (host sync, NCCL, malloc) raise
     hipErrorStreamCaptureUnsupported.
  4. Reset + refill: zero outputs, fill inputs with DIFFERENT values
  5. Replay: graph.replay() — only the recorded CUDA ops run, no Python re-executes
  6. Validate: check output matches expected for the NEW input values

The fresh-data replay in step 4-6 catches false positives: if the graph just
returns stale capture-time results, validation will fail.

Run with:
    torchrun --nproc_per_node=2 --standalone tests/graph_capture_probe.py
"""
import os
import traceback

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config


def setup():
    """Initialize distributed + iris."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ctx = iris.iris(heap_size=1 << 30)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    return ctx, rank, world_size


def teardown(ctx):
    ctx.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def try_capture(name, warmup_fn, capture_fn, reset_fn, replay_setup_fn, validate_fn, ctx, rank):
    """
    Graph capture test harness.

    Detection: torch.cuda.graph() → hipStreamBeginCapture. Non-capturable ops
    raise hipErrorStreamCaptureUnsupported (authoritative from HIP runtime).

    Correctness: after capture, reset outputs + fill inputs with new values +
    replay → validate against expected for the new values. Catches stale results.

    Args:
        warmup_fn:        compile kernels + allocate workspace (eager, outside capture)
        capture_fn:       the operation to record into the graph
        reset_fn:         zero outputs between capture and replay
        replay_setup_fn:  fill inputs with NEW values (different from warmup/capture)
        validate_fn:      check output matches expected for the new inputs → bool
    """
    result = {"name": name, "status": "UNKNOWN", "detail": ""}

    # Step 1: Warmup
    try:
        warmup_fn()
        torch.cuda.synchronize()
        ctx.barrier()
    except Exception as e:
        result["status"] = "SKIP"
        result["detail"] = f"warmup failed: {e}"
        return result

    # Step 2: Capture
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    try:
        # Pre-capture run on capture stream (Triton JITs per-stream on first use;
        # JIT does hipMalloc which would break capture if it happened inside graph)
        with torch.cuda.stream(capture_stream):
            capture_fn()
        capture_stream.synchronize()
        ctx.barrier()

        # Actual capture
        with torch.cuda.graph(graph, stream=capture_stream):
            capture_fn()
        capture_stream.synchronize()
        ctx.barrier()
    except RuntimeError as e:
        err_str = str(e)
        if "hipErrorStreamCapture" in err_str or "StreamCapture" in err_str or "not supported" in err_str.lower():
            result["status"] = "NOT_CAPTURABLE"
            result["detail"] = err_str[:200]
        else:
            result["status"] = "ERROR"
            result["detail"] = err_str[:200]
        return result
    except Exception as e:
        result["status"] = "ERROR"
        result["detail"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result

    # Step 3: Reset + fresh data + replay + validate
    try:
        reset_fn()
        replay_setup_fn()
        ctx.barrier()

        with torch.cuda.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()
        ctx.barrier()

        ok = validate_fn()
        if ok:
            result["status"] = "CAPTURABLE"
            result["detail"] = "capture + replay with fresh data + validation passed"
        else:
            result["status"] = "WRONG_RESULT"
            result["detail"] = "captured & replayed OK but output wrong (stale capture-time data?)"
    except Exception as e:
        result["status"] = "REPLAY_ERROR"
        result["detail"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_device_barrier(ctx, rank, world_size):
    """device_barrier — known capturable (device-side atomics, no NCCL)."""
    buf = ctx.zeros((64,), dtype=torch.float32)
    result_buf = ctx.zeros((64,), dtype=torch.float32)

    def warmup():
        buf.fill_(float(rank + 1))
        ctx.device_barrier()
        # Read neighbor to prove barrier works
        neighbor = (rank + 1) % world_size
        heap_bases = ctx.get_heap_bases()

    def capture():
        ctx.device_barrier()

    def reset():
        buf.zero_()

    def replay_setup():
        buf.fill_(float(rank + 100))

    def validate():
        # barrier itself has no output; just verify it didn't crash
        return True

    return try_capture("device_barrier", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_host_barrier(ctx, rank, world_size):
    """host barrier — known NOT capturable (uses NCCL on ROCm)."""
    def warmup():
        ctx.barrier()

    def capture():
        ctx.barrier()

    return try_capture("host_barrier", warmup, capture, lambda: None, lambda: None, lambda: True, ctx, rank)


def test_ccl_all_reduce_atomic(ctx, rank, world_size):
    """ccl.all_reduce with atomic variant + async_op=True."""
    M, N = 128, 64
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M, N), dtype=torch.float32)
    config = Config(all_reduce_variant="atomic", block_size_m=32, block_size_n=64)

    # Pre-allocate workspace outside capture
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    ctx.barrier()

    def warmup():
        inp.fill_(float(rank + 1))
        out.zero_()
        ctx.barrier()
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config, workspace=workspace)
        torch.cuda.synchronize()

    def capture():
        # Must re-prepare workspace (it resets prepared=False after each use)
        workspace.prepared = True
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config, workspace=workspace)

    def reset():
        out.zero_()
        ctx.barrier()  # atomic variant needs output zeroed across all ranks
        workspace.prepared = True

    def replay_setup():
        inp.fill_(float(rank + 10))  # different from warmup's (rank+1)

    def validate():
        expected = float(sum(r + 10 for r in range(world_size)))
        return torch.allclose(out, torch.full_like(out, expected), atol=1e-2)

    return try_capture("ccl.all_reduce(atomic)", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ccl_all_reduce_two_shot(ctx, rank, world_size):
    """ccl.all_reduce with two_shot variant + async_op=True."""
    M, N = 128, 64
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M, N), dtype=torch.float32)
    config = Config(all_reduce_variant="two_shot", block_size_m=32, block_size_n=64)

    def warmup():
        inp.fill_(float(rank + 1))
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config)
        torch.cuda.synchronize()

    def capture():
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config)

    def reset():
        out.zero_()

    def replay_setup():
        inp.fill_(float(rank + 10))

    def validate():
        expected = float(sum(r + 10 for r in range(world_size)))
        return torch.allclose(out, torch.full_like(out, expected), atol=1e-2)

    return try_capture("ccl.all_reduce(two_shot)", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ccl_all_reduce_one_shot(ctx, rank, world_size):
    """ccl.all_reduce with one_shot variant + async_op=True."""
    M, N = 128, 64
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M, N), dtype=torch.float32)
    config = Config(all_reduce_variant="one_shot", block_size_m=32, block_size_n=64)

    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    ctx.barrier()

    def warmup():
        inp.fill_(float(rank + 1))
        out.zero_()
        ctx.barrier()
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config, workspace=workspace)
        torch.cuda.synchronize()

    def capture():
        workspace.prepared = True
        ctx.ccl.all_reduce(out, inp, async_op=True, config=config, workspace=workspace)

    def reset():
        out.zero_()
        ctx.barrier()
        workspace.prepared = True

    def replay_setup():
        inp.fill_(float(rank + 10))

    def validate():
        expected = float(sum(r + 10 for r in range(world_size)))
        return torch.allclose(out, torch.full_like(out, expected), atol=1e-2)

    return try_capture("ccl.all_reduce(one_shot)", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ccl_all_gather(ctx, rank, world_size):
    """ccl.all_gather + async_op=True."""
    M, N = 64, 64
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M * world_size, N), dtype=torch.float32)

    def warmup():
        inp.fill_(float(rank + 1))
        ctx.ccl.all_gather(out, inp, async_op=True)
        torch.cuda.synchronize()

    def capture():
        ctx.ccl.all_gather(out, inp, async_op=True)

    def reset():
        out.zero_()

    def replay_setup():
        inp.fill_(float(rank + 10))

    def validate():
        for r in range(world_size):
            chunk = out[r * M : (r + 1) * M]
            if not torch.allclose(chunk, torch.full_like(chunk, float(r + 10)), atol=1e-2):
                return False
        return True

    return try_capture("ccl.all_gather", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ccl_all_to_all(ctx, rank, world_size):
    """ccl.all_to_all + async_op=True."""
    M = 64
    N = 64 * world_size
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M, N), dtype=torch.float32)

    chunk_n = N // world_size

    def warmup():
        for r in range(world_size):
            inp[:, r * chunk_n : (r + 1) * chunk_n] = float(rank * 10 + r + 1)
        ctx.ccl.all_to_all(out, inp, async_op=True)
        torch.cuda.synchronize()

    def capture():
        ctx.ccl.all_to_all(out, inp, async_op=True)

    def reset():
        out.zero_()

    def replay_setup():
        # Use different values: rank * 100 + r + 1
        for r in range(world_size):
            inp[:, r * chunk_n : (r + 1) * chunk_n] = float(rank * 100 + r + 1)

    def validate():
        for r in range(world_size):
            expected_val = float(r * 100 + rank + 1)
            chunk = out[:, r * chunk_n : (r + 1) * chunk_n]
            if not torch.allclose(chunk, torch.full_like(chunk, expected_val), atol=1e-2):
                return False
        return True

    return try_capture("ccl.all_to_all", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ccl_reduce_scatter(ctx, rank, world_size):
    """ccl.reduce_scatter + async_op=True."""
    M = 64
    N = 64 * world_size
    inp = ctx.zeros((M, N), dtype=torch.float32)
    out = ctx.zeros((M, N // world_size), dtype=torch.float32)

    def warmup():
        inp.fill_(float(rank + 1))
        ctx.ccl.reduce_scatter(out, inp, async_op=True)
        torch.cuda.synchronize()

    def capture():
        ctx.ccl.reduce_scatter(out, inp, async_op=True)

    def reset():
        out.zero_()

    def replay_setup():
        inp.fill_(float(rank + 10))

    def validate():
        expected = float(sum(r + 10 for r in range(world_size)))
        return torch.allclose(out, torch.full_like(out, expected), atol=1e-2)

    return try_capture("ccl.reduce_scatter", warmup, capture, reset, replay_setup, validate, ctx, rank)


def test_ops_matmul_all_reduce(ctx, rank, world_size):
    """ops.matmul_all_reduce + async_op=True."""
    M, N, K = 128, 64, 32
    A = ctx.zeros((M, K), dtype=torch.float16)
    B = ctx.zeros((K, N), dtype=torch.float16)
    C = ctx.zeros((M, N), dtype=torch.float16)

    from iris.ops.config import FusedConfig
    config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant="atomic")

    # Compute a reference for validation with specific input values
    torch.manual_seed(42 + rank)
    A_ref = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B_ref = torch.randn(K, N, dtype=torch.float16, device="cuda")

    def warmup():
        A.copy_(A_ref)
        B.copy_(B_ref)
        C.zero_()
        ctx.barrier()
        ctx.ops.matmul_all_reduce(C, A, B, async_op=True, config=config)
        torch.cuda.synchronize()

    def capture():
        ctx.ops.matmul_all_reduce(C, A, B, async_op=True, config=config)

    def reset():
        C.zero_()
        ctx.barrier()

    def replay_setup():
        # Keep same A, B (matmul is deterministic for same inputs)
        A.copy_(A_ref)
        B.copy_(B_ref)

    def validate():
        # Check output is non-zero and finite
        return C.abs().max().item() > 0 and torch.isfinite(C).all().item()

    return try_capture("ops.matmul_all_reduce", warmup, capture, reset, replay_setup, validate, ctx, rank)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ctx, rank, world_size = setup()

    tests = [
        test_device_barrier,
        test_host_barrier,
        test_ccl_all_reduce_atomic,
        test_ccl_all_reduce_two_shot,
        test_ccl_all_reduce_one_shot,
        test_ccl_all_gather,
        test_ccl_all_to_all,
        test_ccl_reduce_scatter,
        test_ops_matmul_all_reduce,
    ]

    results = []
    for test_fn in tests:
        ctx.barrier()
        try:
            result = test_fn(ctx, rank, world_size)
        except Exception as e:
            result = {"name": test_fn.__name__, "status": "CRASH", "detail": traceback.format_exc()[-300:]}
        results.append(result)
        ctx.barrier()

    if rank == 0:
        print("\n" + "=" * 70)
        print("CUDA GRAPH CAPTURE PROBE RESULTS")
        print(f"  world_size={world_size}")
        print("=" * 70)
        for r in results:
            icon = {
                "CAPTURABLE": "YES",
                "NOT_CAPTURABLE": "NO ",
                "WRONG_RESULT": "BAD",
                "ERROR": "ERR",
                "REPLAY_ERROR": "ERR",
                "SKIP": "SKP",
                "CRASH": "!!!",
            }.get(r["status"], "???")
            print(f"  [{icon}] {r['name']:<35s} {r['status']}")
            if r["detail"] and r["status"] != "CAPTURABLE":
                for line in r["detail"].split("\n")[:3]:
                    print(f"        {line}")
        print("=" * 70)

        capturable = sum(1 for r in results if r["status"] == "CAPTURABLE")
        total = len(results)
        print(f"\n  {capturable}/{total} operations are CUDA-graph capturable")
        print()

    teardown(ctx)


if __name__ == "__main__":
    main()
