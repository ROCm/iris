# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Minimal CUDA graph capture test for iris gluon all-reduce.

Isolates graph capture / replay without vLLM or aiter.
Run with: torchrun --nproc_per_node=N python tests/test_graph_capture_allreduce.py

Part A — single capture:
  1. Eager correctness (baseline)
  2. Graph capture succeeds (no non-capturable ops)
  3. Single replay correctness
  4. Double replay correctness (2nd decode step crash repro)
  5. Replay with new input data (pointer table validity)

Part B — piecewise capture (vLLM pattern):
  6. Three separate graphs with different tensor sizes, shared workspace
  7. Interleaved replay of all three graphs
  8. Catches data_ptr reuse bugs across captures
"""

import os
import sys
import torch
import torch.distributed as dist


def check(name, actual, expected_val, shape, rank):
    expected = torch.full(shape, expected_val, device="cuda", dtype=torch.float32)
    if torch.allclose(actual.float(), expected, rtol=1e-2, atol=1e-2):
        if rank == 0:
            print(f"PASS: {name}")
        return True
    else:
        print(f"FAIL: {name} rank={rank} got={actual.view(-1)[0].item():.4f} expected={expected_val:.4f}")
        return False


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    import iris
    from iris.ccl.config import Config

    ctx = iris.iris(heap_size=2 ** 30)
    cfg = Config(use_gluon=True)

    dtype = torch.bfloat16
    passed = 0
    total = 0

    # =========================================
    # Part A: single capture, multiple replays
    # =========================================
    shape = (2, 8192)

    # Test 1: eager correctness
    total += 1
    inp = ctx.empty(shape, dtype=dtype)
    inp.fill_(rank + 1.0)
    out = ctx.empty(shape, dtype=dtype)

    ws = ctx.ccl.all_reduce(out, inp, config=cfg)
    torch.cuda.synchronize()

    expected = sum(r + 1.0 for r in range(world_size))
    if check("eager correctness", out, expected, shape, rank):
        passed += 1

    # Test 2-5: graph capture + replay
    graph_out = ctx.empty(shape, dtype=dtype)

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    dist.barrier()

    # warmup in capture stream
    with torch.cuda.stream(stream):
        ws = ctx.ccl.all_reduce(graph_out, inp, config=cfg, workspace=ws)
    torch.cuda.synchronize()
    dist.barrier()

    # capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            ws = ctx.ccl.all_reduce(graph_out, inp, config=cfg, workspace=ws)

    total += 1
    if rank == 0:
        print("PASS: graph capture succeeded")
    passed += 1

    # Test 3: single replay
    total += 1
    inp.fill_(rank + 1.0)
    graph.replay()
    torch.cuda.synchronize()
    if check("single replay", graph_out, expected, shape, rank):
        passed += 1

    # Test 4: double replay
    total += 1
    inp.fill_(rank + 1.0)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    if check("double replay", graph_out, expected, shape, rank):
        passed += 1

    # Test 5: replay with new data
    total += 1
    inp.fill_((rank + 1.0) * 2)
    graph.replay()
    torch.cuda.synchronize()
    expected2 = sum((r + 1.0) * 2 for r in range(world_size))
    if check("replay new data", graph_out, expected2, shape, rank):
        passed += 1

    # =========================================
    # Part B: piecewise capture (vLLM pattern)
    # 3 graphs with different sizes, shared workspace
    # =========================================
    if rank == 0:
        print("\n--- Part B: piecewise capture ---")

    shapes = [(1, 8192), (4, 8192), (2, 8192)]
    graphs = []
    inputs = []
    outputs = []
    ws_piece = None

    for i, s in enumerate(shapes):
        inp_i = ctx.empty(s, dtype=dtype)
        out_i = ctx.empty(s, dtype=dtype)
        inp_i.fill_(rank + 1.0)
        inputs.append(inp_i)
        outputs.append(out_i)

        # warmup
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            ws_piece = ctx.ccl.all_reduce(out_i, inp_i, config=cfg, workspace=ws_piece)
        torch.cuda.synchronize()
        dist.barrier()

    for i, s in enumerate(shapes):
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            with torch.cuda.graph(g, stream=st):
                ws_piece = ctx.ccl.all_reduce(outputs[i], inputs[i], config=cfg, workspace=ws_piece)
        graphs.append(g)

    total += 1
    if rank == 0:
        print(f"PASS: piecewise capture ({len(shapes)} graphs)")
    passed += 1

    # Test 7: replay each graph
    for i, (g, s) in enumerate(zip(graphs, shapes)):
        total += 1
        inputs[i].fill_(rank + 1.0)
        g.replay()
        torch.cuda.synchronize()
        if check(f"piecewise replay graph[{i}] shape={s}", outputs[i], expected, s, rank):
            passed += 1

    # Test 8: interleaved replay (catches cross-capture corruption)
    total += 1
    for inp_i in inputs:
        inp_i.fill_(rank + 1.0)
    graphs[2].replay()
    graphs[0].replay()
    graphs[1].replay()
    torch.cuda.synchronize()
    all_ok = all(
        torch.allclose(outputs[i].float(), torch.full(shapes[i], expected, device="cuda"), rtol=1e-2, atol=1e-2)
        for i in range(len(shapes))
    )
    if all_ok:
        if rank == 0:
            print("PASS: interleaved replay correctness")
        passed += 1
    else:
        for i in range(len(shapes)):
            if not torch.allclose(outputs[i].float(), torch.full(shapes[i], expected, device="cuda"), rtol=1e-2, atol=1e-2):
                print(f"FAIL: interleaved replay graph[{i}] rank={rank} got={outputs[i].view(-1)[0].item():.4f}")

    # Summary
    if rank == 0:
        print(f"\n{passed}/{total} tests passed")
        if passed == total:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
    sys.exit(0 if passed == total else 1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
