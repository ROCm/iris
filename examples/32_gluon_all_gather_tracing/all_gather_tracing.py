#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon All-Gather Tracing Example
=================================

Demonstrates IrisDeviceCtx tracing support inside a ``@gluon.jit`` kernel.
The kernel performs a one-hop ring put (all-gather step) and can be compiled
in two modes via a constexpr flag:

- ``TRACING=False``  (default) — zero overhead; the entire tracing path is
  dead-code-eliminated at compile time because ``enabled`` is a ``tl.constexpr``.
- ``TRACING=True``  — ``record_event_start`` / ``record_event_end`` bracket
  every remote put; the trace is exported to per-rank JSON files.

Usage::

    # Without tracing (default)
    torchrun --nproc_per_node=4 \\
        examples/32_gluon_all_gather_tracing/all_gather_tracing.py

    # With tracing enabled and JSON export
    torchrun --nproc_per_node=4 \\
        examples/32_gluon_all_gather_tracing/all_gather_tracing.py --trace --export

Zero-overhead proof
-------------------
When ``TRACING=False``, no assembly is generated for the tracing code path.
You can verify this by comparing the cached AMDGCN ISA for the two variants
after running (see ``--asm_diff`` flag).
"""

import argparse
import json
import os
import sys

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import iris.experimental.iris_gluon as iris_gl
from iris.tracing.events import TraceEvent


# ---------------------------------------------------------------------------
# Device kernel
# ---------------------------------------------------------------------------


@gluon.jit
def all_gather_put_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    local_buf,
    global_buf,
    num_elements: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    TRACING: gl.constexpr,
):
    """
    One-hop ring all-gather put kernel.

    Each CTA handles one tile of ``BLOCK_SIZE`` elements from this rank's
    ``local_buf`` and pushes it into the next rank's ``global_buf`` slice
    at the offset reserved for this rank.

    When ``TRACING=True``, the put is bracketed by tracing calls.
    When ``TRACING=False``, the tracing calls compile away completely
    because ``GluonDeviceTracing.enabled`` is a ``tl.constexpr``.

    Args:
        IrisDeviceCtx: aggregate class passed as constexpr from the host.
        context_tensor: encoded context tensor (from ``shmem.get_device_context()``).
        local_buf: source buffer (``num_elements`` elements on this rank).
        global_buf: output buffer (``num_ranks * num_elements`` elements).
        num_elements: per-rank element count (constexpr).
        BLOCK_SIZE: tile width in elements (constexpr).
        NUM_WARPS: number of warps per CTA (constexpr).
        TRACING: enable/disable tracing at compile time (constexpr).
    """
    ctx = IrisDeviceCtx.initialize(context_tensor, tracing=TRACING)

    cur_rank = ctx.cur_rank
    num_ranks = ctx.num_ranks
    target_rank = (cur_rank + 1) % num_ranks

    pid = gl.program_id(0)

    # AMD GPUs have 64 threads per warp (wavefront size 64).
    # Total threads per CTA = NUM_WARPS * 64.
    # Each thread handles SPT = BLOCK_SIZE // (NUM_WARPS * 64) elements.
    SPT: gl.constexpr = BLOCK_SIZE // (NUM_WARPS * 64)
    layout: gl.constexpr = gl.BlockedLayout([SPT], [64], [NUM_WARPS], [0])
    offsets = pid * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < num_elements

    # Address in the target rank's global buffer for *this* rank's slice
    target_offset = cur_rank * num_elements + offsets
    target_ptr = global_buf + target_offset

    # Optional tracing — compiles away when TRACING=False
    handle = ctx.tracing.record_event_start(
        event_id=TraceEvent().put,
        target_rank=target_rank,
        address=target_ptr,
        pid_m=gl.program_id(0),
        pid_n=tl.cast(0, tl.int32),
        mask=mask,
    )

    # Remote put: push local slice to the target rank
    ctx.put(local_buf + offsets, target_ptr, to_rank=target_rank, mask=mask)

    ctx.tracing.record_event_end(handle)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _launch(shmem, local_buf, global_buf, context_tensor, enable_tracing: bool):
    """Launch one iteration of the all-gather kernel.

    Layout constraints for AMD GPUs (warp size = 64):
      BLOCK_SIZE = sizePerThread * 64 * NUM_WARPS
    Here NUM_WARPS=4, sizePerThread=1 → BLOCK_SIZE = 1 * 64 * 4 = 256.
    """
    num_elements = local_buf.numel()
    NUM_WARPS = 4
    BLOCK_SIZE = 64 * NUM_WARPS  # 256 elements per tile (1 el/thread, 64 threads/warp, 4 warps)
    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    all_gather_put_kernel[grid](
        iris_gl.IrisDeviceCtx,
        context_tensor,
        local_buf,
        global_buf,
        num_elements,
        BLOCK_SIZE,
        NUM_WARPS,
        enable_tracing,
        num_warps=NUM_WARPS,
    )


def run_all_gather(shmem, local_buf, global_buf, context_tensor, enable_tracing: bool, num_warmup: int = 5):
    """Warm up then run one timed iteration; return elapsed ms."""
    # Warm-up always without tracing to avoid polluting trace buffers
    for _ in range(num_warmup):
        _launch(shmem, local_buf, global_buf, context_tensor, enable_tracing=False)
        shmem.barrier()

    if enable_tracing:
        shmem.tracing.reset()
        shmem.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    shmem.barrier()
    start_event.record()
    _launch(shmem, local_buf, global_buf, context_tensor, enable_tracing=enable_tracing)
    end_event.record()
    torch.cuda.synchronize()
    shmem.barrier()

    return start_event.elapsed_time(end_event)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Gluon all-gather tracing example")
    parser.add_argument("--trace", action="store_true", help="Enable device-side event tracing")
    parser.add_argument("--export", action="store_true", help="Export trace to JSON after run")
    parser.add_argument("--max_events", type=int, default=1_000_000, help="Max trace events per rank")
    parser.add_argument("--num_elements", type=int, default=65536, help="Elements per rank")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size in bytes")
    args = parser.parse_args()

    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE; initialize distributed process group
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    shmem = iris_gl.iris(args.heap_size)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()

    if rank == 0:
        print("Gluon All-Gather Tracing Example")
        print(f"  ranks        : {num_ranks}")
        print(f"  num_elements : {args.num_elements} per rank")
        print(f"  tracing      : {args.trace}")
        print()

    # Allocate symmetric buffers
    local_buf = shmem.zeros((args.num_elements,), dtype=torch.float32)
    local_buf.fill_(float(rank))  # rank r fills with value r
    global_buf = shmem.zeros((num_ranks * args.num_elements,), dtype=torch.float32)
    shmem.barrier()

    # Enable host-side tracing before building the context tensor
    if args.trace:
        shmem.tracing.enable(max_events=args.max_events)

    context_tensor = shmem.get_device_context()

    # --- Run WITHOUT tracing (baseline) ---
    ms_no_trace = run_all_gather(shmem, local_buf, global_buf, context_tensor, enable_tracing=False)
    if rank == 0:
        print(f"[tracing=False] {ms_no_trace:.3f} ms  ← zero-overhead path")

    # --- Run WITH tracing (only when enabled on host) ---
    if args.trace:
        ms_trace = run_all_gather(shmem, local_buf, global_buf, context_tensor, enable_tracing=True)
        if rank == 0:
            print(f"[tracing=True ] {ms_trace:.3f} ms  ← tracing path")

    # --- Validate correctness ---
    shmem.barrier()
    torch.cuda.synchronize()
    errors = 0
    # We only check the slice written to *this* rank by the rank behind us
    src = (rank - 1) % num_ranks
    slice_start = src * args.num_elements
    slice_end = slice_start + args.num_elements
    actual = global_buf[slice_start:slice_end]
    wrong = (actual != float(src)).sum().item()
    if wrong > 0:
        print(f"  [rank {rank}] MISMATCH: {wrong} of {args.num_elements} elements wrong for src={src}", file=sys.stderr)
        errors += 1

    # Aggregate errors across all ranks before reporting
    error_tensor = torch.tensor(errors, device=f"cuda:{local_rank}", dtype=torch.int32)
    dist.all_reduce(error_tensor, op=dist.ReduceOp.SUM)
    total_errors = error_tensor.item()

    shmem.barrier()
    if rank == 0:
        print(f"\nValidation: {'PASSED' if total_errors == 0 else 'FAILED'}")

    # --- Export trace ---
    if args.trace and args.export:
        out_file = "gluon_trace.json"
        shmem.tracing.export(out_file)
        trace_count = shmem.tracing.trace_counter.item()
        if rank == 0:
            print(f"\nTrace summary (rank {rank}):")
            print(f"  events recorded : {trace_count}")
            per_rank_file = out_file.replace(".json", f"_rank{rank}.json")
            if os.path.exists(per_rank_file):
                with open(per_rank_file) as f:
                    data = json.load(f)
                trace_events = [e for e in data["traceEvents"] if e.get("ph") != "M"]
                print(f"  events in JSON  : {len(trace_events)}")
                if trace_events:
                    ev = trace_events[0]
                    print(f"  first event     : name={ev['name']}, ts={ev['ts']}, dur={ev.get('dur', 'N/A')}")
                print(f"  exported to     : {per_rank_file}")
                print("  View at         : https://ui.perfetto.dev")

    shmem.barrier()
    del shmem


if __name__ == "__main__":
    main()
