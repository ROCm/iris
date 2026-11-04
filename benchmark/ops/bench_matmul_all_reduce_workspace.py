# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Measure the public workspace-reuse path with one harness for base and head.

Run with torchrun. Select the implementation with --source-dir; this script must
always come from the same revision for both measurements. See README_workspace.md.
"""

import argparse
from datetime import timedelta
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--label", choices=("base", "head"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=512, help="Local K on each rank, for every implementation.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if min(*args.sizes, args.n, args.k, args.warmup, args.iterations, args.trials) < 1:
        parser.error("Dimensions and iteration counts must be positive.")
    if min(*args.sizes, args.n, args.k) < 64:
        parser.error("Dimensions must be at least 64 for the fixed 64x64x32 configuration.")
    return args


def main():
    args = parse_args()
    source = args.source_dir.resolve(strict=True)
    sys.path.insert(0, str(source))

    import torch
    import torch.distributed as dist
    import triton
    import iris
    from iris.ops import FusedConfig, matmul_all_reduce, matmul_all_reduce_preamble

    Path(iris.__file__).resolve().relative_to(source)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ctx = iris.iris(1 << 29)
    results = []

    def measure(run):
        for _ in range(args.warmup):
            run()
            ctx.barrier()
        trials = []
        for _ in range(args.trials):
            wall_ms, stream_ms = [], []
            # Create events before timing; exclude compilation and allocation.
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
            for start, end in zip(starts, ends):
                ctx.barrier()
                start.record()
                begin = time.perf_counter()
                run()
                end.record()
                # Include completion on all ranks in end-to-end wall latency.
                ctx.barrier()
                wall_ms.append((time.perf_counter() - begin) * 1000)
                stream_ms.append(start.elapsed_time(end))
            trials.append({"wall_ms": wall_ms, "stream_ms": stream_ms})
        gathered = [None] * world_size
        dist.all_gather_object(gathered, trials)
        return gathered

    def benchmark_case(m, implementation):
        torch.manual_seed(1234 + rank)
        a = torch.randn((m, args.k), device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16) * 0.1
        c = ctx.zeros((m, args.n), dtype=torch.float16)
        reference = torch.mm(a, b)
        dist.all_reduce(reference)
        workspace = None
        if implementation != "torch_mm_rccl":
            config = FusedConfig(block_size_m=64, block_size_n=64, block_size_k=32, all_reduce_variant=implementation)
            workspace = matmul_all_reduce_preamble(ctx, c, a, b, config=config)

        def run():
            nonlocal workspace
            if implementation == "torch_mm_rccl":
                torch.mm(a, b, out=c)
                dist.all_reduce(c)
            else:
                workspace = matmul_all_reduce(ctx, c, a, b, config=config, workspace=workspace, async_op=True)

        run()
        ctx.barrier()
        torch.testing.assert_close(c, reference, atol=0.2, rtol=0.02)
        samples = measure(run)
        torch.testing.assert_close(c, reference, atol=0.2, rtol=0.02)
        # For each trial, report the slowest rank's median sample, then the
        # median across trials. Retain every rank/sample for independent analysis.
        row = {"m": m, "n": args.n, "k_local": args.k, "implementation": implementation, "samples": samples}
        for metric in ("wall_ms", "stream_ms"):
            trial_medians = [
                max(statistics.median(samples[r][t][metric]) for r in range(world_size)) for t in range(args.trials)
            ]
            row[metric] = statistics.median(trial_medians)
        row["tflops_per_rank"] = 2 * m * args.n * args.k / (row["wall_ms"] * 1e9)
        row["validated"] = True
        if rank == 0:
            print(f"{args.label}: {implementation} M={m} ranks={world_size} wall={row['wall_ms']:.4f} ms", flush=True)
        ctx.barrier()
        return row

    try:
        for m in args.sizes:
            for implementation in ("atomic", "one_shot", "two_shot", "torch_mm_rccl"):
                results.append(benchmark_case(m, implementation))
                gc.collect()
                ctx.barrier()
        if rank == 0:
            report = {
                "schema_version": 1,
                "label": args.label,
                "commit": args.commit,
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "triton": triton.__version__,
                "tritonblas": importlib.metadata.version("tritonblas"),
                "dtype": "float16",
                "config": {"block_size_m": 64, "block_size_n": 64, "block_size_k": 32},
                "warmup": args.warmup,
                "iterations": args.iterations,
                "trials": args.trials,
                "results": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        ctx.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
