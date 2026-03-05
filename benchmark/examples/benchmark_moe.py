#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark script for the expert-sharded MoE example.

This script follows the style and sweep intent of Triton's bench_mlp.py
for GPT-OSS-like sizes, adapted to the current Iris MoE example:
  - examples/31_expert_sharded_moe

It benchmarks distributed MoE forward:
  mixture_of_expt_epsharded(...)

Optional:
  - Validate against single-device reference (nosharded) output
  - Benchmark single-device reference latency on rank 0 for comparison

Run:
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark/examples/benchmark_moe.py \
      --num_ranks 8 --benchmark --output_file moe_gpt_oss.json
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import iris


def _load_example_modules():
    project_root = Path(__file__).resolve()
    while not (project_root / "tests").is_dir() or not (project_root / "examples").is_dir():
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root")
        project_root = project_root.parent

    example_dir = project_root / "examples" / "31_expert_sharded_moe"
    sys.path.insert(0, str(example_dir))

    from expert_assignment import make_expt_assignment, make_expt_dict_uniform
    from moe import MoeFusionConfig, mixture_of_expt_epsharded, mixture_of_expt_nosharded

    return (
        make_expt_assignment,
        make_expt_dict_uniform,
        MoeFusionConfig,
        mixture_of_expt_epsharded,
        mixture_of_expt_nosharded,
    )


(
    make_expt_assignment,
    make_expt_dict_uniform,
    MoeFusionConfig,
    mixture_of_expt_epsharded,
    mixture_of_expt_nosharded,
) = _load_example_modules()


def gpt_oss_batch_per_expert_sweep() -> list[int]:
    # Matches Triton bench_mlp.py:
    # batch_ranges = [(2**(2+k), 2**(3+k), min(2**k, 32)) for k in range(8)]
    # batch_sizes = list(chain(*[range(*r) for r in batch_ranges]))
    out: list[int] = []
    for k in range(8):
        start = 2 ** (2 + k)
        end = 2 ** (3 + k)
        step = min(2**k, 32)
        out.extend(list(range(start, end, step)))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark expert-sharded MoE with GPT-OSS-style sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks (GPUs)")
    parser.add_argument("--init_port", type=int, default=29531, help="TCP port for torch.distributed init")

    # GPT-OSS-like defaults from Triton bench_mlp.py end-goal.
    parser.add_argument("--d_model", type=int, default=5760, help="Model hidden dimension")
    parser.add_argument("--n_expts_tot", type=int, default=128, help="Total experts")
    parser.add_argument("--n_expts_act", type=int, default=4, help="Top-k experts per token")

    parser.add_argument(
        "--datatype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Activation/weight dtype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size in bytes")

    parser.add_argument(
        "--batch_per_expt",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit batch_per_expert values; default uses GPT-OSS sweep",
    )

    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark")
    parser.add_argument(
        "--validate", action="store_true", help="Validate distributed output vs single-device reference"
    )
    parser.add_argument(
        "--compare_single_gpu",
        action="store_true",
        help="Also benchmark single-device reference path on rank 0 for latency comparison",
    )

    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for do_bench")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations for do_bench")
    parser.add_argument("--breakdown", action="store_true", help="Print per-stage timing breakdown (rank 0)")

    parser.add_argument("--output_dir", type=str, default="benchmark/results/moe", help="Output directory")
    parser.add_argument("--output_file", type=str, default="benchmark_moe.json", help="Output JSON filename")
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="unfused",
        choices=[
            "unfused",
            "fused_grouped_matmul_convert_ep_to_dp",
            "fused_convert_dp_to_ep_grouped_matmul",
            "fused_convert_dp_to_ep_grouped_matmul__grouped_matmul_convert_ep_to_dp",
        ],
        help="MoE fusion mode selector",
    )
    return parser.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[s]


def _make_heap_resetter(allocator, offset):
    """Return a callable that resets the bump allocator to *offset*."""

    def _reset():
        allocator.heap_offset = offset

    return _reset


def _run_dist_once(
    x_dp_local,
    l_dp_local,
    w_ep_local,
    b_ep_local,
    expt_assignment,
    n_expts_act,
    shmem,
    fusion_config,
):
    return mixture_of_expt_epsharded(
        x_dp_local,
        l_dp_local,
        w_ep_local,
        b_ep_local,
        expt_assignment,
        n_expts_act,
        shmem,
        fusion_config=fusion_config,
    )


def _worker(rank: int, world_size: int, init_url: str, args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.cuda.set_device(rank)
    shmem = iris.iris(args.heap_size)

    try:
        ws = shmem.get_num_ranks()
        device = torch.device(f"cuda:{rank}")
        dtype = _dtype_from_str(args.datatype)
        fusion_config = MoeFusionConfig.from_mode_name(args.fusion_mode)

        if args.n_expts_tot % ws != 0:
            raise ValueError(f"n_expts_tot ({args.n_expts_tot}) must be divisible by world_size ({ws})")

        if args.batch_per_expt:
            sweep = args.batch_per_expt
        else:
            sweep = gpt_oss_batch_per_expert_sweep()

        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        results: list[dict] = []
        sweep_heap_base = shmem.heap.allocator.heap_offset

        for bpe in sweep:
            shmem.heap.allocator.heap_offset = sweep_heap_base
            n_tokens = bpe * args.n_expts_tot // args.n_expts_act
            if n_tokens % ws != 0:
                if rank == 0:
                    print(f"Skipping bpe={bpe}: n_tokens={n_tokens} not divisible by world_size={ws}")
                continue

            n_tokens_local = n_tokens // ws

            torch.manual_seed(0)
            x_global = torch.randn(n_tokens, args.d_model, device=device, dtype=dtype)
            l_global = torch.rand(n_tokens, args.n_expts_tot, device=device, dtype=torch.float32)
            w_global = torch.randn(args.n_expts_tot, args.d_model, args.d_model, device=device, dtype=dtype)
            b_global = torch.randn(args.n_expts_tot, args.d_model, device=device, dtype=torch.float32)

            dist.broadcast(x_global, src=0)
            dist.broadcast(l_global, src=0)
            dist.broadcast(w_global, src=0)
            dist.broadcast(b_global, src=0)

            expt_dict = make_expt_dict_uniform(ws, args.n_expts_tot)
            expt_assignment = make_expt_assignment(ws, args.n_expts_tot, expt_dict, device)

            first = rank * n_tokens_local
            last = first + n_tokens_local
            x_dp_local = x_global[first:last].contiguous()
            l_dp_local = l_global[first:last].contiguous()
            w_ep_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
            b_ep_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

            run_dist = functools.partial(
                _run_dist_once,
                x_dp_local,
                l_dp_local,
                w_ep_local,
                b_ep_local,
                expt_assignment,
                args.n_expts_act,
                shmem,
                fusion_config,
            )

            if args.validate or args.compare_single_gpu:
                y_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, args.n_expts_act)

            # Warmup one run for graph/kernels.
            z_dp_local = run_dist()
            y_tri = torch.empty((n_tokens, args.d_model), dtype=dtype, device=device)
            dist.all_gather_into_tensor(y_tri, z_dp_local.contiguous())

            if args.breakdown:
                N_BREAKDOWN_ITERS = 10
                stage_ms = {}
                for _ in range(N_BREAKDOWN_ITERS):
                    shmem.heap.allocator.heap_offset = sweep_heap_base
                    td = [] if rank == 0 else None
                    mixture_of_expt_epsharded(
                        x_dp_local,
                        l_dp_local,
                        w_ep_local,
                        b_ep_local,
                        expt_assignment,
                        args.n_expts_act,
                        shmem,
                        fusion_config=fusion_config,
                        timing_dict=td,
                    )
                    if rank == 0:
                        for j in range(1, len(td)):
                            key = td[j][0]
                            ms = td[j - 1][1].elapsed_time(td[j][1])
                            stage_ms.setdefault(key, []).append(ms)
                if rank == 0:
                    total_avg = sum(sum(v) / len(v) for v in stage_ms.values())
                    parts = []
                    for k, v in stage_ms.items():
                        avg = sum(v) / len(v)
                        pct = 100 * avg / total_avg if total_avg > 0 else 0
                        parts.append("{}={:.2f}ms ({:.1f}%)".format(k, avg, pct))
                    print("  [breakdown bpe={} total={:.2f}ms] ".format(bpe, total_avg) + "  ".join(parts))

            result = {
                "world_size": ws,
                "batch_per_expt": bpe,
                "n_tokens": n_tokens,
                "d_model": args.d_model,
                "n_expts_tot": args.n_expts_tot,
                "n_expts_act": args.n_expts_act,
                "dtype": args.datatype,
                "fusion_mode": fusion_config.mode_name(),
            }

            if args.validate:
                diff = (y_ref.float() - y_tri.float()).abs()
                result["validate_max_diff"] = float(diff.max().item())
                result["validate_mean_diff"] = float(diff.mean().item())
                result["validate_pass"] = bool(torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2))

            if args.benchmark:
                heap_snapshot = shmem.heap.allocator.heap_offset
                reset_heap = _make_heap_resetter(shmem.heap.allocator, heap_snapshot)
                saved_refresh = shmem.heap.refresh_peer_access
                shmem.heap.refresh_peer_access = lambda: None
                dist_ms = iris.do_bench(
                    run_dist,
                    barrier_fn=shmem.barrier,
                    preamble_fn=reset_heap,
                    n_warmup=args.warmup,
                    n_repeat=args.repeat,
                    return_mode="mean",
                )
                shmem.heap.refresh_peer_access = saved_refresh
                reset_heap()
                result["dist_ms"] = float(dist_ms)

            if args.compare_single_gpu:
                if rank == 0:

                    def run_ref():
                        return mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, args.n_expts_act)

                    ref_ms = iris.do_bench(
                        run_ref,
                        barrier_fn=torch.cuda.synchronize,
                        n_warmup=args.warmup,
                        n_repeat=args.repeat,
                        return_mode="mean",
                    )
                    result["single_gpu_ref_ms"] = float(ref_ms)
                    if args.benchmark and ref_ms > 0:
                        result["speedup_vs_single_gpu"] = float(ref_ms / dist_ms)

                # keep all ranks aligned before next config
                shmem.barrier()

            if rank == 0:
                print(
                    f"[bpe={bpe:4d}] n_tokens={n_tokens:6d}"
                    + (f" dist={result.get('dist_ms', 0.0):8.3f} ms" if args.benchmark else "")
                    + (
                        f" ref={result.get('single_gpu_ref_ms', 0.0):8.3f} ms"
                        if args.compare_single_gpu and "single_gpu_ref_ms" in result
                        else ""
                    )
                    + (f" max_diff={result.get('validate_max_diff', 0.0):.4f}" if args.validate else "")
                )
                results.append(result)

            shmem.barrier()

        if rank == 0:
            out_path = Path(args.output_dir) / args.output_file
            payload = {
                "sweep": "gpt_oss_batch_per_expt" if not args.batch_per_expt else "custom",
                "results": results,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved benchmark results: {out_path}")

    finally:
        try:
            shmem.barrier()
        except Exception:
            pass
        del shmem
        import gc

        gc.collect()
        dist.destroy_process_group()


def main():
    args = parse_args()
    if not args.benchmark and not args.validate and not args.compare_single_gpu:
        print("No mode selected. Use at least one of: --benchmark, --validate, --compare_single_gpu")
        sys.exit(1)

    init_url = f"tcp://127.0.0.1:{args.init_port}"
    mp.spawn(
        fn=_worker,
        args=(args.num_ranks, init_url, args),
        nprocs=args.num_ranks,
        join=True,
    )


if __name__ == "__main__":
    main()
