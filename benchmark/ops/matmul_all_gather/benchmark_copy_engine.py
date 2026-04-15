#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris.ops matmul_all_gather_copy_engine fused operation.

This benchmark showcases the copy engine (SDMA) variant of fused GEMM + All-Gather
where each rank computes a local matmul and then uses SDMA hardware to scatter
results along M dimension.
"""

import os
import torch
import torch.distributed as dist
import random
import argparse
import numpy as np

from examples.common.utils import JSONWriter

import iris

from iris.ops.matmul_all_gather_copy_engine import (
    _device_quiet_kernel,
    matmul_all_gather_copy_engine,
    matmul_all_gather_copy_engine_preamble,
)
from iris.ops import FusedConfig

_DERIVE_AVAILABLE = False
try:
    import sys as _sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)
    from derive_params import (
        derive as _derive_params,
        _tile_roofline,
        _gemm_wg_time_us,
        _scatter_sdma_time_us,
        DEFAULT_NUM_CUS,
        DEFAULT_PEAK_TFLOPS_FP16,
        DEFAULT_HBM_BW_GBPS,
        DEFAULT_L2_SIZE_BYTES,
        DEFAULT_SCHEDULING_FACTOR,
    )

    _DERIVE_AVAILABLE = True
except Exception:
    pass

torch.manual_seed(123)
random.seed(123)

TICKS_PER_US = 100  # s_memrealtime runs at 100 MHz: 1 tick = 10 ns = 0.01 us


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark matmul_all_gather_copy_engine fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M_local)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of tensors",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="matmul_all_gather_copy_engine.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--num_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument(
        "--benchmark_baseline",
        action="store_true",
        help="Also benchmark baseline (non-copy-engine) variant for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size for M dimension (auto if None)")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size for N dimension (auto if None)")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size for K dimension (auto if None)")
    parser.add_argument(
        "--group_size_m", type=int, default=None, help="Group size for M dimension tiling (auto if None)"
    )
    parser.add_argument(
        "--m_tiles_per_batch",
        type=int,
        default=None,
        help="Number of M tiles grouped behind one readiness flag (auto if None)",
    )
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")

    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype, device, dtype_bytes=2):
    """Fill None-valued kernel parameters using tritonBLAS selector.

    Priority:
    1. User-specified values (keep as-is)
    2. tritonBLAS selector (always)

    Returns a list of parameter names that were set by the selector.
    """
    applied = []

    # Import tritonBLAS selector
    from tritonblas.matmul import _make_matmul_selector

    # Always use tritonBLAS selector for block sizes
    selector = _make_matmul_selector(
        args["m"],  # M_local
        args["n"],
        args["k"],
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )

    # Only apply if not user-specified
    if args.get("block_size_m") is None:
        args["block_size_m"] = selector.block_m
        applied.append("block_size_m (tritonBLAS)")
    if args.get("block_size_n") is None:
        args["block_size_n"] = selector.block_n
        applied.append("block_size_n (tritonBLAS)")
    if args.get("block_size_k") is None:
        args["block_size_k"] = selector.block_k
        applied.append("block_size_k (tritonBLAS)")
    if args.get("group_size_m") is None:
        args["group_size_m"] = selector.group_m
        applied.append("group_size_m (tritonBLAS)")

    return selector, applied


def _analyze_performance(bm, bn, bk, M_local, N, K, world_size, link_bw, dtype_bytes):
    """Analyze GEMM vs scatter bottleneck using performance model from derive_params.

    Returns dict with performance metrics (to be written to JSON):
        - gemm_wg_us: Per-workgroup GEMM time
        - scatter_wg_us: Per-workgroup scatter time
        - bottleneck: "compute" or "communication"
        - ratio: scatter_wg_us / gemm_wg_us
        - roofline_tflops: Achievable TFLOPS from roofline model
        - intensity: Arithmetic intensity (FLOPs/byte)
    """
    if not _DERIVE_AVAILABLE:
        return None

    try:
        # Roofline model for GEMM
        roofline_tflops, intensity, ridge, b_in_l2 = _tile_roofline(
            bm,
            bn,
            bk,
            M_local,
            K,
            N,
            dtype_bytes,
            DEFAULT_PEAK_TFLOPS_FP16,
            DEFAULT_HBM_BW_GBPS,
            DEFAULT_L2_SIZE_BYTES,
        )

        # Per-WG GEMM time
        gemm_wg_us = _gemm_wg_time_us(
            bm,
            bn,
            bk,
            K,
            roofline_tflops,
            DEFAULT_NUM_CUS,
        )

        # Per-WG scatter time
        scatter_wg_us = _scatter_sdma_time_us(bm, bn, world_size, link_bw, dtype_bytes)

        # Determine bottleneck
        ratio = scatter_wg_us / gemm_wg_us
        bottleneck = "communication" if ratio > 1.0 else "compute"

        return {
            "gemm_wg_us": gemm_wg_us,
            "scatter_wg_us": scatter_wg_us,
            "bottleneck": bottleneck,
            "ratio": ratio,
            "roofline_tflops": roofline_tflops,
            "intensity": intensity,
        }
    except Exception as e:
        print(f"Warning: Performance analysis failed ({e})")
        return None


def _auto_tune_batching(perf_analysis, num_m_tiles, num_n_tiles, group_size_m, bm, bn, dtype_bytes):
    """Auto-tune m_tiles_per_batch based on bottleneck analysis.

    Baseline heuristic: choose one wave-aligned batch worth of M-tiles.

    Returns: (m_tiles_per_batch, adjusted_block_m, adjusted_block_n)
    """
    if perf_analysis is None:
        return 1, bm, bn  # Conservative default

    num_cus = DEFAULT_NUM_CUS

    # Batch one wave's worth of M-tiles when possible.
    tiles_per_group = group_size_m * num_n_tiles
    groups_per_wave = max(1, num_cus // tiles_per_group)
    m_tiles_per_batch = max(1, groups_per_wave * group_size_m)

    if perf_analysis["bottleneck"] == "compute":
        # Compute-bound: keep original block sizes
        # Larger tiles = more compute per tile = better amortization
        adjusted_bm, adjusted_bn = bm, bn
    else:
        # Communication-bound: reduce tile sizes to increase overlap
        ratio = perf_analysis["ratio"]

        # TODO need a better way to determine if we can halve
        # TODO this breaks the selector
        # if ratio > 1.0:  # Severely communication-bound
        #     print(f"Communication-bound (ratio={ratio:.2f}x), halving tile sizes for better overlap")
        #     # Halve block sizes to create MORE tiles (but enforce minimum)
        #     MIN_BLOCK_SIZE = 64
        #     adjusted_bm = MIN_BLOCK_SIZE #max(MIN_BLOCK_SIZE, bm // 4)
        #     adjusted_bn = MIN_BLOCK_SIZE #max(MIN_BLOCK_SIZE, bn // 4)
        # else:
        adjusted_bm, adjusted_bn = bm, bn

    # Clamp to valid range
    m_tiles_per_batch = max(1, min(m_tiles_per_batch, num_m_tiles))

    return m_tiles_per_batch, adjusted_bm, adjusted_bn


def _derive_batch_metadata(selector, m_local, n, m_tiles_per_batch):
    """Return selector- and batching-derived metadata for JSON/reporting."""
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    num_stages = getattr(selector, "num_stages", 2)
    waves_per_eu = getattr(selector, "waves_per_eu", 0)
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None:
        active_cus = getattr(selector._hardware, "N_CU", getattr(selector._hardware, "NUM_XCD", 1))

    num_tiles_m = (m_local + block_size_m - 1) // block_size_m
    num_tiles_n = (n + block_size_n - 1) // block_size_n
    tiles_per_group = max(1, group_size_m * num_tiles_n)
    groups_per_wave = max(1, active_cus // tiles_per_group)
    m_tiles_per_wave = min(num_tiles_m, groups_per_wave * group_size_m)
    num_batches = (num_tiles_m + m_tiles_per_batch - 1) // m_tiles_per_batch
    last_batch_m_tiles = num_tiles_m - max(0, num_batches - 1) * m_tiles_per_batch

    return {
        "output_tile_size_m": block_size_m,
        "output_tile_size_n": block_size_n,
        "output_tile_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "active_cus": active_cus,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "tiles_per_group": tiles_per_group,
        "groups_per_wave": groups_per_wave,
        "m_tiles_per_wave": m_tiles_per_wave,
        "num_batches": num_batches,
        "last_batch_m_tiles": last_batch_m_tiles,
        "m_tiles_per_batch_over_wave": m_tiles_per_batch / max(1, m_tiles_per_wave),
    }


def _worker(args: dict):
    """Worker function for PyTorch distributed execution."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    shmem = iris.iris(args["heap_size"])
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    datatype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    datatype = datatype_map.get(args["datatype"], torch.float16)
    dtype_bytes = torch.tensor([], dtype=datatype).element_size()

    # Get device for tritonBLAS selector
    device = torch.device(f"cuda:{local_rank}")

    # Apply tritonBLAS selector to get block sizes
    selector, model_applied = _apply_model_defaults(args, world_size, datatype, device, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"tritonBLAS selector applied: {', '.join(model_applied)}")

    M_local = args["m"]  # Local M dimension
    M = M_local * world_size  # Total M after gather
    N = args["n"]
    K = args["k"]

    # if args.get("m_tiles_per_batch") is None:
    #     args["m_tiles_per_batch"] = 1

    # Performance analysis
    perf_analysis = _analyze_performance(
        args["block_size_m"],
        args["block_size_n"],
        args["block_size_k"],
        M_local,
        N,
        K,
        world_size,
        args.get("link_bw", 50.0),
        dtype_bytes,
    )

    # Auto-tune m_tiles_per_batch if not user-specified
    if args.get("m_tiles_per_batch") is None:
        num_m_tiles = (M_local + args["block_size_m"] - 1) // args["block_size_m"]
        num_n_tiles = (N + args["block_size_n"] - 1) // args["block_size_n"]

        m_tiles_per_batch, adj_bm, adj_bn = _auto_tune_batching(
            perf_analysis,
            num_m_tiles,
            num_n_tiles,
            args["group_size_m"],
            args["block_size_m"],
            args["block_size_n"],
            dtype_bytes,
        )

        # Apply tuned values
        args["m_tiles_per_batch"] = m_tiles_per_batch

        # If block sizes were adjusted, update config
        if adj_bm != args["block_size_m"] or adj_bn != args["block_size_n"]:
            if rank == 0:
                shmem.info(f"Adjusted block sizes for communication-bound: {adj_bm}×{adj_bn}")
            args["block_size_m"] = adj_bm
            args["block_size_n"] = adj_bn

        if rank == 0:
            shmem.info(f"Auto-tuned m_tiles_per_batch: {m_tiles_per_batch}")

    # Print performance analysis
    if rank == 0 and perf_analysis is not None:
        shmem.info("\n" + "=" * 80)
        shmem.info("PERFORMANCE ANALYSIS")
        shmem.info("=" * 80)
        shmem.info(f"Block sizes: {args['block_size_m']}×{args['block_size_n']}×{args['block_size_k']}")
        num_m_tiles = (M_local + args["block_size_m"] - 1) // args["block_size_m"]
        num_n_tiles = (N + args["block_size_n"] - 1) // args["block_size_n"]
        shmem.info(f"Tiles: {num_m_tiles} M-tiles × {num_n_tiles} N-tiles = {num_m_tiles * num_n_tiles} total")
        shmem.info(f"\nPer-tile timing:")
        shmem.info(f"  GEMM:    {perf_analysis['gemm_wg_us']:.2f} μs")
        shmem.info(f"  Scatter: {perf_analysis['scatter_wg_us']:.2f} μs")
        shmem.info(f"  Ratio:   {perf_analysis['ratio']:.2f}x")
        shmem.info(f"\nBottleneck: {perf_analysis['bottleneck'].upper()}")
        shmem.info(f"m_tiles_per_batch: {args['m_tiles_per_batch']}")
        shmem.info("=" * 80 + "\n")

    batch_metadata = _derive_batch_metadata(selector, M_local, N, args["m_tiles_per_batch"])

    # Create config with parameters
    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["num_sms"] is not None:
        config_kwargs["num_sms"] = args["num_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]
    config = FusedConfig(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("operation", "matmul_all_gather_copy_engine")
    json_writer.add_field("m_local", M_local)
    json_writer.add_field("m_total", M)

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Write performance analysis to JSON
    if perf_analysis is not None:
        for key, value in perf_analysis.items():
            json_writer.add_field(f"perf_analysis_{key}", value)

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)
    json_writer.add_field("m_tiles_per_batch", args["m_tiles_per_batch"])
    for key, value in batch_metadata.items():
        json_writer.add_field(key, value)

    # Create input and output tensors
    # A_local is M_local x K, output is M x N (gathered)
    A_local = shmem.zeros((M_local, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # Expected
    expected_tensor = None
    if args["validate"]:
        # Gather all A_local matrices and compute expected result
        A_local_list = [torch.zeros((M_local, K), dtype=datatype, device=f"cuda:{rank}") for _ in range(world_size)]
        dist.all_gather(A_local_list, A_local_data)

        # Expected: [A_0 @ B; A_1 @ B; ...; A_n @ B] stacked along M
        expected_tensor = shmem.zeros((M, N), dtype=datatype)
        expected_parts = []
        for i, A_rank_local in enumerate(A_local_list):
            C_rank_local = torch.matmul(A_rank_local, B_data)
            expected_parts.append(C_rank_local)
        expected_result = torch.cat(expected_parts, dim=0)
        expected_tensor.copy_(expected_result)

    # Pre-allocate workspace
    workspace = matmul_all_gather_copy_engine_preamble(
        shmem,
        A_local,
        B,
        config,
        m_tiles_per_batch=args["m_tiles_per_batch"],
    )
    workspace.selector = selector

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "copy_engine": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
        "baseline": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }
    flag_iteration = 0

    def run_experiment():
        nonlocal kernel_timing, workspace, flag_iteration

        shmem.barrier()

        torch.cuda.nvtx.range_push("Matmul-All-Gather-CopyEngine")
        with torch.cuda.stream(comm_stream):
            kernel_timing["copy_engine"]["start_event"].record()
            workspace = matmul_all_gather_copy_engine(
                shmem,
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                use_copy_engine=True,
                flag_iteration=flag_iteration,
                m_tiles_per_batch=args["m_tiles_per_batch"],
            )
            kernel_timing["copy_engine"]["end_event"].record()
            kernel_timing["copy_engine"]["experiments"] += 1
            flag_iteration += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier(comm_stream)

        # Update timing
        ms = kernel_timing["copy_engine"]["start_event"].elapsed_time(kernel_timing["copy_engine"]["end_event"])
        kernel_timing["copy_engine"]["ms"] += ms

    def run_baseline_experiment():
        nonlocal kernel_timing, workspace

        shmem.barrier()

        torch.cuda.nvtx.range_push("Matmul-All-Gather-Baseline")
        with torch.cuda.stream(comm_stream):
            kernel_timing["baseline"]["start_event"].record()
            workspace = matmul_all_gather_copy_engine(
                shmem,
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                use_copy_engine=False,
                m_tiles_per_batch=args["m_tiles_per_batch"],
            )
            kernel_timing["baseline"]["end_event"].record()
            kernel_timing["baseline"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier(comm_stream)

        # Update timing
        ms = kernel_timing["baseline"]["start_event"].elapsed_time(kernel_timing["baseline"]["end_event"])
        kernel_timing["baseline"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating copy engine variant...")

        # Reset output before validation
        C.zero_()
        shmem.barrier()

        run_experiment()
        torch.cuda.synchronize()
        shmem.barrier()

        atol = 1e-1 if datatype == torch.float16 else 1e-3
        rtol = 1e-2 if datatype == torch.float16 else 1e-5
        success = torch.allclose(C, expected_tensor, atol=atol, rtol=rtol)
        if not success:
            max_diff = torch.abs(C - expected_tensor).max().item()
            shmem.error(f"Rank {rank}: Validation FAILED, max diff: {max_diff}")
        else:
            shmem.info("Validation PASSED!")
        shmem.barrier()
        json_writer.add_field("success", success)

    # ── Benchmark ────────────────────────────────────────────────────────
    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["copy_engine", "baseline"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["copy_engine", "baseline"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        shmem.barrier()

        shmem.info("Benchmarking copy engine variant...")

        # Calculate TFLOPS: 2*M_local*N*K flops per rank (but total is same across all ranks)
        total_flops = 2 * M_local * N * K
        total_tflops_unit = total_flops * 1e-12

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        tflops = total_tflops_unit / (
            (kernel_timing["copy_engine"]["ms"] / kernel_timing["copy_engine"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        # All-gather moves (world_size - 1) * M_local * N * element_size bytes
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["copy_engine"]["ms"] / kernel_timing["copy_engine"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Matmul-all-gather copy engine (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {tflops:.3f} TFLOPS, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "avg_ms",
            kernel_timing["copy_engine"]["ms"] / kernel_timing["copy_engine"]["experiments"],
        )
        json_writer.add_field("copy_engine_experiments", kernel_timing["copy_engine"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark baseline (compute scatter) for comparison
    if args["benchmark_baseline"] and args["benchmark"]:
        shmem.info("Benchmarking baseline (compute scatter) variant...")

        # Warmup
        iris.do_bench(run_baseline_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        kernel_timing["baseline"]["ms"] = 0
        kernel_timing["baseline"]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        workspace.locks.zero_()
        shmem.barrier()

        # Calculate TFLOPS: 2*M_local*N*K flops per rank
        total_flops = 2 * M_local * N * K
        total_tflops_unit = total_flops * 1e-12

        baseline_ms = iris.do_bench(run_baseline_experiment, shmem.barrier)
        baseline_tflops = total_tflops_unit / (
            (kernel_timing["baseline"]["ms"] / kernel_timing["baseline"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        baseline_bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["baseline"]["ms"] / kernel_timing["baseline"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Matmul-all-gather baseline (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{baseline_ms:.3f} ms, {baseline_tflops:.3f} TFLOPS, {baseline_bandwidth_gbps:.3f} GB/s"
        )

        # Calculate speedup
        copy_engine_tflops = tflops
        speedup = (copy_engine_tflops / baseline_tflops) if baseline_tflops > 0 else 0
        shmem.info(f"Speedup (CopyEngine/Baseline): {speedup:.2f}x")

        json_writer.add_field("baseline_tflops", baseline_tflops)
        json_writer.add_field("baseline_bandwidth_gbps", baseline_bandwidth_gbps)
        json_writer.add_field("baseline_ms", baseline_ms)
        json_writer.add_field("speedup_vs_baseline", speedup)

        # Wait for all to finish baseline benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    # Synchronize device before exiting
    for rank in range(world_size):
        _device_quiet_kernel[(world_size,)](
                        shmem.get_copy_engine_ctx(),
                        rank,
                        world_size,
                    )
    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting matmul_all_gather_copy_engine benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/matmul_all_gather/benchmark_copy_engine.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
