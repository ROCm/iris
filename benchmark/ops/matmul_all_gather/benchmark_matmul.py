#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris.ops matmul operation.

This benchmark showcases the GEMM  operation where each rank
computes a local matmul.
"""

import os
import torch
import torch.distributed as dist
import random
import argparse

from examples.common.utils import JSONWriter

import iris
from iris.ops.matmul import (
    matmul,
    matmul_preamble,
)
from iris.ops import FusedConfig

# Try to import performance model
_DERIVE_AVAILABLE = False
try:
    import sys as _sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)
    from derive_params import (
        derive as _derive_params,
        DEFAULT_NUM_CUS,
        DEFAULT_PEAK_TFLOPS_FP16,
        DEFAULT_HBM_BW_GBPS,
        DEFAULT_L2_SIZE_BYTES,
        DEFAULT_SCHEDULING_FACTOR,
    )

    _DERIVE_AVAILABLE = True
except Exception:
    pass

_MODEL_PARAMS = (
    "block_size_m",
    "block_size_n",
    "block_size_k",
    "group_size_m",
    "num_warps",
)

_FALLBACK_DEFAULTS = {
    "block_size_m": 256,
    "block_size_n": 128,
    "block_size_k": 64,
    "group_size_m": 4,
}

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark matmul operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
    parser.add_argument("-v", "--validate", action="store_true", help="Enable validation mode")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Enable benchmarking mode")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Tensor datatype",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--num_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size M (model-derived if omitted)")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size N (model-derived if omitted)")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size K (model-derived if omitted)")
    parser.add_argument("--group_size_m", type=int, default=None, help="Group size M (model-derived if omitted)")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto if None)")
    parser.add_argument("--num_warps", type=int, default=None, help="Triton num_warps (auto if None)")
    parser.add_argument("--num_stages", type=int, default=None, help="Triton num_stages (auto if None)")
    parser.add_argument(
        "--output_file",
        type=str,
        default="matmul.json",
        help="Output file",
    )
    parser.add_argument(
        "--benchmark_pytorch",
        action="store_true",
        help="Also benchmark PyTorch (all_gather_into_tensor + matmul) for comparison",
    )


    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype_bytes=2):
    """Fill None-valued kernel parameters with model-derived predictions.

    Returns a list of parameter names that were set by the model.
    """
    applied = []
    if _DERIVE_AVAILABLE:
        try:
            # Derive parameters from performance model
            p = _derive_params(
                args["m"],
                args["n"],
                args["k"],
                world_size,
                link_bw=50.0,
                num_cus=DEFAULT_NUM_CUS,
                peak_tflops=DEFAULT_PEAK_TFLOPS_FP16,
                hbm_bw_gbps=DEFAULT_HBM_BW_GBPS,
                l2_size=DEFAULT_L2_SIZE_BYTES,
                scheduling_factor=DEFAULT_SCHEDULING_FACTOR,
                dtype_bytes=dtype_bytes,
            )
            for name in _MODEL_PARAMS:
                if args.get(name) is None and name in p:
                    args[name] = p[name]
                    applied.append(name)
        except Exception:
            pass

    # Apply fallback defaults for any remaining None values
    for name, fallback in _FALLBACK_DEFAULTS.items():
        if args.get(name) is None:
            args[name] = fallback

    return applied


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

    # Apply model-derived defaults
    model_applied = _apply_model_defaults(args, world_size, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"Model-derived defaults: {', '.join(model_applied)}")
    if rank == 0:
        param_summary = " ".join(f"{k}={args[k]}" for k in _MODEL_PARAMS)
        shmem.info(f"Kernel params: {param_summary}")

    M = args["m"]
    N = args["n"]
    K = args["k"]

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
    json_writer.add_field("operation", "matmul")

    for key, value in args.items():
        json_writer.add_field(key, value)

    # Export actual config values to JSON (including defaults)
    json_writer.add_field("block_size_m", config.block_size_m)
    json_writer.add_field("block_size_n", config.block_size_n)
    json_writer.add_field("block_size_k", config.block_size_k)
    json_writer.add_field("group_size_m", config.group_size_m)
    json_writer.add_field("num_sms", config.num_sms)
    json_writer.add_field("num_xcds", config.num_xcds)

    # Create input and output tensors
    # A_local is M x K, output is M x N (local matmul, no gather)
    A_local = shmem.zeros((M, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # Expected
    expected_tensor = None
    if args["validate"]:
        # Plain matmul: just A_local @ B (local computation, no gather)
        expected_tensor = torch.matmul(A_local_data, B_data)

    # Pre-allocate workspace
    workspace = matmul_preamble(shmem, A_local, B, config)

    # ── Timing ───────────────────────────────────────────────────────────
    comm_stream = torch.cuda.Stream()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    num_experiments = 0

    num_warps = args["num_warps"]
    num_stages = args["num_stages"]

    def run_experiment():
        nonlocal total_ms, num_experiments
        shmem.barrier()

        with torch.cuda.stream(comm_stream):
            start_ev.record()
            matmul(
                shmem,
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            end_ev.record()
            num_experiments += 1
        shmem.barrier()
        total_ms += start_ev.elapsed_time(end_ev)

    shmem.barrier()

    # ── Validate ─────────────────────────────────────────────────────────
    if args["validate"]:
        shmem.info("Validating...")
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
        if args.get("single_run"):
            n_warmup, n_repeat = 0, 1
        else:
            n_warmup, n_repeat = 25, 100

        # Warmup
        total_ms = 0.0
        num_experiments = 0
        if n_warmup > 0:
            iris.do_bench(run_experiment, shmem.barrier, n_warmup=n_warmup, n_repeat=1)

        total_ms = 0.0
        num_experiments = 0
        C.zero_()
        shmem.barrier()

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=0, n_repeat=n_repeat)
        avg_ms = total_ms / num_experiments if num_experiments > 0 else 0

        total_flops = 2 * M * N * K
        tflops = (total_flops * 1e-12) / (avg_ms * 1e-3) if avg_ms > 0 else 0
        element_size = torch.tensor([], dtype=datatype).element_size()
        # Plain matmul has no communication, just local compute
        input_bytes = (M * K + K * N) * element_size
        output_bytes = M * N * element_size
        total_bytes = input_bytes + output_bytes
        total_bytes_gb = total_bytes / (1024**3)
        bw_gbps = (total_bytes / (1024**3)) / (avg_ms * 1e-3) if avg_ms > 0 else 0

        shmem.info(
            f"Matmul (M={M}, N={N}, K={K}, dtype={args['datatype']}): "
            f"{avg_ms:.3f} ms, {tflops:.3f} TFLOPS, {bw_gbps:.3f} GB/s (HBM)"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bw_gbps)
        json_writer.add_field("avg_ms", avg_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark PyTorch (all_gather_into_tensor + matmul) for comparison
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (all_gather_into_tensor + matmul)...")

        # Create PyTorch tensors (not on Iris heap)
        pytorch_A = torch.randn(M, K, dtype=datatype, device=f"cuda:{rank}")
        pytorch_B = torch.randn(K, N, dtype=datatype, device=f"cuda:{rank}")
        # pytorch_A_gathered = torch.zeros(M, K, dtype=datatype, device=f"cuda:{rank}")
        pytorch_C = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")

        # Warmup
        for _ in range(10):
            # dist.all_gather_into_tensor(pytorch_A_gathered, pytorch_A_sharded)
            pytorch_C = torch.matmul(pytorch_A, pytorch_B)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        dist.barrier()

        # Calculate TFLOPS: 2*M*N*K flops
        total_flops = 2 * M * N * K
        total_tflops_unit = total_flops * 1e-12

        # Calculate bandwidth for all-gather part
        element_size = torch.tensor([], dtype=datatype).element_size()
        input_bytes = M * K * element_size
        total_bytes = input_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        def run_pytorch_experiment():
            # dist.all_gather_into_tensor(pytorch_A_gathered, pytorch_A_sharded)
            pytorch_C = torch.matmul(pytorch_A, pytorch_B)

        pytorch_ms = iris.do_bench(run_pytorch_experiment, dist.barrier)

        # Calculate TFLOPS and bandwidth
        pytorch_tflops = total_tflops_unit / (pytorch_ms * 1e-3)
        pytorch_bandwidth_gbps = total_bytes_gb / (pytorch_ms * 1e-3)

        shmem.info(
            f"PyTorch all_gather_into_tensor+matmul (M={M}, K={K}, N={N}, world_size={world_size}, dtype={args['datatype']}): "
            f"{pytorch_ms:.3f} ms, {pytorch_tflops:.3f} TFLOPS, {pytorch_bandwidth_gbps:.3f} GB/s"
        )

        if args["benchmark"]:
            # Calculate performance ratio
            iris_tflops = tflops
            speedup = (iris_tflops / pytorch_tflops) if pytorch_tflops > 0 else 0
            shmem.info(f"Speedup (Iris/PyTorch): {speedup:.2f}x")

            json_writer.add_field("pytorch_tflops", pytorch_tflops)
            json_writer.add_field("pytorch_bandwidth_gbps", pytorch_bandwidth_gbps)
            json_writer.add_field("pytorch_ms", pytorch_ms)
            json_writer.add_field("iris_speedup", speedup)

        # Wait for all to finish PyTorch benchmarking
        shmem.barrier()

    if rank == 0:
        json_writer.flush()
        json_writer.display()

    shmem.barrier()
    dist.destroy_process_group()


def main():
    print("Starting matmul benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/matmul_all_gather/benchmark_matmul.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
