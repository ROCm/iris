#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for iris.ops matmul_all_gather fused operation.

This benchmark showcases the fused GEMM + All-Gather operation where each rank
computes a local matmul and then gathers results along M dimension.
"""

import os
import torch
import torch.distributed as dist
import random
import argparse

from examples.common.utils import JSONWriter

import iris
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
        description="Benchmark matmul_all_gather fused operation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=16384, help="Number of rows per rank in matrix A (M_local)")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B (N)")
    parser.add_argument("-k", type=int, default=131072, help="Common dimension (K)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
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
        default="matmul_all_gather.json",
        help="Output file",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 34, help="Iris heap size")
    parser.add_argument("--comm_sms", type=int, default=None, help="Number of SMs for operation (auto-detect if None)")
    parser.add_argument(
        "--benchmark_pytorch",
        action="store_true",
        help="Also benchmark PyTorch (matmul + all_gather_into_tensor) for comparison",
    )
    parser.add_argument("--block_size_m", type=int, default=None, help="Block size for M dimension (auto if None)")
    parser.add_argument("--block_size_n", type=int, default=None, help="Block size for N dimension (auto if None)")
    parser.add_argument("--block_size_k", type=int, default=None, help="Block size for K dimension (auto if None)")
    parser.add_argument(
        "--group_size_m", type=int, default=None, help="Group size for M dimension tiling (auto if None)"
    )
    parser.add_argument("--num_warps", type=int, default=None, help="Number of warps (auto if None)")
    parser.add_argument("--num_xcds", type=int, default=None, help="Number of XCDs (auto-detected if not set)")
    return vars(parser.parse_args())


def _apply_model_defaults(args, world_size, dtype_bytes=2):
    """Fill None-valued kernel parameters with model-derived predictions.

    Returns a list of parameter names that were set by the model.
    """
    applied = []
    if _DERIVE_AVAILABLE:
        try:
            # Note: args["m"] is M_local (rows per rank)
            # M_local = args["m"]
            # M = M_local * world_size  # Total M after gather
            # N = args["n"]
            # K = args["k"]

            # Derive parameters from performance model
            p = _derive_params(
                args["m"],
                args["n"],
                args["k"],
                world_size=world_size,
                link_bw=50.0,  # Default, could be profiled
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

    # Datatype mapping
    datatype = torch.float32
    if args["datatype"] == "fp16":
        datatype = torch.float16
    elif args["datatype"] == "fp32":
        datatype = torch.float32
    elif args["datatype"] == "bf16":
        datatype = torch.bfloat16
    else:
        print("Unknown datatype.")
        exit(1)

    dtype_bytes = torch.tensor([], dtype=datatype).element_size()

    # Apply model-derived defaults
    model_applied = _apply_model_defaults(args, world_size, dtype_bytes)
    if rank == 0 and model_applied:
        shmem.info(f"Model-derived defaults: {', '.join(model_applied)}")
    if rank == 0:
        param_summary = " ".join(f"{k}={args[k]}" for k in _MODEL_PARAMS)
        shmem.info(f"Kernel params: {param_summary}")

    M_local = args["m"]  # Local M dimension
    M = M_local * world_size  # Total M after gather
    N = args["n"]
    K = args["k"]

    # Create config with parameters
    config_kwargs = {
        "block_size_m": args["block_size_m"],
        "block_size_n": args["block_size_n"],
        "block_size_k": args["block_size_k"],
        "group_size_m": args["group_size_m"],
    }
    if args["comm_sms"] is not None:
        config_kwargs["num_sms"] = args["comm_sms"]
    if args["num_xcds"] is not None:
        config_kwargs["num_xcds"] = args["num_xcds"]

    config = FusedConfig(**config_kwargs)

    json_writer = JSONWriter(args["output_file"])
    json_writer.add_field("world_size", world_size)
    json_writer.add_field("operation", "matmul_all_gather")
    json_writer.add_field("m_local", M_local)
    json_writer.add_field("m_total", M)

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
    # A_local is M_local x K, output is M x N (gathered)
    A_local = shmem.zeros((M_local, K), dtype=datatype)
    B = shmem.zeros((K, N), dtype=datatype)
    C = shmem.zeros((M, N), dtype=datatype)
    expected_tensor = None

    # Fill inputs with deterministic values
    # Each rank has different A_local, same B
    torch.manual_seed(123 + rank)
    A_local_data = torch.randn((M_local, K), dtype=datatype, device=f"cuda:{rank}")
    A_local.copy_(A_local_data)

    torch.manual_seed(456)  # Same B for all ranks
    B_data = torch.randn((K, N), dtype=datatype, device=f"cuda:{rank}")
    B.copy_(B_data)

    # For validation: compute expected result
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

    comm_stream = torch.cuda.Stream()

    kernel_timing = {
        "matmul_all_gather": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }

    workspace = None

    def run_experiment():
        nonlocal kernel_timing, workspace

        shmem.barrier()

        torch.cuda.nvtx.range_push("Matmul-All-Gather")
        with torch.cuda.stream(comm_stream):
            kernel_timing["matmul_all_gather"]["start_event"].record()
            shmem.ops.matmul_all_gather(
                C,
                A_local,
                B,
                config=config,
                async_op=False,
                workspace=workspace,
            )
            kernel_timing["matmul_all_gather"]["end_event"].record()
            kernel_timing["matmul_all_gather"]["experiments"] += 1
        torch.cuda.nvtx.range_pop()

        # Synchronize before querying event timing
        shmem.barrier()

        # Update timing
        ms = kernel_timing["matmul_all_gather"]["start_event"].elapsed_time(
            kernel_timing["matmul_all_gather"]["end_event"]
        )
        kernel_timing["matmul_all_gather"]["ms"] += ms

    # Synchronize across all GPUs
    shmem.barrier()

    if args["validate"]:
        shmem.info("Validating...")

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

    if args["benchmark"]:
        # Warmup for benchmarking
        for k in ["matmul_all_gather"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        iris.do_bench(run_experiment, shmem.barrier, n_warmup=25, n_repeat=1)

        for k in ["matmul_all_gather"]:
            kernel_timing[k]["ms"] = 0
            kernel_timing[k]["experiments"] = 0

        # Reset output before benchmarking
        C.zero_()
        shmem.barrier()

        shmem.info("Benchmarking...")

        # Calculate TFLOPS: 2*M_local*N*K flops per rank (but total is same across all ranks)
        total_flops = 2 * M_local * N * K
        total_tflops_unit = total_flops * 1e-12

        triton_ms = iris.do_bench(run_experiment, shmem.barrier)
        tflops = total_tflops_unit / (
            (kernel_timing["matmul_all_gather"]["ms"] / kernel_timing["matmul_all_gather"]["experiments"]) * 1e-3
        )

        # Calculate bandwidth for all-gather part
        # All-gather moves (world_size - 1) * M_local * N * element_size bytes
        element_size = torch.tensor([], dtype=datatype).element_size()
        output_bytes = M_local * N * element_size
        total_bytes = output_bytes * (world_size - 1)
        total_bytes_gb = total_bytes / (1024**3)

        bandwidth_gbps = total_bytes_gb / (
            (kernel_timing["matmul_all_gather"]["ms"] / kernel_timing["matmul_all_gather"]["experiments"]) * 1e-3
        )

        shmem.info(
            f"Matmul-all-gather (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
            f"{triton_ms:.3f} ms, {tflops:.3f} TFLOPS, {bandwidth_gbps:.3f} GB/s"
        )

        json_writer.add_field("tflops", tflops)
        json_writer.add_field("bandwidth_gbps", bandwidth_gbps)
        json_writer.add_field("total_ms", triton_ms)
        json_writer.add_field("total_flops", total_flops)
        json_writer.add_field("total_bytes", total_bytes)
        json_writer.add_field("total_bytes_gb", total_bytes_gb)
        json_writer.add_field(
            "matmul_all_gather_ms",
            kernel_timing["matmul_all_gather"]["ms"] / kernel_timing["matmul_all_gather"]["experiments"],
        )
        json_writer.add_field("matmul_all_gather_experiments", kernel_timing["matmul_all_gather"]["experiments"])

        # Wait for all to finish benchmarking
        shmem.barrier()

    # Benchmark PyTorch (matmul + all_gather_into_tensor) for comparison
    if args["benchmark_pytorch"]:
        shmem.info("Benchmarking PyTorch (matmul + all_gather_into_tensor)...")

        # Create PyTorch tensors (not on Iris heap)
        pytorch_A_local = torch.randn(M_local, K, dtype=datatype, device=f"cuda:{rank}")
        pytorch_B = torch.randn(K, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_C_local = torch.zeros(M_local, N, dtype=datatype, device=f"cuda:{rank}")
        pytorch_C = torch.zeros(M, N, dtype=datatype, device=f"cuda:{rank}")

        # Warmup
        for _ in range(10):
            pytorch_C_local = torch.matmul(pytorch_A_local, pytorch_B)
            dist.all_gather_into_tensor(pytorch_C, pytorch_C_local)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        dist.barrier()

        def run_pytorch_experiment():
            pytorch_C_local = torch.matmul(pytorch_A_local, pytorch_B)
            dist.all_gather_into_tensor(pytorch_C, pytorch_C_local)

        pytorch_ms = iris.do_bench(run_pytorch_experiment, dist.barrier)

        # Calculate TFLOPS and bandwidth
        pytorch_tflops = total_tflops_unit / (pytorch_ms * 1e-3)
        pytorch_bandwidth_gbps = total_bytes_gb / (pytorch_ms * 1e-3)

        shmem.info(
            f"PyTorch matmul+all_gather_into_tensor (M_local={M_local}, M_total={M}, N={N}, K={K}, world_size={world_size}, dtype={args['datatype']}): "
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
    print("Starting matmul_all_gather benchmark...")
    args = parse_args()
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        _worker(args)
    else:
        print(
            "Please run with torchrun:\n"
            "  torchrun --nproc_per_node=N "
            "benchmark/ops/matmul_all_gather/benchmark.py [OPTIONS]"
        )


if __name__ == "__main__":
    main()
