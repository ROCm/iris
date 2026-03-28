#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Tracing example for fused all-gather + GEMM.

Runs the fused kernel with iris device-side tracing enabled and exports
Perfetto-compatible JSON traces for visualization at https://ui.perfetto.dev.

Usage:
    torchrun --nproc_per_node=4 scripts/trace_fused_ag_gemm.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import iris
from iris.ccl import Config


def main():
    if not dist.is_initialized():
        print("Run with: torchrun --nproc_per_node=<N> scripts/trace_fused_ag_gemm.py")
        sys.exit(1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    props = torch.cuda.get_device_properties(device)
    total_sms = props.multi_processor_count

    if rank == 0:
        print(f"GPU: {props.name}, SMs: {total_sms}, World size: {world_size}")

    # Problem size: TP-relevant shape
    M, K_local, N = 1024, 2048, 4096
    K = K_local * world_size
    dtype = torch.float16

    heap_size = 2**33  # 8 GB
    shmem = iris.iris(heap_size)

    # Enable tracing
    shmem.tracing.enable(max_events=100_000)

    # Allocate tensors
    torch.manual_seed(42 + rank)
    A_shard = torch.randn(M, K_local, dtype=dtype, device=device)
    torch.manual_seed(123)
    weight = torch.randn(K, N, dtype=dtype, device=device)

    A_shard_sym = shmem.zeros((M, K_local), dtype=dtype)
    A_shard_sym.copy_(A_shard)
    weight_sym = shmem.zeros((K, N), dtype=dtype)
    weight_sym.copy_(weight)
    output = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()

    # Use optimized config
    config = Config(
        block_size_m=256,
        block_size_n=64,
        swizzle_size=6,
        comm_sms=total_sms,
        num_warps=4,
        num_stages=1,
    )

    # Warmup (without tracing)
    for _ in range(10):
        shmem.ccl.all_gather_gemm(output, A_shard_sym, weight_sym, config=config, block_size_k=64)
    torch.cuda.synchronize()

    # Reset trace buffers and run with tracing
    shmem.tracing.reset()

    # Single traced iteration
    shmem.ccl.all_gather_gemm(output, A_shard_sym, weight_sym, config=config, block_size_k=64, tracing=True)
    torch.cuda.synchronize()

    # Export traces
    trace_dir = "traces"
    os.makedirs(trace_dir, exist_ok=True)

    trace_data = shmem.tracing.export(
        filename=os.path.join(trace_dir, "fused_ag_gemm_trace.json"),
        merge=True,
    )

    if rank == 0:
        num_events = trace_data.get("metadata", {}).get("total_events", 0) if trace_data else 0
        print(f"Exported {num_events} trace events")
        print("View at: https://ui.perfetto.dev")
        print(f"Trace files in: {trace_dir}/")

    shmem.barrier()
    del shmem
    gc.collect()


if __name__ == "__main__":
    main()
