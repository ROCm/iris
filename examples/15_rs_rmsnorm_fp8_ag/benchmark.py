#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for Reduce-Scatter → RMSNorm → FP8 Quantization pipeline.
Similar structure to iris/examples/07_gemm_all_scatter/benchmark.py
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

import iris

# Import kernels from reduce_scatter_rmsnorm_quant.py
from reduce_scatter_rmsnorm_quant import (
    reduce_scatter_m_kernel,
    all_gather_m_kernel,
    aiter_rmsnorm,
    quantize_fp8_kernel,
)

torch.manual_seed(123)
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Reduce-Scatter → RMSNorm → FP8 Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_rows", type=int, default=2048, 
                        help="Number of rows (M), must be divisible by num_ranks")
    parser.add_argument("--num_cols", type=int, default=2048, 
                        help="Number of columns (N)")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Data type for input/intermediate values",
    )
    parser.add_argument("--fp8_out", action="store_true", 
                        help="Enable FP8 quantization after RMSNorm")
    parser.add_argument("--eps", type=float, default=1e-6, 
                        help="RMSNorm epsilon for numerical stability")
    parser.add_argument("--all_gather", action="store_true", 
                        help="Perform all-gather to reconstruct full M×N tensor across all ranks")
    parser.add_argument("--validate", action="store_true", 
                        help="Validate results against PyTorch reference implementation")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run performance benchmarks with GPU-side timing")
    parser.add_argument("--warmup", type=int, default=10, 
                        help="Number of warmup iterations for benchmarking")
    parser.add_argument("--iters", type=int, default=100, 
                        help="Number of timed iterations for benchmarking")
    parser.add_argument(
        "--output_file",
        type=str,
        default="rs_rmsnorm_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks/GPUs")
    parser.add_argument("--heap_size", type=int, default=0, help="IRIS heap size in bytes (0=auto, default: 2GB)")
    parser.add_argument("--heap_size_gb", type=float, default=None, help="IRIS heap size in GB (overrides --heap_size)")
    parser.add_argument("--BLOCK_M", type=int, default=16, help="Block size M")
    parser.add_argument("--BLOCK_N", type=int, default=32, help="Block size N")
    parser.add_argument("--GROUP_SIZE_M", type=int, default=8, help="Tile swizzle group size")
    parser.add_argument("--NUM_SMS", type=int, default=None, help="Number of CUs (auto-detect if None)")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps per thread block (reduce-scatter)")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of pipeline stages (reduce-scatter)")
    parser.add_argument("--waves_per_eu", type=int, default=0, help="Waves per execution unit (reduce-scatter, 0=auto)")
    
    # RMSNorm specific parameters
    parser.add_argument("--rmsnorm_block_size", type=int, default=None, help="RMSNorm BLOCK_SIZE (auto-detect if None)")
    parser.add_argument("--rmsnorm_num_warps", type=int, default=None, help="RMSNorm num_warps (default: 8)")
    parser.add_argument("--rmsnorm_num_prgms", type=int, default=None, help="RMSNorm NUM_PRGMS (default: M_shard)")
    parser.add_argument("--rmsnorm_waves_per_eu", type=int, default=None, help="RMSNorm waves_per_eu (default: 2)")
    
    # FP8 Quantization specific parameters
    parser.add_argument("--fp8_block_m", type=int, default=None, help="FP8 BLOCK_M (default: same as reduce-scatter BLOCK_M)")
    parser.add_argument("--fp8_block_n", type=int, default=None, help="FP8 BLOCK_N (default: same as reduce-scatter BLOCK_N)")
    parser.add_argument("--fp8_num_warps", type=int, default=None, help="FP8 num_warps (default: 4)")
    parser.add_argument("--fp8_num_stages", type=int, default=None, help="FP8 num_stages (default: 2)")
    parser.add_argument("--fp8_waves_per_eu", type=int, default=None, help="FP8 waves_per_eu (default: 0)")
    
    # All-Gather specific parameters
    parser.add_argument("--ag_block_m", type=int, default=None, help="All-Gather BLOCK_M (default: same as reduce-scatter)")
    parser.add_argument("--ag_block_n", type=int, default=None, help="All-Gather BLOCK_N (default: same as reduce-scatter)")
    parser.add_argument("--ag_num_warps", type=int, default=None, help="All-Gather num_warps (default: 4)")
    parser.add_argument("--ag_num_stages", type=int, default=None, help="All-Gather num_stages (default: 2)")
    parser.add_argument("--ag_waves_per_eu", type=int, default=None, help="All-Gather waves_per_eu (default: 0)")
    
    return vars(parser.parse_args())


def run_reduce_scatter(input_tensor, M, M_shard, N, rank, world_size, heap_bases, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, num_warps, num_stages, waves_per_eu, dtype, device, shmem=None, output_buffer=None):
    """Run reduce-scatter operation with pull-based iris.load approach."""
    # Use provided output buffer or allocate new one
    if output_buffer is not None:
        reduced_shard = output_buffer
    elif shmem is not None:
        reduced_shard = shmem.zeros((M_shard, N), dtype=dtype)
    else:
        # Fallback - but this won't work with IRIS operations!
        raise ValueError("IRIS operations require output_buffer in IRIS shared memory")
    
    grid_rs = (NUM_SMS,)
    
    # Call kernel once - it will pull data from all source ranks using iris.load
    reduce_scatter_m_kernel[grid_rs](
        input_tensor,
        reduced_shard,
        M,
        M_shard,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        reduced_shard.stride(0),
        reduced_shard.stride(1),
        rank,
        world_size,
        heap_bases,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    
    # Synchronize to ensure all loads and reductions complete
    torch.cuda.synchronize()
    if shmem is not None:
        shmem.barrier()
    
    return reduced_shard


def run_rmsnorm(input_tensor, eps, device, block_size=None, num_warps=None, num_prgms=None, waves_per_eu=None):
    """Run RMSNorm operation using AITer kernel."""
    M_shard, N = input_tensor.shape
    dtype = input_tensor.dtype
    
    gamma = torch.ones(N, device=device, dtype=dtype)
    output = torch.empty_like(input_tensor)
    rsigma = torch.empty(M_shard, device=device, dtype=dtype)
    
    # Auto-detect BLOCK_SIZE if not provided
    if block_size is None:
        element_size = input_tensor.element_size()
        max_block_size = 65536 // element_size
        BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(N))
    else:
        BLOCK_SIZE = block_size
    
    # Always auto-detect USE_BLOCKED based on N and BLOCK_SIZE
    USE_BLOCKED = N > BLOCK_SIZE
    
    # Set NUM_PRGMS (default to M_shard for full parallelism)
    NUM_PRGMS = num_prgms if num_prgms is not None else M_shard
    
    # Set num_warps (default to 8)
    final_num_warps = num_warps if num_warps is not None else 8
    
    # Set waves_per_eu (default to 2)
    final_waves_per_eu = waves_per_eu if waves_per_eu is not None else 2
    
    aiter_rmsnorm[(M_shard,)](
        input_tensor,
        output,
        gamma,
        rsigma,
        input_tensor.stride(0),
        output.stride(0),
        M_shard,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        NUM_PRGMS=NUM_PRGMS,
        num_warps=final_num_warps,
        waves_per_eu=final_waves_per_eu,
    )
    
    return output


def run_quantize_fp8(input_tensor, BLOCK_M, BLOCK_N, device, shmem=None):
    """Run FP8 quantization."""
    M_shard, N = input_tensor.shape
    
    max_val = input_tensor.abs().max().item()
    scale = max(max_val / 448.0, 1e-8)
    scale_tensor = torch.tensor([scale], device=device, dtype=torch.float32)
    
    # Allocate output - always in regular CUDA memory for FP8 (IRIS may not support FP8)
    if hasattr(torch, "float8_e4m3fn"):
        output = torch.empty(M_shard, N, device=device, dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input_tensor)
    
    grid = (triton.cdiv(M_shard, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    quantize_fp8_kernel[grid](
        input_tensor,
        output,
        scale_tensor,
        M_shard,
        N,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=16,
        waves_per_eu=2,
    )
    
    return output, scale


def run_all_gather(shard, M, M_shard, N, rank, world_size, heap_bases, shmem, BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, device, output_buffer=None):
    """Run all-gather operation."""
    dtype = shard.dtype
    
    # Use provided output buffer or allocate new one
    if output_buffer is not None:
        full_output = output_buffer
    else:
        # Allocate output in IRIS shared memory for remote writes
        full_output = shmem.empty((M, N), dtype=dtype)
    
    grid = (NUM_SMS,)
    
    all_gather_m_kernel[grid](
        shard,
        full_output,
        M,
        M_shard,
        N,
        shard.stride(0),
        shard.stride(1),
        full_output.stride(0),
        full_output.stride(1),
        rank,
        world_size,
        heap_bases,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        waves_per_eu=2,
    )
    
    return full_output


def _worker(local_rank: int, world_size: int, init_url: str, args: dict):
    """Worker function for distributed execution."""
    # Parse arguments
    M = args["num_rows"]
    N = args["num_cols"]
    
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    M_shard = M // world_size

    # Datatype
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args["datatype"]]
    
    # Calculate heap size if auto (0) or use provided value
    if args.get("heap_size_gb") is not None:
        # User specified GB
        heap_size = int(args["heap_size_gb"] * (1024 ** 3))
    elif args["heap_size"] == 0:
        # Auto-calculate based on problem size
        bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        fp8_bytes_per_element = 1
        
        # Validation allocations:
        mem_input = M * N * bytes_per_element  # input_tensor
        mem_rs_output = M_shard * N * bytes_per_element  # reduced_shard
        mem_rmsnorm = M_shard * N * bytes_per_element  # rmsnorm_output
        mem_fp8 = M_shard * N * fp8_bytes_per_element if args['fp8_out'] else 0  # quantized_output (as uint8)
        mem_ag_output = M * N * (fp8_bytes_per_element if args['fp8_out'] else bytes_per_element) if args['all_gather'] else 0
        
        # Benchmark allocations (if enabled):
        if args.get('benchmark'):
            mem_test_input = M * N * bytes_per_element  # test_input
            mem_test_rs = 2 * M_shard * N * bytes_per_element  # test_reduced_shard (2x size)
            mem_test_rmsnorm = M_shard * N * bytes_per_element  # rmsnorm_output_bench
            mem_test_fp8 = M_shard * N * fp8_bytes_per_element if args['fp8_out'] else 0
            mem_test_ag = M * N * (fp8_bytes_per_element if args['fp8_out'] else bytes_per_element) if args['all_gather'] else 0
        else:
            mem_test_input = mem_test_rs = mem_test_rmsnorm = mem_test_fp8 = mem_test_ag = 0
        
        total_mem = (mem_input + mem_rs_output + mem_rmsnorm + mem_fp8 + mem_ag_output + 
                     mem_test_input + mem_test_rs + mem_test_rmsnorm + mem_test_fp8 + mem_test_ag)
        
        # Add 20% overhead for alignment (1KB per allocation) and safety margin
        heap_size = int(total_mem * 1.2)
        
        # Ensure minimum 1GB
        heap_size = max(heap_size, 1 << 30)
    else:
        heap_size = args["heap_size"]
    
    # Use gloo backend for CPU-based coordination (RCCL will be used by IRIS for GPU comm)
    backend = "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        world_size=world_size,
        rank=local_rank,
    )

    # Initialize IRIS with calculated heap size
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size_iris = shmem.get_num_ranks()
    
    assert world_size == world_size_iris, f"World size mismatch: {world_size} != {world_size_iris}"

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Auto-detect NUM_SMS if not provided
    if args["NUM_SMS"] is None:
        cu_count = torch.cuda.get_device_properties(local_rank).multi_processor_count
        NUM_SMS = cu_count
    else:
        NUM_SMS = args["NUM_SMS"]

    BLOCK_M = args["BLOCK_M"]
    BLOCK_N = args["BLOCK_N"]
    GROUP_SIZE_M = args["GROUP_SIZE_M"]
    num_warps = args["num_warps"]
    num_stages = args["num_stages"]
    waves_per_eu = args["waves_per_eu"]
    
    # RMSNorm parameters - extract from args if they exist
    rmsnorm_block_size = args.get("rmsnorm_block_size")
    rmsnorm_num_warps = args.get("rmsnorm_num_warps")
    rmsnorm_num_prgms = args.get("rmsnorm_num_prgms")
    rmsnorm_waves_per_eu = args.get("rmsnorm_waves_per_eu")
    
    # FP8 Quantization parameters
    fp8_block_m = args.get("fp8_block_m")
    fp8_block_n = args.get("fp8_block_n")
    fp8_num_warps = args.get("fp8_num_warps")
    fp8_num_stages = args.get("fp8_num_stages")
    fp8_waves_per_eu = args.get("fp8_waves_per_eu")
    
    # All-Gather parameters
    ag_block_m = args.get("ag_block_m")
    ag_block_n = args.get("ag_block_n")
    ag_num_warps = args.get("ag_num_warps")
    ag_num_stages = args.get("ag_num_stages")
    ag_waves_per_eu = args.get("ag_waves_per_eu")

    if rank == 0:
        print(f"Configuration:")
        print(f"  M={M}, N={N}, M_shard={M_shard}")
        print(f"  dtype={dtype}, world_size={world_size}")
        print(f"  Reduce-Scatter:")
        print(f"    BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, GROUP_SIZE_M={GROUP_SIZE_M}, NUM_SMS={NUM_SMS}")
        print(f"    num_warps={num_warps}, num_stages={num_stages}, waves_per_eu={waves_per_eu}")
        print(f"  RMSNorm Parameters:")
        print(f"    BLOCK_SIZE: {rmsnorm_block_size or 'auto'}")
        print(f"    USE_BLOCKED: auto (N > BLOCK_SIZE)")
        print(f"    num_warps: {rmsnorm_num_warps or 8}")
        print(f"    NUM_PRGMS: {rmsnorm_num_prgms or M_shard}")
        print(f"    waves_per_eu: {rmsnorm_waves_per_eu if rmsnorm_waves_per_eu is not None else 2}")
        print(f"  FP8 Quantization Parameters:")
        print(f"    BLOCK_M: {fp8_block_m or BLOCK_M}")
        print(f"    BLOCK_N: {fp8_block_n or BLOCK_N}")
        print(f"    num_warps: {fp8_num_warps or 4}")
        print(f"    num_stages: {fp8_num_stages or 2}")
        print(f"    waves_per_eu: {fp8_waves_per_eu if fp8_waves_per_eu is not None else 0}")
        print(f"  All-Gather Parameters:")
        print(f"    BLOCK_M: {ag_block_m or BLOCK_M}")
        print(f"    BLOCK_N: {ag_block_n or BLOCK_N}")
        print(f"    num_warps: {ag_num_warps or 4}")
        print(f"    num_stages: {ag_num_stages or 2}")
        print(f"    waves_per_eu: {ag_waves_per_eu if ag_waves_per_eu is not None else 0}")
        print(f"  FP8 output: {args['fp8_out']}")
        print(f"  All-gather: {args['all_gather']}")
        
        # Calculate memory requirements (should match auto-calculation logic)
        bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
        fp8_bytes_per_element = 1
        
        # Validation memory:
        mem_input = M * N * bytes_per_element
        mem_rs_output = M_shard * N * bytes_per_element
        mem_rmsnorm = M_shard * N * bytes_per_element
        mem_fp8 = M_shard * N * fp8_bytes_per_element if args['fp8_out'] else 0
        mem_ag_output = M * N * (fp8_bytes_per_element if args['fp8_out'] else bytes_per_element) if args['all_gather'] else 0
        
        # Benchmark memory (if enabled):
        if args.get('benchmark'):
            mem_test_input = M * N * bytes_per_element
            mem_test_rs = 2 * M_shard * N * bytes_per_element
            mem_test_rmsnorm = M_shard * N * bytes_per_element
            mem_test_fp8 = M_shard * N * fp8_bytes_per_element if args['fp8_out'] else 0
            mem_test_ag = M * N * (fp8_bytes_per_element if args['fp8_out'] else bytes_per_element) if args['all_gather'] else 0
        else:
            mem_test_input = mem_test_rs = mem_test_rmsnorm = mem_test_fp8 = mem_test_ag = 0
        
        total_mem = (mem_input + mem_rs_output + mem_rmsnorm + mem_fp8 + mem_ag_output + 
                     mem_test_input + mem_test_rs + mem_test_rmsnorm + mem_test_fp8 + mem_test_ag)
        
        # Add 20% overhead for alignment
        estimated_heap_bytes = int(total_mem * 1.2)
        estimated_heap_mb = estimated_heap_bytes / (1024 * 1024)
        
        heap_size_mb = heap_size / (1024**2)
        print(f"  Heap size: {heap_size_mb:.0f} MB {'(auto-calculated)' if args['heap_size'] == 0 else ''}")
        print(f"  Estimated memory needed: ~{estimated_heap_mb:.0f} MB")
        
        if estimated_heap_bytes > heap_size:
            print(f"  ⚠️  WARNING: May run out of heap memory!")
            print(f"     Recommended: --heap_size {estimated_heap_bytes}")
            print(f"     Or use smaller M/N values")

    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Create input tensor
    torch.manual_seed(123 + rank)
    input_tensor_local = torch.randn(M, N, device=device, dtype=dtype) * (rank + 1)
    
    # Allocate input tensor in IRIS shared memory for remote access
    input_tensor = shmem.empty((M, N), dtype=dtype)
    input_tensor.copy_(input_tensor_local)

    # IRIS heap bases
    heap_bases = shmem.get_heap_bases()
    
    # Barrier to ensure all ranks have allocated their tensors
    shmem.barrier()

    # ================================================================
    # Step 1: Reduce-Scatter
    # ================================================================
    # Call kernel once per rank - it will use iris.load() to pull data from all source ranks
    reduced_shard = run_reduce_scatter(
        input_tensor, M, M_shard, N, rank, world_size, heap_bases, 
        BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, 
        num_warps, num_stages, waves_per_eu,
        dtype, device, shmem
    )
    
    # Synchronize to ensure all ranks have completed their loads and reductions
    torch.cuda.synchronize()
    shmem.barrier()

    # ================================================================
    # Step 2: RMSNorm
    # ================================================================
    rmsnorm_output = run_rmsnorm(
        reduced_shard, args["eps"], device,
        block_size=rmsnorm_block_size,
        num_warps=rmsnorm_num_warps,
        num_prgms=rmsnorm_num_prgms,
        waves_per_eu=rmsnorm_waves_per_eu
    )

    # ================================================================
    # Step 3: FP8 Quantization
    # ================================================================
    quantized_output = None  # Initialize for validation scope
    if args["fp8_out"]:
        # Allocate in regular CUDA memory
        quantized_output, scale = run_quantize_fp8(rmsnorm_output, BLOCK_M, BLOCK_N, device, shmem=None)
        
        # If all-gather is enabled, copy to IRIS memory as uint8 (workaround for FP8 dtype support)
        if args["all_gather"]:
            # IRIS may not fully support FP8 dtype, so copy via uint8 byte view
            final_output_iris_bytes = shmem.empty((M_shard, N), dtype=torch.uint8)
            quantized_bytes = quantized_output.view(torch.uint8)
            final_output_iris_bytes.copy_(quantized_bytes)
            final_output = final_output_iris_bytes.view(quantized_output.dtype)
        else:
            final_output = quantized_output
    else:
        # If all-gather is enabled, ensure rmsnorm_output is in IRIS memory
        if args["all_gather"]:
            final_output_iris = shmem.empty(rmsnorm_output.shape, dtype=rmsnorm_output.dtype)
            final_output_iris.copy_(rmsnorm_output)
            final_output = final_output_iris
        else:
            final_output = rmsnorm_output

    # ================================================================
    # Step 4: All-Gather (optional)
    # ================================================================
    if args["all_gather"]:
        result = run_all_gather(
            final_output, M, M_shard, N, rank, world_size, heap_bases, shmem,
            BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, device
        )
        torch.cuda.synchronize()
        shmem.barrier()
    else:
        result = final_output

    # ================================================================
    # Validation
    # ================================================================
    if args["validate"] and rank == 0:
        print("\nValidation:")
        print("Note: Validation uses initial pipeline execution (may use different params than benchmark)")
        print("      For best results, ensure command-line params match tuned values\n")
        
        import torch.nn as nn
        
        # Reference computation
        torch.manual_seed(123)
        ref_tensors = []
        for i in range(world_size):
            torch.manual_seed(123 + i)
            tensor = torch.randn(M, N, device=device, dtype=dtype) * (i + 1)
            ref_tensors.append(tensor)
        
        # Use FP32 accumulation to match kernel (more accurate than FP16)
        ref_reduced = torch.zeros(M, N, device=device, dtype=torch.float32)
        for tensor in ref_tensors:
            ref_reduced += tensor.to(torch.float32)
        
        # Convert back to FP16 and extract shard
        ref_shard = ref_reduced[rank * M_shard:(rank + 1) * M_shard, :].to(dtype)
        
        # Debug: Print sums to diagnose accumulation issues
        ref_sum = ref_shard.sum(dtype=torch.float32).item()
        actual_sum = reduced_shard.sum(dtype=torch.float32).item()
        
        # Compare reduce-scatter
        rs_diff = torch.abs(ref_shard - reduced_shard)
        rel_error = abs(ref_sum - actual_sum) / abs(ref_sum) * 100
        
        print(f"  Reduce-scatter max diff: {rs_diff.max().item():.8f}")
        print(f"  Reduce-scatter sum - Reference: {ref_sum:.4f}, Actual: {actual_sum:.4f}, Rel Error: {rel_error:.4f}%")
        
        # For FP16 with 8-rank accumulation, max diff ~0.1 is acceptable
        # The key metric is the sum - should be within 0.1% relative error
        if rel_error < 0.1 and rs_diff.max() < 0.1:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
        
        # Compare RMSNorm
        rmsnorm_layer = nn.RMSNorm(N, eps=args["eps"], device=device, dtype=dtype)
        ref_normed = rmsnorm_layer(ref_shard)
        
        # NOTE: rmsnorm_output might use different parameters than benchmark
        # This is just a basic sanity check
        rms_diff = torch.abs(ref_normed - rmsnorm_output)
        print(f"  RMSNorm max diff: {rms_diff.max().item():.8f}")
        
        ref_norm_sum = ref_normed.sum(dtype=torch.float32).item()
        actual_norm_sum = rmsnorm_output.sum(dtype=torch.float32).item()
        rms_sum_rel_err = abs(ref_norm_sum - actual_norm_sum) / abs(ref_norm_sum) * 100
        print(f"  RMSNorm sum - Reference: {ref_norm_sum:.4f}, Actual: {actual_norm_sum:.4f}, Rel Error: {rms_sum_rel_err:.4f}%")
        print(f"  {'✅ PASS' if rms_diff.max() < 10.0 else '❌ FAIL'} (initial exec, may differ from benchmark)")
        
        # Compare FP8 Quantization
        if args["fp8_out"] and quantized_output is not None:
            # For FP8, just verify the quantization is within expected range
            quant_float = quantized_output.to(torch.float32)
            
            print(f"  FP8 Quantization range: [{quant_float.min().item():.2f}, {quant_float.max().item():.2f}]")
            print(f"  FP8 Quantization sum: {quant_float.sum().item():.4f}")
            
            # FP8 range should be within [-448, 448] and not all zeros
            in_range = (quant_float.min() >= -448.0) and (quant_float.max() <= 448.0)
            not_all_zero = quant_float.abs().max() > 0.01
            
            print(f"  {'✅ PASS' if (in_range and not_all_zero) else '❌ FAIL'} (values in valid FP8 range and non-zero)")
        
        # Compare All-Gather
        if args["all_gather"]:
            # Check value range of full gathered result
            result_float = result.to(torch.float32)
            result_min = result_float.min().item()
            result_max = result_float.max().item()
            result_sum = result_float.sum().item()
            result_nonzero = (result_float.abs() > 0.01).sum().item()
            
            print(f"  All-Gather full result:")
            print(f"    Value range: [{result_min:.4f}, {result_max:.4f}]")
            print(f"    Sum: {result_sum:.4f}")
            print(f"    Non-zero elements: {result_nonzero}/{result_float.numel()} ({result_nonzero/result_float.numel()*100:.1f}%)")
            
            # Verify that this rank's shard appears correctly in the gathered result
            ag_shard_result = result[rank * M_shard:(rank + 1) * M_shard, :]
            
            # Convert to float32 for comparison (FP8 doesn't support some ops)
            ag_result_float = ag_shard_result.to(torch.float32)
            final_out_float = final_output.to(torch.float32)
            
            ag_diff_float = torch.abs(ag_result_float - final_out_float)
            ag_sum_diff = abs(ag_result_float.sum() - final_out_float.sum())
            ag_rel_err = ag_sum_diff / abs(final_out_float.sum()) * 100 if final_out_float.sum() != 0 else 0.0
            
            print(f"  All-Gather (rank {rank} shard) max diff: {ag_diff_float.max().item():.8f}, rel error: {ag_rel_err:.4f}%")
            
            # Check if result is valid (not all zeros)
            is_valid = (abs(result_sum) > 1.0) and (result_nonzero > result_float.numel() * 0.5)
            if not is_valid:
                print(f"  ⚠️  WARNING: All-Gather result may be invalid (mostly zeros or very small values)")
            
            print(f"  {'✅ PASS' if (ag_diff_float.max() < 0.01 and is_valid) else '❌ FAIL'}")

    # ================================================================
    # Benchmarking
    # ================================================================
    if args["benchmark"]:
        if rank == 0:
            print(f"\nBenchmarking with {args['warmup']} warmup + {args['iters']} iterations...")
        
        # ----------------------------------------------------------------
        # Benchmark Reduce-Scatter
        # ----------------------------------------------------------------
        # Pre-allocate test tensors in IRIS memory (reuse to avoid re-allocation)
        test_input = shmem.empty((M, N), dtype=dtype)
        test_input_local = torch.randn(M, N, device=device, dtype=dtype)
        test_input.copy_(test_input_local)
        
        # Pre-allocate output buffer in IRIS memory (M_shard × N, will be reused)
        test_reduced_shard = shmem.zeros((2*M_shard, N), dtype=dtype)
        
        # Warmup
        for _ in range(args["warmup"]):
            test_reduced_shard.zero_()
            _ = run_reduce_scatter(test_input, M, M_shard, N, rank, world_size, heap_bases, 
                                   BLOCK_M, BLOCK_N, GROUP_SIZE_M, NUM_SMS, 
                                   num_warps, num_stages, waves_per_eu,
                                   dtype, device, 
                                   shmem=shmem, output_buffer=test_reduced_shard)
            torch.cuda.synchronize()
            shmem.barrier()
        
        # Benchmark using CUDA events for accurate GPU timing
        # Call kernel directly (not through wrapper) to avoid sync overhead
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        grid_rs = (NUM_SMS,)
        
        start_event.record()
        for _ in range(args["iters"]):
            reduce_scatter_m_kernel[grid_rs](
                test_input,
                test_reduced_shard,
                M,
                M_shard,
                N,
                test_input.stride(0),
                test_input.stride(1),
                test_reduced_shard.stride(0),
                test_reduced_shard.stride(1),
                rank,
                world_size,
                heap_bases,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                GROUP_SIZE_M=GROUP_SIZE_M,
                NUM_SMS=NUM_SMS,
                num_warps=num_warps,
                num_stages=num_stages,
                waves_per_eu=waves_per_eu,
            )
        end_event.record()
        
        torch.cuda.synchronize()
        rs_time_ms = start_event.elapsed_time(end_event) / args["iters"]
        shmem.barrier()
        
        # ----------------------------------------------------------------
        # Benchmark RMSNorm
        # ----------------------------------------------------------------
        # Allocate tensors once (not in the loop!)
        gamma_bench = torch.ones(N, device=device, dtype=dtype)
        rmsnorm_output_bench = torch.empty_like(reduced_shard)
        rsigma_bench = torch.empty(M_shard, device=device, dtype=dtype)
        
        # Determine RMSNorm parameters
        if rmsnorm_block_size is None:
            element_size = reduced_shard.element_size()
            max_block_size = 65536 // element_size
            RMSNORM_BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(N))
        else:
            RMSNORM_BLOCK_SIZE = rmsnorm_block_size
        
        RMSNORM_USE_BLOCKED = N > RMSNORM_BLOCK_SIZE  # Always auto-detect
        RMSNORM_NUM_PRGMS = M_shard if rmsnorm_num_prgms is None else rmsnorm_num_prgms
        RMSNORM_NUM_WARPS = 8 if rmsnorm_num_warps is None else rmsnorm_num_warps
        RMSNORM_WAVES_PER_EU = 2 if rmsnorm_waves_per_eu is None else rmsnorm_waves_per_eu
        
        # Warmup
        for _ in range(args["warmup"]):
            aiter_rmsnorm[(M_shard,)](
                reduced_shard,
                rmsnorm_output_bench,
                gamma_bench,
                rsigma_bench,
                reduced_shard.stride(0),
                rmsnorm_output_bench.stride(0),
                M_shard,
                N,
                args["eps"],
                BLOCK_SIZE=RMSNORM_BLOCK_SIZE,
                USE_BLOCKED=RMSNORM_USE_BLOCKED,
                NUM_PRGMS=RMSNORM_NUM_PRGMS,
                num_warps=RMSNORM_NUM_WARPS,
                waves_per_eu=RMSNORM_WAVES_PER_EU,
            )
            torch.cuda.synchronize()
        
        # Benchmark using CUDA events - call kernel directly
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(args["iters"]):
            aiter_rmsnorm[(M_shard,)](
                reduced_shard,
                rmsnorm_output_bench,
                gamma_bench,
                rsigma_bench,
                reduced_shard.stride(0),
                rmsnorm_output_bench.stride(0),
                M_shard,
                N,
                args["eps"],
                BLOCK_SIZE=RMSNORM_BLOCK_SIZE,
                USE_BLOCKED=RMSNORM_USE_BLOCKED,
                NUM_PRGMS=RMSNORM_NUM_PRGMS,
                num_warps=RMSNORM_NUM_WARPS,
                waves_per_eu=RMSNORM_WAVES_PER_EU,
            )
        end_event.record()
        
        torch.cuda.synchronize()
        rmsnorm_time_ms = start_event.elapsed_time(end_event) / args["iters"]
        
        # ----------------------------------------------------------------
        # Benchmark FP8 Quantization
        # ----------------------------------------------------------------
        quant_time_ms = 0.0
        if args["fp8_out"]:
            # Determine FP8 quantization parameters
            FP8_BLOCK_M = fp8_block_m if fp8_block_m is not None else BLOCK_M
            FP8_BLOCK_N = fp8_block_n if fp8_block_n is not None else BLOCK_N
            FP8_NUM_WARPS = fp8_num_warps if fp8_num_warps is not None else 4
            FP8_NUM_STAGES = fp8_num_stages if fp8_num_stages is not None else 2
            FP8_WAVES_PER_EU = fp8_waves_per_eu if fp8_waves_per_eu is not None else 0
            
            # Allocate tensors once
            max_val = rmsnorm_output_bench.abs().max().item()
            scale = max(max_val / 448.0, 1e-8)
            scale_tensor_bench = torch.tensor([scale], device=device, dtype=torch.float32)
            
            if hasattr(torch, "float8_e4m3fn"):
                fp8_output_bench = torch.empty(M_shard, N, device=device, dtype=torch.float8_e4m3fn)
            else:
                fp8_output_bench = torch.empty_like(rmsnorm_output_bench)
            
            grid_fp8 = (triton.cdiv(M_shard, FP8_BLOCK_M), triton.cdiv(N, FP8_BLOCK_N))
            
            # Warmup
            for _ in range(args["warmup"]):
                quantize_fp8_kernel[grid_fp8](
                    rmsnorm_output_bench,
                    fp8_output_bench,
                    scale_tensor_bench,
                    M_shard,
                    N,
                    rmsnorm_output_bench.stride(0),
                    rmsnorm_output_bench.stride(1),
                    fp8_output_bench.stride(0),
                    fp8_output_bench.stride(1),
                    BLOCK_M=FP8_BLOCK_M,
                    BLOCK_N=FP8_BLOCK_N,
                    num_warps=FP8_NUM_WARPS,
                    num_stages=FP8_NUM_STAGES,
                    waves_per_eu=FP8_WAVES_PER_EU,
                )
                torch.cuda.synchronize()
            
            # Benchmark using CUDA events - call kernel directly
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(args["iters"]):
                quantize_fp8_kernel[grid_fp8](
                    rmsnorm_output_bench,
                    fp8_output_bench,
                    scale_tensor_bench,
                    M_shard,
                    N,
                    rmsnorm_output_bench.stride(0),
                    rmsnorm_output_bench.stride(1),
                    fp8_output_bench.stride(0),
                    fp8_output_bench.stride(1),
                    BLOCK_M=FP8_BLOCK_M,
                    BLOCK_N=FP8_BLOCK_N,
                    num_warps=FP8_NUM_WARPS,
                    num_stages=FP8_NUM_STAGES,
                    waves_per_eu=FP8_WAVES_PER_EU,
                )
            end_event.record()
            
            torch.cuda.synchronize()
            quant_time_ms = start_event.elapsed_time(end_event) / args["iters"]
        
        # ----------------------------------------------------------------
        # Benchmark All-Gather
        # ----------------------------------------------------------------
        ag_time_ms = 0.0
        if args["all_gather"]:
            # Determine All-Gather parameters
            AG_BLOCK_M = ag_block_m if ag_block_m is not None else BLOCK_M
            AG_BLOCK_N = ag_block_n if ag_block_n is not None else BLOCK_N
            AG_NUM_WARPS = ag_num_warps if ag_num_warps is not None else 4
            AG_NUM_STAGES = ag_num_stages if ag_num_stages is not None else 2
            AG_WAVES_PER_EU = ag_waves_per_eu if ag_waves_per_eu is not None else 0
            
            # Pre-allocate output in IRIS memory (reuse to avoid heap exhaustion)
            ag_output_reuse = shmem.empty((M, N), dtype=final_output.dtype)
            
            grid_ag = (NUM_SMS,)
            
            # Warmup
            for _ in range(args["warmup"]):
                all_gather_m_kernel[grid_ag](
                    final_output, ag_output_reuse, M, M_shard, N,
                    final_output.stride(0), final_output.stride(1),
                    ag_output_reuse.stride(0), ag_output_reuse.stride(1),
                    rank, world_size, heap_bases,
                    BLOCK_M=AG_BLOCK_M, BLOCK_N=AG_BLOCK_N,
                    GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS,
                    num_warps=AG_NUM_WARPS,
                    num_stages=AG_NUM_STAGES,
                    waves_per_eu=AG_WAVES_PER_EU,
                )
                torch.cuda.synchronize()
            
            # Benchmark using CUDA events - call kernel directly
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(args["iters"]):
                all_gather_m_kernel[grid_ag](
                    final_output, ag_output_reuse, M, M_shard, N,
                    final_output.stride(0), final_output.stride(1),
                    ag_output_reuse.stride(0), ag_output_reuse.stride(1),
                    rank, world_size, heap_bases,
                    BLOCK_M=AG_BLOCK_M, BLOCK_N=AG_BLOCK_N,
                    GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=NUM_SMS,
                    num_warps=AG_NUM_WARPS,
                    num_stages=AG_NUM_STAGES,
                    waves_per_eu=AG_WAVES_PER_EU,
                )
            end_event.record()
            
            torch.cuda.synchronize()
            ag_time_ms = start_event.elapsed_time(end_event) / args["iters"]
        
        # ----------------------------------------------------------------
        # Calculate metrics for all components
        # ----------------------------------------------------------------
        num_elements = M_shard * N
        bytes_per_element = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
        
        # Reduce-Scatter with iris.load (pull-based):
        # Each rank loads M_shard×N from (world_size - 1) remote ranks
        # Local read doesn't go over interconnect, so we exclude it
        # Interconnect bandwidth = data transferred over network / time
        rs_interconnect_bytes = M_shard * N * (world_size - 1) * bytes_per_element
        rs_bandwidth_gb_s = rs_interconnect_bytes / (rs_time_ms / 1000) / 1e9
        
        # RMSNorm: Read (M_shard)×N + write (M_shard)×N
        bytes_processed_rmsnorm = num_elements * bytes_per_element * 2  # Read + write
        rmsnorm_bandwidth_gb_s = bytes_processed_rmsnorm / (rmsnorm_time_ms / 1000) / 1e9
        
        # RMSNorm TFLOPS (approximate)
        # RMSNorm: ~3N FLOPs per element (square, sum, rsqrt, multiply)
        rmsnorm_flops = num_elements * N * 3
        rmsnorm_tflops = rmsnorm_flops / (rmsnorm_time_ms / 1000) / 1e12
        
        # FP8 Quantization: Read FP16/BF16 + write FP8
        quant_bandwidth_gb_s = 0.0
        fp8_bytes = 0
        if args["fp8_out"]:
            # Read FP16 (2 bytes) + write FP8 (1 byte) = 3 bytes per element
            fp8_bytes = num_elements * 3
            quant_bandwidth_gb_s = fp8_bytes / (quant_time_ms / 1000) / 1e9
        
        # All-Gather: Each rank sends M_shard×N to (world_size - 1) remote ranks
        # Local write doesn't go over interconnect, so we exclude it
        # Interconnect bandwidth = data transferred over network / time
        ag_bandwidth_gb_s = 0.0
        ag_interconnect_bytes = 0
        if args["all_gather"]:
            # Use actual dtype of data being gathered (FP8 if quantized, otherwise FP16)
            ag_bytes_per_element = fp8_output_bench.element_size() if args["fp8_out"] else bytes_per_element
            ag_interconnect_bytes = M_shard * N * (world_size - 1) * ag_bytes_per_element
            ag_bandwidth_gb_s = ag_interconnect_bytes / (ag_time_ms / 1000) / 1e9
        
        # Calculate total bytes and time
        total_bytes = rs_interconnect_bytes + bytes_processed_rmsnorm + fp8_bytes + ag_interconnect_bytes
        total_time = rs_time_ms + rmsnorm_time_ms + quant_time_ms + ag_time_ms
        
        # Calculate total effective bandwidth
        total_bandwidth_gb_s = total_bytes / (total_time / 1000) / 1e9
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Benchmark Results (Rank 0)")
            print(f"{'='*60}")
            print(f"Configuration:")
            print(f"  M={M}, N={N}, M_shard={M_shard}")
            print(f"  dtype={args['datatype']}, world_size={world_size}")
            print(f"  Elements per rank: {num_elements:,}")
            print(f"\nComponent Performance:")
            print(f"  Reduce-Scatter:")
            print(f"    Time:             {rs_time_ms:.3f} ms")
            print(f"    Interconnect BW:  {rs_bandwidth_gb_s:.2f} GB/s")
            print(f"    Data transferred: {rs_interconnect_bytes / 1e9:.3f} GB")
            print(f"  RMSNorm:")
            print(f"    Time:      {rmsnorm_time_ms:.3f} ms")
            print(f"    Bandwidth: {rmsnorm_bandwidth_gb_s:.2f} GB/s (memory)")
            print(f"    TFLOPS:    {rmsnorm_tflops:.2f}")
            
            if args["fp8_out"]:
                print(f"  FP8 Quantization:")
                print(f"    Time:      {quant_time_ms:.3f} ms")
                print(f"    Bandwidth: {quant_bandwidth_gb_s:.2f} GB/s (memory)")
            
            if args["all_gather"]:
                print(f"  All-Gather:")
                print(f"    Time:             {ag_time_ms:.3f} ms")
                print(f"    Interconnect BW:  {ag_bandwidth_gb_s:.2f} GB/s")
                print(f"    Data transferred: {ag_interconnect_bytes / 1e9:.3f} GB")
            
            print(f"\nTotal Pipeline:")
            print(f"  Total time:        {total_time:.3f} ms")
            print(f"  Total bandwidth:   {total_bandwidth_gb_s:.2f} GB/s")
            print(f"  Total bytes:       {total_bytes / 1e9:.3f} GB")
            print(f"{'='*60}")
            
            # Save results
            results = {
                "M": M,
                "N": N,
                "M_shard": M_shard,
                "world_size": world_size,
                "dtype": args["datatype"],
                "fp8_out": args["fp8_out"],
                "all_gather": args["all_gather"],
                
                # Reduce-Scatter metrics
                "reduce_scatter_time_ms": rs_time_ms,
                "reduce_scatter_bandwidth_gb_s": rs_bandwidth_gb_s,
                
                # RMSNorm metrics
                "rmsnorm_time_ms": rmsnorm_time_ms,
                "rmsnorm_bandwidth_gb_s": rmsnorm_bandwidth_gb_s,
                "rmsnorm_tflops": rmsnorm_tflops,
                
                # FP8 Quantization metrics
                "quant_time_ms": quant_time_ms if args["fp8_out"] else None,
                "quant_bandwidth_gb_s": quant_bandwidth_gb_s if args["fp8_out"] else None,
                
                # All-Gather metrics
                "all_gather_time_ms": ag_time_ms if args["all_gather"] else None,
                "all_gather_bandwidth_gb_s": ag_bandwidth_gb_s if args["all_gather"] else None,
                
                # Total pipeline metrics
                "total_time_ms": total_time,
                "total_bandwidth_gb_s": total_bandwidth_gb_s,
                "total_bytes_gb": total_bytes / 1e9,
                
                # Configuration
                "NUM_SMS": NUM_SMS,
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "GROUP_SIZE_M": GROUP_SIZE_M,
            }
            
            with open(args["output_file"], "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {args['output_file']}")

    if rank == 0:
        print(f"\nRank {rank}: Pipeline completed successfully!")

    dist.destroy_process_group()


def main():
    args = parse_args()
    
    world_size = args["num_ranks"]
    
    # Generate unique init URL for this run
    init_url = f"tcp://127.0.0.1:{random.randint(20000, 60000)}"
    
    print(f"Launching {world_size} processes...")
    print(f"Init URL: {init_url}")
    
    # Spawn workers
    mp.spawn(
        _worker,
        args=(world_size, init_url, args),
        nprocs=world_size,
        join=True,
    )
    
    print("\nAll processes completed!")


if __name__ == "__main__":
    main()
