#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc.

"""
Reduce-Scatter → RMSNorm → FP8 Quantization Pipeline

Task:
- Start with M×N tensor on each of 8 GPUs (same position, different values)
- Reduce (sum) pointwise across all GPUs
- Split along M dimension: Each GPU gets (M/8)×N piece
- RMSNorm along N dimension (locally, since we have full N)
- Quantize to FP8

Pipeline:
1. Reduce-Scatter along M dimension: 8 M×N → Each GPU gets (M/world_size)×N
2. RMSNorm on (M/world_size)×N with full N dimension
3. FP8 Quantization
4. (Optional) All-Gather along M dimension to reconstruct full M×N

Usage:
    # Run with torchrun for multi-GPU
    torchrun --nproc_per_node=8 reduce_scatter_rmsnorm_quant.py --verify

    # Or use the benchmark script which handles multi-process spawning
    python benchmark.py --num_rows 8192 --num_cols 7168 --num_ranks 8 --validate
"""

import os
import argparse

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris


@triton.jit
def reduce_scatter_m_kernel(
    input_ptr,  # Local input tensor in IRIS memory: *[M, N]
    output_ptr,  # Output shard in IRIS memory: *[M_shard, N]
    M,
    M_shard,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Reduce-scatter kernel along M dimension using pull-based approach with iris.load.

    Each rank computes its own output shard by:
    - Loading the relevant portion from all ranks (including itself)
    - Accumulating the sum locally
    - Storing the result

    For example, rank 0 computes output[0:M_shard, :] by:
    - Loading input[0:M_shard, :] from rank 0 (local)
    - Loading input[0:M_shard, :] from rank 1 (remote via iris.load)
    - ...
    - Loading input[0:M_shard, :] from rank 7 (remote via iris.load)
    - Summing all loaded data

    This kernel is called once per rank.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # Persistent loop over tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Swizzle pattern
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Local indices in this rank's output shard (M_shard × N)
        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Add compiler hints
        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

        # Masks
        mask_m_local = rm_local < M_shard
        mask_n = rn < N
        mask = mask_m_local[:, None] & mask_n[None, :]

        # Calculate which rows to read from each source rank's input
        # This rank (cur_rank) needs rows [cur_rank*M_shard : (cur_rank+1)*M_shard]
        # from ALL source ranks
        rm_global = cur_rank * M_shard + rm_local
        mask_m_global = rm_global < M
        load_mask = mask_m_global[:, None] & mask_n[None, :]

        # Accumulator for the sum across all ranks
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Pointers to the data we need from all ranks
        src_ptrs = input_ptr + rm_global[:, None] * stride_im + rn[None, :] * stride_in

        # Load from all source ranks and accumulate
        for src_rank in tl.static_range(world_size):
            data = iris.load(src_ptrs, cur_rank, src_rank, heap_bases, mask=load_mask)
            accumulator += data.to(tl.float32)

        # Store the result to output shard
        output_ptrs = output_ptr + rm_local[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(output_ptrs, accumulator.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def all_gather_m_kernel(
    shard_ptr,  # *[M_shard, N]
    out_ptr,  # *[M, N]
    M,
    M_shard,
    N,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    All-gather kernel along M dimension with 1D persistent-style PID mapping.
    Each rank sends its (M_shard)×N to all other ranks.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # Persistent loop over tiles
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Swizzle pattern
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Local indices
        rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        mask_m_local = rm_local < M_shard
        mask_n = rn < N

        # Load local shard
        shard_ptrs = shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn
        shard_data = tl.load(shard_ptrs, mask=mask_m_local[:, None] & mask_n[None, :], other=0.0)

        # Send to all ranks at the appropriate M offset
        for dst in range(world_size):
            # Calculate global M indices
            rm_global = cur_rank * M_shard + rm_local
            mask_m_global = rm_global < M

            if dst == cur_rank:
                # Local store
                out_ptrs = out_ptr + rm_global[:, None] * stride_om + rn[None, :] * stride_on
                tl.store(out_ptrs, shard_data, mask=mask_m_global[:, None] & mask_n[None, :])
            else:
                # Remote store using IRIS
                # iris.put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask)
                # from_ptr: local source, to_ptr: remote destination
                iris.put(
                    shard_ptr + rm_local[:, None] * stride_sm + rn[None, :] * stride_sn,  # from_ptr (local source)
                    out_ptr + rm_global[:, None] * stride_om + rn[None, :] * stride_on,  # to_ptr (remote dest)
                    cur_rank,
                    dst,
                    heap_bases,
                    mask=mask_m_global[:, None] & mask_n[None, :],
                )


@triton.jit
def aiter_rmsnorm(
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """RMSNorm kernel from AITer."""
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs, cache_modifier=".cg").to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs, cache_modifier=".cg").to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(
                g_ptrs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            rms_norm = row * norm_factor * g
            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def quantize_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    M,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FP8 quantization kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)

    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load input
    input_ptrs = input_ptr + rm[:, None] * stride_im + rn[None, :] * stride_in
    data = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load scale
    scale = tl.load(scale_ptr)

    # Quantize
    fp8_max = 448.0
    scaled = data / scale
    clamped = tl.clamp(scaled, -fp8_max, fp8_max)

    # Store
    output_ptrs = output_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(output_ptrs, clamped.to(output_ptr.type.element_ty), mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", "--m", type=int, default=8192, help="Number of rows (M)")
    parser.add_argument("--num_cols", "--n", type=int, default=7168, help="Number of columns (N)")
    parser.add_argument("--num_ranks", "--world_size", type=int, default=8, help="Number of ranks")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fp8_out", action="store_true", help="Enable FP8 quantization")
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument("--all_gather", action="store_true", help="All-gather at the end to reconstruct full M×N")
    parser.add_argument("--verify", action="store_true", help="Verify against PyTorch reference")
    args = parser.parse_args()

    M = args.num_rows
    N = args.num_cols
    world_size = args.num_ranks

    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    M_shard = M // world_size

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cur_rank = int(os.environ.get("RANK", "0"))
    actual_world_size = int(os.environ.get("WORLD_SIZE", str(world_size)))

    if actual_world_size != world_size:
        print(f"Warning: WORLD_SIZE ({actual_world_size}) != requested world_size ({world_size})")
        world_size = actual_world_size
        assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
        M_shard = M // world_size

    print(f"Rank {cur_rank}/{world_size}: M={M}, N={N}, M_shard={M_shard}")

    # ================================================================
    # Initialize PyTorch Distributed (required for IRIS)
    # ================================================================
    if not dist.is_initialized():
        # Set up distributed environment
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["RANK"] = str(cur_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend="gloo", rank=cur_rank, world_size=world_size)

    # ================================================================
    # Initialize IRIS for distributed communication
    # ================================================================
    heap_size = 1 << 28  # 256MB
    shmem = iris.iris(heap_size)

    # Get heap base addresses for all ranks
    heap_bases = shmem.get_heap_bases()

    # ================================================================
    # Create input: Each rank has M×N tensor (same position, different values)
    # Must be in IRIS shared memory for remote access via iris.load
    # ================================================================
    torch.manual_seed(42 + cur_rank)  # Different seed per rank for different values
    local_input_temp = torch.randn(M, N, device=device, dtype=dtype) * (cur_rank + 1)

    # Allocate in IRIS shared memory
    local_input = shmem.empty((M, N), dtype=dtype)
    local_input.copy_(local_input_temp)
    del local_input_temp

    print(f"Rank {cur_rank}: Input shape: {local_input.shape}")

    # Barrier to ensure all ranks have allocated their input tensors
    shmem.barrier()

    # Default parameters (can be overridden via tuning)
    BLOCK_M = 16
    BLOCK_N = 64
    GROUP_SIZE_M = 8
    # MI350
    NUM_SMS = 256

    # ================================================================
    # Step 1: Reduce-Scatter along M dimension
    # Sum all M×N tensors and each rank gets (M/world_size)×N piece
    # ================================================================
    print(f"Rank {cur_rank}: Step 1 - Reduce-Scatter along M dimension")

    # Allocate output buffer in IRIS shared memory (must be accessible to all ranks)
    reduced_shard = shmem.zeros((M_shard, N), dtype=dtype)

    grid_rs = (NUM_SMS,)

    # Call kernel once - it will use iris.load() to pull data from all source ranks
    reduce_scatter_m_kernel[grid_rs](
        local_input,
        reduced_shard,
        M,
        M_shard,
        N,
        local_input.stride(0),
        local_input.stride(1),
        reduced_shard.stride(0),
        reduced_shard.stride(1),
        cur_rank,
        world_size,
        heap_bases,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        num_warps=16,  # Tuned for better performance
        num_stages=4,
        waves_per_eu=4,
    )

    # Synchronize to ensure all ranks have completed their loads and reductions
    torch.cuda.synchronize()
    shmem.barrier()

    print(f"Rank {cur_rank}: Reduce-scatter complete, shard shape: {reduced_shard.shape}")

    # ================================================================
    # Step 2: RMSNorm on (M_shard)×N with FULL N dimension
    # ================================================================
    print(f"Rank {cur_rank}: Step 2 - RMSNorm on (M_shard)×N")

    gamma = torch.ones(N, device=device, dtype=dtype)
    rmsnorm_output = torch.empty_like(reduced_shard)
    rsigma = torch.empty(M_shard, device=device, dtype=dtype)

    # AITer RMSNorm configuration
    # Note: Tuning found BLOCK_SIZE=1024 optimal for N=7168 (avoid VGPR spills with larger sizes)
    BLOCK_SIZE = 1024
    USE_BLOCKED = False  # Tuned: non-blocked mode is faster for moderate N
    NUM_PRGMS = M_shard  # Full parallelism: each program processes one row

    aiter_rmsnorm[(M_shard,)](
        reduced_shard,
        rmsnorm_output,
        gamma,
        rsigma,
        reduced_shard.stride(0),
        rmsnorm_output.stride(0),
        M_shard,
        N,
        args.eps,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        NUM_PRGMS=NUM_PRGMS,
        num_warps=8,  # Tuned for better occupancy
        waves_per_eu=2,
    )

    print(f"Rank {cur_rank}: RMSNorm complete, output shape: {rmsnorm_output.shape}")

    # ================================================================
    # Step 3: FP8 Quantization
    # ================================================================
    if args.fp8_out:
        print(f"Rank {cur_rank}: Step 3 - FP8 Quantization")

        # Compute scale
        max_val = rmsnorm_output.abs().max()
        scale = (max_val / 448.0).clamp(min=1e-8)
        scale_tensor = torch.tensor([scale], device=device, dtype=torch.float32)

        # Quantize
        if hasattr(torch, "float8_e4m3fn"):
            quantized_output = torch.empty_like(rmsnorm_output, dtype=torch.float8_e4m3fn)
        else:
            quantized_output = torch.empty_like(rmsnorm_output)

        # FP8 quantization uses medium tile sizes
        FP8_BLOCK_M = 64
        FP8_BLOCK_N = 64
        grid_quant = (triton.cdiv(M_shard, FP8_BLOCK_M), triton.cdiv(N, FP8_BLOCK_N))

        quantize_fp8_kernel[grid_quant](
            rmsnorm_output,
            quantized_output,
            scale_tensor,
            M_shard,
            N,
            rmsnorm_output.stride(0),
            rmsnorm_output.stride(1),
            quantized_output.stride(0),
            quantized_output.stride(1),
            BLOCK_M=FP8_BLOCK_M,
            BLOCK_N=FP8_BLOCK_N,
            num_warps=4,
            num_stages=2,
            waves_per_eu=2,
        )

        final_shard = quantized_output
        print(
            f"Rank {cur_rank}: Quantization complete, shape: {quantized_output.shape}, dtype: {quantized_output.dtype}"
        )
    else:
        final_shard = rmsnorm_output
        print(f"Rank {cur_rank}: No quantization, final shard shape: {final_shard.shape}")

    # ================================================================
    # Step 4 (Optional): All-Gather along M dimension
    # ================================================================
    if args.all_gather:
        print(f"Rank {cur_rank}: Step 4 - All-Gather along M dimension")

        # Determine output dtype
        if args.fp8_out and hasattr(torch, "float8_e4m3fn"):
            out_dtype = torch.float8_e4m3fn
        else:
            out_dtype = dtype

        # Allocate output in IRIS shared memory
        full_output = shmem.zeros((M, N), dtype=out_dtype)

        grid_ag = (NUM_SMS,)

        # All-gather uses similar parameters to reduce-scatter
        AG_BLOCK_M = 64
        AG_BLOCK_N = 64

        all_gather_m_kernel[grid_ag](
            final_shard,
            full_output,
            M,
            M_shard,
            N,
            final_shard.stride(0),
            final_shard.stride(1),
            full_output.stride(0),
            full_output.stride(1),
            cur_rank,
            world_size,
            heap_bases,
            BLOCK_M=AG_BLOCK_M,
            BLOCK_N=AG_BLOCK_N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_SMS=NUM_SMS,
            num_warps=8,
            num_stages=3,
            waves_per_eu=2,
        )

        # Synchronize to ensure all ranks have completed their puts
        torch.cuda.synchronize()

        print(f"Rank {cur_rank}: All-gather complete, full output shape: {full_output.shape}")
        result = full_output
    else:
        result = final_shard
        print(f"Rank {cur_rank}: Skipping all-gather, result shape: {result.shape}")

    # ================================================================
    # Verification
    # ================================================================
    if args.verify and cur_rank == 0:
        print("\n" + "=" * 60)
        print("Verification against PyTorch reference")
        print("=" * 60)

        import torch.nn as nn

        # Reference computation
        torch.manual_seed(42)
        ref_tensors = []
        for i in range(world_size):
            torch.manual_seed(42 + i)
            tensor = torch.randn(M, N, device=device, dtype=dtype) * (i + 1)
            ref_tensors.append(tensor)

        # Pointwise reduce (sum)
        ref_reduced = torch.zeros(M, N, device=device, dtype=dtype)
        for tensor in ref_tensors:
            ref_reduced += tensor

        print(f"Reference reduced sum: {ref_reduced.sum(dtype=torch.float32):.4f}")

        # Extract this rank's shard
        start_row = cur_rank * M_shard
        end_row = (cur_rank + 1) * M_shard
        ref_shard = ref_reduced[start_row:end_row, :]

        # Compare reduce-scatter result
        rs_diff = torch.abs(ref_shard - reduced_shard)
        print(f"Reduce-scatter max diff: {rs_diff.max().item():.8f}")

        if rs_diff.max().item() < 1e-5:
            print("✅ Reduce-scatter verification PASSED")
        else:
            print("❌ Reduce-scatter verification FAILED")

        # RMSNorm
        rmsnorm_layer = nn.RMSNorm(N, eps=args.eps, device=device, dtype=dtype)
        ref_normed = rmsnorm_layer(ref_shard)

        print(f"\nReference RMSNorm sum: {ref_normed.sum(dtype=torch.float32):.4f}")
        print(f"Triton RMSNorm sum: {rmsnorm_output.sum(dtype=torch.float32):.4f}")

        rms_diff = torch.abs(ref_normed - rmsnorm_output)
        print(f"RMSNorm max diff: {rms_diff.max().item():.8f}")
        print(f"RMSNorm mean diff: {rms_diff.mean().item():.8f}")

        if rms_diff.max().item() < 1e-2:
            print("✅ RMSNorm verification PASSED")
        else:
            print("❌ RMSNorm verification FAILED")

    print(f"\nRank {cur_rank}: Pipeline completed successfully!")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
