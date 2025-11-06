# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc.

import os
import argparse

import torch
import triton
import triton.language as tl

import iris  # type: ignore


# Inline AITer RMSNorm kernel (forward only)
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
                x = tl.load(input_ptrs).to(tl.float32)
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
                x = tl.load(input_ptrs).to(tl.float32)
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
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
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

@triton.jit()
def persistent_gemm_all_scatter(
    A,
    B,
    C,
    c_global,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cm_global,
    stride_cn_global,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        # Accumulator registers with C results
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # Add compiler hints
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how the N-dimension is being multiplied by current rank.
        # This is because each rank is computing a portion of the N-dimension
        # locally and then scattering it to all other ranks to complete
        # the global N-dimension.
        global_offset = rm[:, None] * stride_cm_global + (rn[None, :] + cur_rank * N) * stride_cn_global

        # Store data to the global result using puts
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                # For the current rank, we can use store
                tl.store(c_global + global_offset, c, mask=sub_mask)
            else:
                iris.store(
                    c_global + global_offset,
                    c,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask,
                )

gemm_kernel = persistent_gemm_all_scatter

##@triton.jit
##def gemm_all_scatter(
##    A,  # input: *[M, K_shard]
##    B,  # weight shard: *[K_shard, N]
##    C_local,  # local partial result: *[M, N]
##    C_global,  # distributed result buffer: *[M, N]
##    M,
##    K_shard,
##    N,
##    stride_am,
##    stride_ak,
##    stride_bk,
##    stride_bn,
##    stride_clm,
##    stride_cln,
##    stride_cgm,
##    stride_cgn,
##    cur_rank: tl.constexpr,
##    world_size: tl.constexpr,
##    heap_bases: tl.tensor,
##    BLOCK_M: tl.constexpr,
##    BLOCK_N: tl.constexpr,
##    BLOCK_K: tl.constexpr,
##):
##    pid_m = tl.program_id(0)
##    pid_n = tl.program_id(1)
##
##    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
##    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
##    rk = tl.arange(0, BLOCK_K)
##
##    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
##    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
##    rk = tl.max_contiguous(tl.multiple_of(rk, BLOCK_K), BLOCK_K)
##
##    mask_m = rm < M
##    mask_n = rn < N
##    mask_k = rk < K_shard
##
##    # Initialize accumulator
##    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
##
##    # GEMM computation
##    for k in range(0, tl.cdiv(K_shard, BLOCK_K)):
##        # Load A block
##        a_ptr = A + rm[:, None] * stride_am + (k * BLOCK_K + rk[None, :]) * stride_ak
##        a_mask = mask_m[:, None] & mask_k[None, :]
##        a = tl.load(a_ptr, mask=a_mask, other=0.0)
##
##        # Load B block
##        b_ptr = B + (k * BLOCK_K + rk[:, None]) * stride_bk + rn[None, :] * stride_bn
##        b_mask = mask_k[:, None] & mask_n[None, :]
##        b = tl.load(b_ptr, mask=b_mask, other=0.0)
##
##        # Accumulate
##        acc += tl.dot(a, b)
##
##    # Convert accumulator to output dtype
##    c = acc.to(C_local.type.element_ty)
##
##    # Store local partial result
##    c_local_ptr = C_local + rm[:, None] * stride_clm + rn[None, :] * stride_cln
##    tl.store(c_local_ptr, c, mask=mask_m[:, None] & mask_n[None, :])
##
##    # All-scatter: distribute partial result to all ranks
##    for dst_rank in range(world_size):
##        if dst_rank == cur_rank:
##            # Local copy
##            c_global_ptr = C_global + rm[:, None] * stride_cgm + rn[None, :] * stride_cgn
##            tl.store(c_global_ptr, c, mask=mask_m[:, None] & mask_n[None, :])
##        else:
##            # Remote scatter using IRIS
##            iris.store(
##                C_global + rm[:, None] * stride_cgm + rn[None, :] * stride_cgn,
##                c,
##                cur_rank,
##                dst_rank,
##                heap_bases,
##                mask=mask_m[:, None] & mask_n[None, :],
##            )
##

@triton.jit
def all_gather_push(
    shard_ptr,  # *[M, N_shard]
    out_ptr,  # *[M, N_total]
    M,
    N_total,
    N_shard,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    heap_bases: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    mask_m = rm < M

    # Send our local shard to each destination's global slot
    for dst in range(world_size):
        start = cur_rank * N_shard
        rn_dst = start + rn
        mask_n_dst = rn_dst < N_total
        iris.put(
            out_ptr + rm[:, None] * stride_om + rn_dst[None, :] * stride_on,
            shard_ptr + rm[:, None] * stride_sm + rn[None, :] * stride_sn,
            cur_rank,
            dst,
            heap_bases,
            mask=mask_m[:, None] & mask_n_dst[None, :],
        )


def maybe_quantize_fp8(x: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable:
        return x
    if hasattr(torch, "float8_e4m3fn") and x.is_cuda:
        return x.to(torch.float8_e4m3fn)
    # Simple fallback: dequantize-style emulation (returns original dtype)
    scale = x.abs().max().clamp(min=1e-8) / 448.0
    q = torch.clamp((x / scale).round_(), -448, 447).to(torch.int16)
    return (q.to(torch.float16) * scale.to(torch.float16)).to(x.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096, help="Input dimension")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fp8_out", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--all_gather", action="store_true", help="Enable all-gather at the end")
    args = parser.parse_args()

    M, K, N, TP = args.m, args.k, args.n, args.tp
    assert K % TP == 0, "K must be divisible by TP"
    K_shard = K // TP

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Set device based on LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cur_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(TP)))
    assert world_size == TP, "WORLD_SIZE should equal TP for this prototype"

    print(f"Rank {cur_rank}: M={M}, K={K}, N={N}, K_shard={K_shard}, TP={TP}")

    # Phase 1: Create input tensor (sharded along K dimension)
    x_input = torch.randn(M, K_shard, device=device, dtype=dtype)  # [M, K/TP]

    # Create weight shard
    weight_shard = torch.randn(K_shard, N, device=device, dtype=dtype)  # [K/TP, N]

    # IRIS heap bases placeholder tensor
    heap_bases = torch.empty(1, device=device, dtype=torch.int64)

    # Phase 2: GEMM + All-Scatter (no atomic operations)
    # Local partial result buffer
    partial_result = torch.empty(M, N, device=device, dtype=dtype)

    # Distributed result buffer (each rank will have the complete [M, N] result)
    distributed_result = torch.empty(M, N, device=device, dtype=dtype)

    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64
    grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    num_xcds = 8
    num_sms = 256

    # TODO: Use arch-specific values.
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfma = 16
    kpack = 1

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    iters_per_tile = triton.cdiv(K, BLK_K)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0


##    gemm_all_scatter[grid_gemm](
##        x_input,  # [M, K_shard]
##        weight_shard,  # [K_shard, N]
##        partial_result,  # [M, N] - local partial
##        distributed_result,  # [M, N] - distributed result
##        M,
##        K_shard,
##        N,
##        x_input.stride(0),
##        x_input.stride(1),
##        weight_shard.stride(0),
##        weight_shard.stride(1),
##        partial_result.stride(0),
##        partial_result.stride(1),
##        distributed_result.stride(0),
##        distributed_result.stride(1),
##        cur_rank,
##        world_size,
##        heap_bases,
##        BLOCK_M=BLOCK_M,
##        BLOCK_N=BLOCK_N,
##        BLOCK_K=BLOCK_K,
##        num_warps=8,
##    )

    kk = gemm_kernel[(num_sms,)](
        x_input,  # [M, K_shard]
        weight_shard,  # [K_shard, N]
        partial_result,  # [M, N] - local partial
        distributed_result,  # [M, N] - distributed result
        M,
        N,
        K_shard,
        x_input.stride(0),
        x_input.stride(1),
        weight_shard.stride(0),
        weight_shard.stride(1),
        partial_result.stride(0),
        partial_result.stride(1),
        distributed_result.stride(0),
        distributed_result.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfma,
        kpack=kpack,
        heap_bases=heap_bases_ptr,
        cur_rank=rank,
        world_size=world_size,
    )

    # Phase 3: RMSNorm (operates on complete [M, N] tensor)
    gamma = torch.ones(N, device=device, dtype=dtype)
    rmsnorm_output = torch.empty_like(distributed_result)
    rsigma = torch.empty(M, device=device, dtype=dtype)

    BLOCK = 128
    USE_BLOCKED = False
    NUM_PRGMS = 1
    aiter_rmsnorm[(M,)](
        distributed_result,
        rmsnorm_output,
        gamma,
        rsigma,
        distributed_result.stride(0),
        rmsnorm_output.stride(0),
        M,
        N,
        args.eps,
        BLOCK_SIZE=BLOCK,
        USE_BLOCKED=USE_BLOCKED,
        NUM_PRGMS=NUM_PRGMS,
        num_warps=4,
    )

    # Phase 4: Optional FP8 quantization
    rmsnorm_output_q = maybe_quantize_fp8(rmsnorm_output, enable=args.fp8_out)

    # Phase 5: Conditional All-Gather (only if needed)
    if args.all_gather:
        # All-gather to ensure all ranks have the complete result
        out_dtype = (
            torch.float8_e4m3fn if (args.fp8_out and hasattr(torch, "float8_e4m3fn")) else rmsnorm_output_q.dtype
        )
        final_output = torch.empty(M, N, device=device, dtype=out_dtype)
        grid_ag = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        all_gather_push[grid_ag](
            rmsnorm_output_q,
            final_output,
            M,
            N,
            N,  # Note: N_shard = N since we're all-gathering the complete result
            rmsnorm_output_q.stride(0),
            rmsnorm_output_q.stride(1),
            final_output.stride(0),
            final_output.stride(1),
            cur_rank,
            world_size,
            heap_bases,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        result = final_output
        print(f"Rank {cur_rank}: All-gather enabled - complete result shape: {result.shape}, dtype: {result.dtype}")
    else:
        # Return the distributed result
        result = rmsnorm_output_q
        print(f"Rank {cur_rank}: No all-gather - distributed result shape: {result.shape}, dtype: {result.dtype}")

    print(f"Rank {cur_rank}: Hybrid approach completed successfully!")


if __name__ == "__main__":
    main()
