#!/usr/bin/env python3
"""Minimal custom Triton RS kernel — no iris.ccl overhead."""

import os
import time
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import iris

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

heap_size = 2**33
shmem = iris.iris(heap_size)


@triton.jit
def fast_reduce_scatter_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    M_local,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Minimal RS: persistent grid, one-shot pull, no flags, no workspace."""
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    num_m_tiles = M_local // BLOCK_SIZE_M
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles

    m_offset = cur_rank * num_m_tiles

    for tile_id in range(pid, total_tiles, NUM_SMS):
        local_pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

        global_pid_m = m_offset + local_pid_m

        rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        in_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        base_ptr = input_ptr + in_offset

        is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

        if is_full:
            # Rotate start rank to distribute XGMI load
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            out_ptrs = output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
            tl.store(out_ptrs, acc.to(output_ptr.type.element_ty), mask=out_mask)


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 50, 200

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
output_tensor = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

heap_bases = shmem.get_heap_bases()
shmem.barrier()

NUM_SMS = 304
BM, BN = 128, 64

if rank == 0:
    print(f"Fast RS: M={M}, N={N}, TP={world_size}, bm={BM}, bn={BN}")
    print(f"Grid: {NUM_SMS} SMS, tiles: {M_local // BM * ((N + BN - 1) // BN)}")

# Warmup
for _ in range(warmup):
    fast_reduce_scatter_kernel[(NUM_SMS,)](
        input_tensor, output_tensor,
        M, N, M_local,
        input_tensor.stride(0), input_tensor.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        heap_bases, rank, world_size,
        BM, BN, NUM_SMS,
    )
torch.cuda.synchronize()

# Correctness
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

max_diff = torch.abs(output_tensor - ref).max().item()
if rank == 0:
    print(f"Correctness: max_diff = {max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")

if max_diff > 1.0:
    if rank == 0:
        print(f"output[0:4,0:4] = {output_tensor[0:4, 0:4]}")
        print(f"ref[0:4,0:4] = {ref[0:4, 0:4]}")
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

# Benchmark — RCCL
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

start_evt.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
end_evt.record()
torch.cuda.synchronize()
rccl_ms = start_evt.elapsed_time(end_evt) / iters

# Benchmark — fast RS (sweep tile sizes)
configs = [(64, 64), (128, 64), (128, 128), (256, 64), (256, 128)]
best_ms = 999.0
best_cfg = None

for bm, bn in configs:
    if M_local % bm != 0:
        continue
    out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

    for _ in range(warmup):
        fast_reduce_scatter_kernel[(NUM_SMS,)](
            input_tensor, out,
            M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size,
            bm, bn, NUM_SMS,
        )
    torch.cuda.synchronize()

    start_evt.record()
    for _ in range(iters):
        fast_reduce_scatter_kernel[(NUM_SMS,)](
            input_tensor, out,
            M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size,
            bm, bn, NUM_SMS,
        )
    end_evt.record()
    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt) / iters
    if rank == 0:
        bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9
        print(f"  bm={bm:3d} bn={bn:3d}: {ms:.3f}ms  ({bw:.1f} GB/s)")
    if ms < best_ms:
        best_ms = ms
        best_cfg = (bm, bn)

if rank == 0:
    print()
    print(f"RCCL RS:       {rccl_ms:.3f}ms")
    print(f"Fast iris RS:  {best_ms:.3f}ms  (best: bm={best_cfg[0]} bn={best_cfg[1]})")
    print(f"Speedup:       {rccl_ms / best_ms:.2f}x {'(faster)' if best_ms < rccl_ms else '(slower)'}")
    print()
    gemm_ms = 0.037
    print(f"Projected torch.mm + RCCL RS:  {gemm_ms + rccl_ms:.3f}ms")
    print(f"Projected torch.mm + fast RS:  {gemm_ms + best_ms:.3f}ms")

shmem.barrier()
dist.destroy_process_group()
