#!/usr/bin/env python3
"""Occupancy sweep — test if more concurrent XGMI requests improve RS bandwidth.

At 57 GB/s we're at 13% of XGMI line rate. Hypothesis: not enough
concurrent loads in flight to saturate the link. More WGs with smaller
tiles = more outstanding XGMI requests = higher throughput.
"""

import os
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
def fast_rs_kernel(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
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
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            start_rank = pid % world_size
            acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (start_rank + i) % world_size
                acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
            out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=out_mask)


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

input_tensor = shmem.zeros((M, N), dtype=dtype)
input_tensor.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
heap_bases = shmem.get_heap_bases()
shmem.barrier()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# RCCL baseline
input_rccl = input_tensor.clone()
output_rccl = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(output_rccl, input_rccl, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Occupancy sweep: M={M}, N={N}, TP={world_size}")
    print(f"RCCL RS: {rccl_ms:.3f}ms")
    print()
    print(f"{'bm':>4} {'bn':>4} {'sms':>4} {'warps':>5} | {'ms':>7} {'GB/s':>7} {'vs_best':>8}")
    print("-" * 55)

# Wide sweep: tile sizes × SMS × warps
best_ms = 999.0
best_cfg = None

configs = []
for bm in [16, 32, 64, 128]:
    for bn in [32, 64]:
        for num_sms in [32, 64, 128, 196, 256, 304]:
            for warps in [2, 4, 8]:
                if M_local % bm != 0:
                    continue
                configs.append((bm, bn, num_sms, warps))

for bm, bn, num_sms, warps in configs:
    out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()

    try:
        for _ in range(warmup // 2):
            fast_rs_kernel[(num_sms,)](
                input_tensor, out, M, N, M_local,
                input_tensor.stride(0), input_tensor.stride(1),
                out.stride(0), out.stride(1),
                heap_bases, rank, world_size, bm, bn, num_sms,
                num_warps=warps,
            )
        torch.cuda.synchronize()
    except Exception:
        continue

    # Quick correctness
    ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    dist.reduce_scatter_tensor(ref, input_tensor.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    diff = torch.abs(out - ref).max().item()
    if diff > 1.0:
        continue

    s.record()
    for _ in range(iters):
        fast_rs_kernel[(num_sms,)](
            input_tensor, out, M, N, M_local,
            input_tensor.stride(0), input_tensor.stride(1),
            out.stride(0), out.stride(1),
            heap_bases, rank, world_size, bm, bn, num_sms,
            num_warps=warps,
        )
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    bw = M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9

    if ms < best_ms:
        best_ms = ms
        best_cfg = (bm, bn, num_sms, warps)

    if rank == 0:
        marker = " ***" if ms <= best_ms else ""
        print(f"{bm:4d} {bn:4d} {num_sms:4d} {warps:5d} | {ms:7.3f} {bw:7.1f} {ms/best_ms:8.2f}{marker}")

if rank == 0:
    print()
    print(f"RCCL RS:  {rccl_ms:.3f}ms")
    print(f"Best RS:  {best_ms:.3f}ms (bm={best_cfg[0]} bn={best_cfg[1]} sms={best_cfg[2]} w={best_cfg[3]})")
    print(f"Speedup:  {rccl_ms / best_ms:.2f}x")

shmem.barrier()
dist.destroy_process_group()
