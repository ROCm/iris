#!/usr/bin/env python3
"""Full benchmark: all GPT-OSS-120B shapes × all TP levels."""

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


def find_best_config(M_local, world_size):
    """Per-shape auto-config."""
    configs = []
    for bm in [4, 8, 16, 32, 64, 128]:
        if M_local % bm != 0:
            continue
        if bm > M_local:
            continue
        for bn in [32, 64]:
            for sms in [16, 32, 64, 128, 196]:
                configs.append((bm, bn, sms))
    return configs


N = 2880
K_global = 4096
dtype = torch.float16
warmup, iters = 100, 500

heap_bases = shmem.get_heap_bases()

M_values = [32, 896, 2048]

if rank == 0:
    print(f"GPT-OSS-120B full sweep: N={N}, K={K_global}, TP={world_size}, FP16, MI355X")
    print(f"{'M':>6} {'M_local':>7} | {'RCCL':>8} {'iris':>8} {'speedup':>8} | {'config':>20}")
    print("-" * 75)

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

for M in M_values:
    K_local = K_global // world_size
    M_local = M // world_size

    if M_local < 4:
        if rank == 0:
            print(f"{M:6d} {M_local:7d} | SKIP (M_local too small)")
        continue

    A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
    B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

    # RCCL baseline
    C_rccl = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
    C_rccl_out = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")

    for _ in range(warmup):
        torch.mm(A, B, out=C_rccl)
        dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        torch.mm(A, B, out=C_rccl)
        dist.reduce_scatter_tensor(C_rccl_out, C_rccl, op=dist.ReduceOp.SUM)
    e.record()
    torch.cuda.synchronize()
    rccl_ms = s.elapsed_time(e) / iters

    # Fast iris RS sweep
    C_sym = shmem.zeros((M, N), dtype=dtype)
    C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
    shmem.barrier()

    configs = find_best_config(M_local, world_size)
    best_ms = 999.0
    best_cfg = None

    for bm, bn, num_sms in configs:
        out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

        try:
            # Warmup
            for _ in range(warmup // 4):
                torch.mm(A, B, out=C_sym)
                fast_rs_kernel[(num_sms,)](
                    C_sym, out, M, N, M_local,
                    C_sym.stride(0), C_sym.stride(1), out.stride(0), out.stride(1),
                    heap_bases, rank, world_size, bm, bn, num_sms,
                )
            torch.cuda.synchronize()

            # Quick correctness
            ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
            torch.mm(A, B, out=C_rccl)
            dist.reduce_scatter_tensor(ref, C_rccl, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            diff = torch.abs(out - ref).max().item()
            if diff > 2.0:
                continue

            s.record()
            for _ in range(iters):
                torch.mm(A, B, out=C_sym)
                fast_rs_kernel[(num_sms,)](
                    C_sym, out, M, N, M_local,
                    C_sym.stride(0), C_sym.stride(1), out.stride(0), out.stride(1),
                    heap_bases, rank, world_size, bm, bn, num_sms,
                )
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / iters

            if ms < best_ms:
                best_ms = ms
                best_cfg = (bm, bn, num_sms)
        except Exception:
            continue

    if rank == 0:
        if best_cfg:
            speedup = rccl_ms / best_ms
            cfg_str = f"bm={best_cfg[0]} bn={best_cfg[1]} sms={best_cfg[2]}"
            print(f"{M:6d} {M_local:7d} | {rccl_ms:7.3f}ms {best_ms:7.3f}ms {speedup:7.2f}x | {cfg_str}")
        else:
            print(f"{M:6d} {M_local:7d} | {rccl_ms:7.3f}ms    FAIL")

    del C_sym
    shmem.barrier()

shmem.barrier()
dist.destroy_process_group()
