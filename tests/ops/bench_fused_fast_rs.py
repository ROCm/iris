#!/usr/bin/env python3
"""Fused GEMM + fast iris RS — single kernel, WG specialization, one-shot pull."""

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
def fused_gemm_fast_rs_kernel(
    A, B,
    C_staged,
    C_out,
    locks,
    M, N, K_local, M_local,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_sc_m, stride_sc_n,
    stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GEMM_SMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Fused GEMM + RS in single kernel.
    GEMM WGs: compute partial C, store to C_staged (symmetric), set lock.
    RS WGs: wait for lock, one-shot pull from all peers via iris.load, store to C_out.
    """
    pid = tl.program_id(0)
    acc_dtype = tl.float32

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    M_per_rank = M // world_size
    num_local_m_tiles = M_per_rank // BLOCK_SIZE_M
    total_local_tiles = num_local_m_tiles * num_pid_n

    if pid < GEMM_SMS:
        # GEMM phase — persistent, compute all tiles
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            rk = tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

            loop_k = tl.cdiv(K_local, BLOCK_SIZE_K)
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
                rk2 = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                A_LAST = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
                B_LAST = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(A_LAST, mask=rk2[None, :] < K_local, other=0.0)
                b = tl.load(B_LAST, mask=rk2[:, None] < K_local, other=0.0)
                acc += tl.dot(a, b)

            c = acc.to(C_staged.type.element_ty)
            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            tl.store(C_staged + sc_offset, c, cache_modifier=".wt")

            # Signal tile done
            tl.debug_barrier()
            iris.atomic_cas(locks + tile_id, 0, 1, cur_rank, cur_rank, heap_bases,
                           sem="release", scope="sys")

    else:
        # RS phase — persistent, one-shot pull for owned tiles
        COMM_SMS = NUM_SMS - GEMM_SMS
        comm_pid = pid - GEMM_SMS

        m_offset = cur_rank * num_local_m_tiles

        for tile_id in range(comm_pid, total_local_tiles, COMM_SMS):
            local_pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            global_pid_m = m_offset + local_pid_m

            global_tile_id = global_pid_m * num_pid_n + pid_n

            # Wait for ALL ranks' GEMM to finish this tile
            for peer in tl.static_range(world_size):
                done = 0
                while done == 0:
                    done = iris.atomic_cas(
                        locks + global_tile_id, 1, 1,
                        cur_rank, peer, heap_bases,
                        sem="acquire", scope="sys",
                    )

            # One-shot pull from all peers
            rm = global_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

            sc_offset = rm[:, None] * stride_sc_m + rn[None, :] * stride_sc_n
            base_ptr = C_staged + sc_offset

            is_full = (global_pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M <= M) & (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N)

            if is_full:
                start_rank = comm_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    r = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, r, heap_bases, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_rm = tl.max_contiguous(tl.multiple_of(out_rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                out_ptrs = C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
                tl.store(out_ptrs, acc.to(C_out.type.element_ty))
            else:
                mask = (rm[:, None] < M) & (rn[None, :] < N)
                start_rank = comm_pid % world_size
                acc = iris.load(base_ptr, cur_rank, start_rank, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)
                for i in tl.static_range(1, world_size):
                    r = (start_rank + i) % world_size
                    acc += iris.load(base_ptr, cur_rank, r, heap_bases, mask=mask, hint=(1, BLOCK_SIZE_N)).to(acc_dtype)

                out_rm = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                out_mask = (out_rm[:, None] < M_local) & (rn[None, :] < N)
                out_ptrs = C_out + out_rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
                tl.store(out_ptrs, acc.to(C_out.type.element_ty), mask=out_mask)


# --- Setup ---
M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 500

# Per-TP configs
TP_CONFIGS = {
    2: dict(bm=128, bn=64, gemm_sms=196, num_sms=304),
    4: dict(bm=64, bn=64, gemm_sms=196, num_sms=304),
    8: dict(bm=128, bn=64, gemm_sms=256, num_sms=304),
}
cfg = TP_CONFIGS.get(world_size, dict(bm=128, bn=64, gemm_sms=196, num_sms=304))

A = torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}")
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}")

num_m_tiles = M // cfg['bm']
num_n_tiles = (N + cfg['bn'] - 1) // cfg['bn']
total_tiles = num_m_tiles * num_n_tiles

C_staged = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
locks = shmem.zeros((total_tiles,), dtype=torch.int32)
heap_bases = shmem.get_heap_bases()

if rank == 0:
    print(f"Fused GEMM+fast RS: M={M}, N={N}, K={K_global}, TP={world_size}")
    print(f"Config: bm={cfg['bm']}, bn={cfg['bn']}, gemm_sms={cfg['gemm_sms']}, num_sms={cfg['num_sms']}")

# Correctness
shmem.barrier()
locks.zero_()
fused_gemm_fast_rs_kernel[(cfg['num_sms'],)](
    A, B, C_staged, C_out, locks,
    M, N, K_local, M_local,
    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
    C_staged.stride(0), C_staged.stride(1),
    C_out.stride(0), C_out.stride(1),
    heap_bases, rank, world_size,
    cfg['bm'], cfg['bn'], 64, 4,
    cfg['gemm_sms'], cfg['num_sms'],
    K_local % 64 == 0,
    num_warps=8, num_stages=2,
)
torch.cuda.synchronize()

ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
C_ref = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
torch.mm(A, B, out=C_ref)
dist.reduce_scatter_tensor(ref, C_ref, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

max_diff = torch.abs(C_out - ref).max().item()
if rank == 0:
    print(f"Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 1.0 else 'FAIL'}")

if max_diff > 1.0:
    shmem.barrier()
    dist.destroy_process_group()
    exit(1)

# Benchmark — sweep gemm_sms ratio
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

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
rccl_e2e = s.elapsed_time(e) / iters

# Fused — sweep GEMM_SMS ratio
if rank == 0:
    print(f"\nRCCL baseline: {rccl_e2e:.3f}ms")
    print("Fused GEMM+fast RS sweep:")

for gemm_sms in [128, 196, 240, 256, 280]:
    comm_sms = cfg['num_sms'] - gemm_sms
    if comm_sms < 8:
        continue

    shmem.barrier()
    for _ in range(warmup):
        locks.zero_()
        fused_gemm_fast_rs_kernel[(cfg['num_sms'],)](
            A, B, C_staged, C_out, locks,
            M, N, K_local, M_local,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C_staged.stride(0), C_staged.stride(1),
            C_out.stride(0), C_out.stride(1),
            heap_bases, rank, world_size,
            cfg['bm'], cfg['bn'], 64, 4,
            gemm_sms, cfg['num_sms'],
            K_local % 64 == 0,
            num_warps=8, num_stages=2,
        )
    torch.cuda.synchronize()

    s.record()
    for _ in range(iters):
        locks.zero_()
        fused_gemm_fast_rs_kernel[(cfg['num_sms'],)](
            A, B, C_staged, C_out, locks,
            M, N, K_local, M_local,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C_staged.stride(0), C_staged.stride(1),
            C_out.stride(0), C_out.stride(1),
            heap_bases, rank, world_size,
            cfg['bm'], cfg['bn'], 64, 4,
            gemm_sms, cfg['num_sms'],
            K_local % 64 == 0,
            num_warps=8, num_stages=2,
        )
    e.record()
    torch.cuda.synchronize()

    fused_ms = s.elapsed_time(e) / iters
    speedup = rccl_e2e / fused_ms
    if rank == 0:
        print(f"  gemm_sms={gemm_sms:3d} comm_sms={comm_sms:3d}: {fused_ms:.3f}ms ({speedup:.2f}x)")

# Also benchmark torch.mm + fast iris RS (unfused, no barrier)
shmem.barrier()
for _ in range(warmup):
    torch.mm(A, B, out=C_staged)
    from tests.ops.bench_fast_rs import fast_reduce_scatter_kernel
torch.cuda.synchronize()

if rank == 0:
    print()

shmem.barrier()
dist.destroy_process_group()
