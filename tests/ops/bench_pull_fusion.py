#!/usr/bin/env python3
"""Pull-direction fusion: overlap WITHOUT forcing a push.

Every fused variant lost because fusion forces a push (producer sends
to owner) and push is 3.2x slower than pull on XGMI.

But there is a fusion shape that preserves the pull direction:

  Order the GEMM so every rank computes the M-chunks in the SAME order:
  chunk 0, chunk 1, ... chunk ws-1.

  After ALL ranks finish chunk c, the rank that OWNS chunk c can pull it
  from every peer -- while all ranks are still computing chunk c+1.

  Rank r's comm work is only for chunk r, so it pulls once, at the point
  where chunk r is globally complete.

This gets overlap with pull-direction comm. Uses two streams so the
comm kernel runs concurrently with the next GEMM chunk.

The sync is a per-chunk arrival counter (ws increments), polled locally.
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
shmem = iris.iris(2**33)

from iris.ops.reduce_scatter_auto import _one_shot_rs_kernel, _get_config


@triton.jit
def _chunk_gemm(
    A, B, C,
    M, N, K,
    m_start, m_rows,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GM: tl.constexpr, NUM_SMS: tl.constexpr, EVEN_K: tl.constexpr,
):
    """GEMM restricted to rows [m_start, m_start + m_rows)."""
    pid = tl.program_id(0)
    npm = tl.cdiv(m_rows, BM)
    npn = tl.cdiv(N, BN)
    total = npm * npn

    for tid in range(pid, total, NUM_SMS):
        ngroup = GM * npn
        gid = tid // ngroup
        first_m = gid * GM
        gsz = min(npm - first_m, GM)
        pm = first_m + ((tid % ngroup) % gsz)
        pn = (tid % ngroup) // gsz

        rm = m_start + pm * BM + tl.arange(0, BM)
        rn = (pn * BN + tl.arange(0, BN)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = tl.max_contiguous(tl.multiple_of(rn, BN), BN)

        rk = tl.arange(0, BK)
        AB = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        BB = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        lk = tl.cdiv(K, BK)
        if not EVEN_K:
            lk -= 1
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(lk):
            a = tl.load(tl.multiple_of(AB, (1, 16)))
            b = tl.load(tl.multiple_of(BB, (16, 1)))
            acc += tl.dot(a, b)
            AB += BK * stride_ak
            BB += BK * stride_bk
        if not EVEN_K:
            rk2 = lk * BK + tl.arange(0, BK)
            AL = A + rm[:, None] * stride_am + rk2[None, :] * stride_ak
            BL = B + rk2[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(AL, mask=rk2[None, :] < K, other=0.0)
            b = tl.load(BL, mask=rk2[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        msk = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
                 acc.to(C.type.element_ty), mask=msk, cache_modifier=".wt")


M, N, K_global = 2048, 2880, 4096
K_local = K_global // world_size
M_local = M // world_size
dtype = torch.float16
warmup, iters = 30, 100

A = shmem.zeros((M, K_local), dtype=dtype)
A.copy_(torch.randn(M, K_local, dtype=dtype, device=f"cuda:{rank}") * 0.1)
B = torch.randn(K_local, N, dtype=dtype, device=f"cuda:{rank}") * 0.1
C_sym = shmem.zeros((M, N), dtype=dtype)
C_out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

hb = shmem.get_heap_bases()
cfg = _get_config(world_size, M_local)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
Cr = torch.mm(A, B)
dist.reduce_scatter_tensor(ref, Cr, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

# ---- baselines ----
Cr2 = torch.empty(M, N, dtype=dtype, device=f"cuda:{rank}")
Cro = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
for _ in range(warmup):
    torch.mm(A, B, out=Cr2)
    dist.reduce_scatter_tensor(Cro, Cr2, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=Cr2)
    dist.reduce_scatter_tensor(Cro, Cr2, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

def rs():
    _one_shot_rs_kernel[(cfg["num_sms"],)](
        C_sym, C_out, M, N, M_local,
        C_sym.stride(0), C_sym.stride(1), C_out.stride(0), C_out.stride(1),
        hb, rank, world_size, cfg["block_m"], cfg["block_n"], cfg["num_sms"],
        num_warps=cfg["num_warps"])

shmem.barrier()
for _ in range(warmup):
    torch.mm(A, B, out=C_sym)
    rs()
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    torch.mm(A, B, out=C_sym)
    rs()
e.record()
torch.cuda.synchronize()
twok_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"Pull-direction fusion: M={M}, N={N}, K={K_global}, TP={world_size}")
    print(f"RCCL:       {rccl_ms:.4f}ms")
    print(f"Two-kernel: {twok_ms:.4f}ms ({rccl_ms/twok_ms:.2f}x)")
    print()
    print("Chunked GEMM + overlapped pull RS (2 streams):")
    print(f"{'bm':>4} {'bn':>4} {'gsms':>5} | {'ms':>9} {'vs RCCL':>8} {'vs 2k':>7}")
    print("-" * 46)

# ---- chunked GEMM with overlapped pull RS ----
# All ranks compute chunk c in the same order. Once chunk c is done
# everywhere, the owner of chunk c pulls it while everyone computes c+1.
comm_stream = torch.cuda.Stream()
best = (999.0, None)

for bm, bn, gsms in [(128,256,304),(128,256,240),(128,128,304),(64,256,304)]:
    if M_local % bm != 0:
        continue
    try:
        kw = {"num_warps": 8, "num_stages": 3}
        if getattr(torch.version, "hip", None):
            kw["matrix_instr_nonkdim"] = 32

        def run():
            main = torch.cuda.current_stream()
            for c in range(world_size):
                m0 = c * M_local
                _chunk_gemm[(gsms,)](
                    A, B, C_sym, M, N, K_local, m0, M_local,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    C_sym.stride(0), C_sym.stride(1),
                    bm, bn, 64, 4, gsms, K_local % 64 == 0, **kw)
                # Chunk c is now complete on THIS rank. Peers reach the same
                # point at roughly the same time (identical work per chunk).
                # The owner of chunk c pulls it on the comm stream while the
                # main stream proceeds to chunk c+1.
                if c == rank:
                    ev = torch.cuda.Event()
                    ev.record(main)
                    comm_stream.wait_event(ev)
                    with torch.cuda.stream(comm_stream):
                        shmem.barrier()
                        rs()
            main.wait_stream(comm_stream)

        C_out.zero_()
        shmem.barrier()
        for _ in range(10):
            run()
        torch.cuda.synchronize()

        d = torch.abs(C_out - ref).max().item()
        if d > 1.0:
            if rank == 0:
                print(f"{bm:4d} {bn:4d} {gsms:5d} | FAIL diff={d:.3f}")
            continue

        s.record()
        for _ in range(iters):
            run()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        if ms < best[0]:
            best = (ms, (bm, bn, gsms))
        if rank == 0:
            print(f"{bm:4d} {bn:4d} {gsms:5d} | {ms:9.4f} {rccl_ms/ms:7.2f}x {twok_ms/ms:6.2f}x")
    except Exception as ex:
        if rank == 0:
            print(f"{bm:4d} {bn:4d} {gsms:5d} | ERROR {str(ex)[:40]}")

if rank == 0:
    print()
    print(f"RCCL:        {rccl_ms:.4f}ms")
    print(f"Two-kernel:  {twok_ms:.4f}ms ({rccl_ms/twok_ms:.2f}x)")
    if best[1]:
        print(f"Pull-fusion: {best[0]:.4f}ms ({rccl_ms/best[0]:.2f}x)")
        if best[0] < twok_ms:
            print(f"  *** BEATS TWO-KERNEL by {(twok_ms-best[0])*1000:.1f}us ***")
        else:
            print(f"  loses to two-kernel by {(best[0]-twok_ms)*1000:.1f}us")

shmem.barrier()
dist.destroy_process_group()
