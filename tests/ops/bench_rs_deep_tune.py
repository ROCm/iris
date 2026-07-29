#!/usr/bin/env python3
"""Deep tune the RS kernel — the 0.092ms floor may not be a floor.

The RS is 70% of E2E time. We tuned tile sizes and SMS but never:
- num_stages (software pipelining of the peer loads)
- waves_per_eu (occupancy control)
- cache modifiers per-load
- unrolled peer loop vs static_range
- 1D vs 2D tile shapes at the same element count
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


@triton.jit
def _rs_tuned(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr,
    ROTATE: tl.constexpr,
):
    pid = tl.program_id(0)
    acc_dtype = tl.float32
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    total = nm * nn
    m_off = cur_rank * nm

    for t in range(pid, total, NUM_SMS):
        lm = t // nn
        pn = t % nn
        gm = m_off + lm

        rm = gm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)

        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        full = (gm * BM + BM <= M) & (pn * BN + BN <= N)

        if full:
            if ROTATE:
                sr = pid % world_size
            else:
                sr = 0
            acc = iris.load(bp, cur_rank, sr, heap_bases, hint=(1, BN)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (sr + i) % world_size
                acc += iris.load(bp, cur_rank, r, heap_bases, hint=(1, BN)).to(acc_dtype)
            om = lm * BM + tl.arange(0, BM)
            om = tl.max_contiguous(tl.multiple_of(om, BM), BM)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            msk = (rm[:, None] < M) & (rn[None, :] < N)
            if ROTATE:
                sr = pid % world_size
            else:
                sr = 0
            acc = iris.load(bp, cur_rank, sr, heap_bases, mask=msk, hint=(1, BN)).to(acc_dtype)
            for i in tl.static_range(1, world_size):
                r = (sr + i) % world_size
                acc += iris.load(bp, cur_rank, r, heap_bases, mask=msk, hint=(1, BN)).to(acc_dtype)
            om = lm * BM + tl.arange(0, BM)
            omsk = (om[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=omsk)


M, N = 2048, 2880
M_local = M // world_size
dtype = torch.float16
warmup, iters = 100, 400

inp = shmem.zeros((M, N), dtype=dtype)
inp.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
heap_bases = shmem.get_heap_bases()
shmem.barrier()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# RCCL + reference
ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
inp_c = inp.clone()
for _ in range(warmup):
    dist.reduce_scatter_tensor(ref, inp_c, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(ref, inp_c, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

if rank == 0:
    print(f"RS deep tune: M={M}, N={N}, TP={world_size}")
    print(f"RCCL: {rccl_ms:.4f}ms")
    print(f"Current best (bm=128 bn=64 sms=196 w=4): 0.092ms")
    print()
    print(f"{'bm':>4} {'bn':>4} {'sms':>4} {'w':>2} {'st':>3} {'wpe':>4} {'rot':>4} | {'ms':>8} {'GB/s':>7} {'vs RCCL':>8}")
    print("-" * 70)

best = (999.0, None)
out = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")

for bm in [32, 64, 128, 256]:
    if M_local % bm != 0:
        continue
    for bn in [32, 64, 128, 256]:
        for sms in [32, 64, 128, 196, 256, 304]:
            for warps in [1, 2, 4, 8]:
                for stages in [1, 2, 3]:
                    for wpe in [1, 2, 4]:
                        for rot in [True, False]:
                            try:
                                kw = {"num_warps": warps, "num_stages": stages,
                                      "waves_per_eu": wpe}
                                out.zero_()
                                for _ in range(5):
                                    _rs_tuned[(sms,)](
                                        inp, out, M, N, M_local,
                                        inp.stride(0), inp.stride(1),
                                        out.stride(0), out.stride(1),
                                        heap_bases, rank, world_size,
                                        bm, bn, sms, rot, **kw)
                                torch.cuda.synchronize()

                                d = torch.abs(out - ref).max().item()
                                if d > 1.0:
                                    continue

                                s.record()
                                for _ in range(iters):
                                    _rs_tuned[(sms,)](
                                        inp, out, M, N, M_local,
                                        inp.stride(0), inp.stride(1),
                                        out.stride(0), out.stride(1),
                                        heap_bases, rank, world_size,
                                        bm, bn, sms, rot, **kw)
                                e.record()
                                torch.cuda.synchronize()
                                ms = s.elapsed_time(e) / iters

                                if ms < best[0]:
                                    best = (ms, (bm, bn, sms, warps, stages, wpe, rot))
                                    if rank == 0:
                                        bw = M * N * 2 * (world_size-1) / world_size / (ms/1000) / 1e9
                                        print(f"{bm:4d} {bn:4d} {sms:4d} {warps:2d} {stages:3d} {wpe:4d} {str(rot):>4} | "
                                              f"{ms:8.4f} {bw:7.1f} {rccl_ms/ms:7.2f}x  ***")
                            except Exception:
                                continue

if rank == 0:
    print()
    print(f"RCCL:      {rccl_ms:.4f}ms")
    print(f"Best RS:   {best[0]:.4f}ms ({rccl_ms/best[0]:.2f}x)")
    if best[1]:
        bm, bn, sms, w, st, wpe, rot = best[1]
        print(f"  bm={bm} bn={bn} sms={sms} warps={w} stages={st} waves_per_eu={wpe} rotate={rot}")
    print(f"Previous best: 0.0920ms")
    if best[0] < 0.092:
        print(f"IMPROVEMENT: {(0.092 - best[0])*1000:.1f}us ({0.092/best[0]:.2f}x)")

shmem.barrier()
dist.destroy_process_group()
