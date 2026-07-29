#!/usr/bin/env python3
"""RS with fully-contiguous peer reads — test if XGMI wants long runs.

Current RS reads BLOCK_M x BLOCK_N 2D tiles. Each row is only
BLOCK_N*2 = 128 bytes contiguous, then a stride jump of N*2 = 5760 bytes.
XGMI/IOMMU may want much longer contiguous runs to reach line rate.

Since each rank owns CONTIGUOUS rows [r*M_local, (r+1)*M_local), the
whole owned region is contiguous in memory: M_local * N elements.
We can read it as a flat 1D range with NO stride jumps at all.

Current: 62 GB/s (14% of 448 GB/s line rate).
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
def _rs_flat_contig(
    input_ptr, output_ptr,
    n_elem,          # M_local * N — this rank's contiguous slice
    elem_offset,     # rank * M_local * N — where our slice starts
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BLOCK: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """Flat contiguous RS: no 2D tiling, no stride jumps.

    Each WG reads a contiguous BLOCK-element run from every peer.
    BLOCK*2 bytes of contiguous XGMI traffic per load.
    """
    pid = tl.program_id(0)
    for base in range(pid * BLOCK, n_elem, NUM_SMS * BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elem
        src = input_ptr + elem_offset + offs

        acc = iris.load(src, cur_rank, 0, heap_bases, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(1, world_size):
            acc += iris.load(src, cur_rank, i, heap_bases, mask=mask, other=0.0).to(tl.float32)

        tl.store(output_ptr + offs, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _rs_2d_tiled(
    input_ptr, output_ptr,
    M, N, M_local,
    stride_in_m, stride_in_n, stride_out_m, stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """Current approach: 2D tiles with stride jumps between rows."""
    pid = tl.program_id(0)
    nm = M_local // BM
    nn = tl.cdiv(N, BN)
    m_off = cur_rank * nm
    for t in range(pid, nm * nn, NUM_SMS):
        lm = t // nn
        pn = t % nn
        gm = m_off + lm
        rm = gm * BM + tl.arange(0, BM)
        rm = tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        rn = pn * BN + tl.arange(0, BN)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
        off = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        bp = input_ptr + off
        if (gm * BM + BM <= M) & (pn * BN + BN <= N):
            acc = iris.load(bp, cur_rank, 0, heap_bases, hint=(1, BN)).to(tl.float32)
            for i in tl.static_range(1, world_size):
                acc += iris.load(bp, cur_rank, i, heap_bases, hint=(1, BN)).to(tl.float32)
            om = lm * BM + tl.arange(0, BM)
            om = tl.max_contiguous(tl.multiple_of(om, BM), BM)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty))
        else:
            msk = (rm[:, None] < M) & (rn[None, :] < N)
            acc = iris.load(bp, cur_rank, 0, heap_bases, mask=msk, hint=(1, BN)).to(tl.float32)
            for i in tl.static_range(1, world_size):
                acc += iris.load(bp, cur_rank, i, heap_bases, mask=msk, hint=(1, BN)).to(tl.float32)
            om = lm * BM + tl.arange(0, BM)
            omsk = (om[:, None] < M_local) & (rn[None, :] < N)
            tl.store(output_ptr + om[:, None] * stride_out_m + rn[None, :] * stride_out_n,
                     acc.to(output_ptr.type.element_ty), mask=omsk)


M, N = 2048, 2880
M_local = M // world_size
n_elem = M_local * N
elem_offset = rank * M_local * N
dtype = torch.float16
warmup, iters = 100, 400

inp = shmem.zeros((M, N), dtype=dtype)
inp.copy_(torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}"))
out2d = torch.zeros(M_local, N, dtype=dtype, device=f"cuda:{rank}")
out1d = torch.zeros(n_elem, dtype=dtype, device=f"cuda:{rank}")
hb = shmem.get_heap_bases()
shmem.barrier()

s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

ref = torch.empty(M_local, N, dtype=dtype, device=f"cuda:{rank}")
dist.reduce_scatter_tensor(ref, inp.clone(), op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

for _ in range(warmup):
    dist.reduce_scatter_tensor(ref, inp, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
s.record()
for _ in range(iters):
    dist.reduce_scatter_tensor(ref, inp, op=dist.ReduceOp.SUM)
e.record()
torch.cuda.synchronize()
rccl_ms = s.elapsed_time(e) / iters

def bw(ms):
    return M * N * 2 * (world_size - 1) / world_size / (ms / 1000) / 1e9

if rank == 0:
    print(f"Contiguous vs tiled RS: M={M}, N={N}, TP={world_size}")
    print(f"This rank owns {n_elem} contiguous elements ({n_elem*2/1e6:.1f} MB)")
    print(f"RCCL: {rccl_ms:.4f}ms ({bw(rccl_ms):.1f} GB/s)")
    print()
    print("2D tiled (current):")

best_2d = 999.0
for bm, bn, sms in [(64,64,128),(128,64,196),(128,128,128),(256,64,128)]:
    if M_local % bm != 0:
        continue
    try:
        out2d.zero_()
        for _ in range(20):
            _rs_2d_tiled[(sms,)](inp, out2d, M, N, M_local,
                inp.stride(0), inp.stride(1), out2d.stride(0), out2d.stride(1),
                hb, rank, world_size, bm, bn, sms, num_warps=4)
        torch.cuda.synchronize()
        if torch.abs(out2d - ref).max().item() > 1.0:
            continue
        s.record()
        for _ in range(iters):
            _rs_2d_tiled[(sms,)](inp, out2d, M, N, M_local,
                inp.stride(0), inp.stride(1), out2d.stride(0), out2d.stride(1),
                hb, rank, world_size, bm, bn, sms, num_warps=4)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        best_2d = min(best_2d, ms)
        if rank == 0:
            print(f"  bm={bm:3d} bn={bn:3d} sms={sms:3d}: {ms:.4f}ms ({bw(ms):.1f} GB/s)")
    except Exception:
        continue

if rank == 0:
    print()
    print("1D flat contiguous (BLOCK elements per load, zero stride jumps):")

best_1d = 999.0
best_1d_cfg = None
for BLOCK in [1024, 2048, 4096, 8192, 16384, 32768]:
    for sms in [32, 64, 128, 196, 256]:
        for warps in [4, 8]:
            try:
                out1d.zero_()
                for _ in range(20):
                    _rs_flat_contig[(sms,)](inp, out1d, n_elem, elem_offset,
                        hb, rank, world_size, BLOCK, sms, num_warps=warps)
                torch.cuda.synchronize()
                d = torch.abs(out1d.view(M_local, N) - ref).max().item()
                if d > 1.0:
                    continue
                s.record()
                for _ in range(iters):
                    _rs_flat_contig[(sms,)](inp, out1d, n_elem, elem_offset,
                        hb, rank, world_size, BLOCK, sms, num_warps=warps)
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e) / iters
                if ms < best_1d:
                    best_1d = ms
                    best_1d_cfg = (BLOCK, sms, warps)
                    if rank == 0:
                        print(f"  BLOCK={BLOCK:6d} sms={sms:3d} w={warps}: "
                              f"{ms:.4f}ms ({bw(ms):.1f} GB/s)  ***")
            except Exception:
                continue

if rank == 0:
    print()
    print(f"RCCL:       {rccl_ms:.4f}ms ({bw(rccl_ms):.1f} GB/s)")
    print(f"2D tiled:   {best_2d:.4f}ms ({bw(best_2d):.1f} GB/s) -> {rccl_ms/best_2d:.2f}x")
    if best_1d < 999:
        print(f"1D flat:    {best_1d:.4f}ms ({bw(best_1d):.1f} GB/s) -> {rccl_ms/best_1d:.2f}x")
        print(f"  config: BLOCK={best_1d_cfg[0]} sms={best_1d_cfg[1]} warps={best_1d_cfg[2]}")
        if best_1d < best_2d:
            print(f"  IMPROVEMENT over 2D: {(best_2d-best_1d)*1000:.1f}us ({best_2d/best_1d:.2f}x)")
        else:
            print(f"  no improvement over 2D")

shmem.barrier()
dist.destroy_process_group()
