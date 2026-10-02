#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Multi-GPU Jacobi example for Iris.

Each GPU owns a strip of rows.
Halo rows carry the neighbor edge values between GPUs.
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris import DeviceContext
from iris.ccl import Config


# This kernel does the actual Jacobi update on one rank.
# Halo rows make the same four-neighbor math work at GPU boundaries.
@triton.jit
def jacobi_kernel(nxt, cur, err, nx: tl.constexpr, mine: tl.constexpr, BX: tl.constexpr, BY: tl.constexpr):
    px = tl.program_id(0)
    py = tl.program_id(1)
    x = (1 + px * BX + tl.arange(0, BX))[None, :]
    y = (1 + py * BY + tl.arange(0, BY))[:, None]
    ok = (x < nx - 1) & (y <= mine)
    pos = y * nx + x

    mid = tl.load(cur + pos, mask=ok, other=0.0)
    r = tl.load(cur + pos + 1, mask=ok, other=0.0)
    l = tl.load(cur + pos - 1, mask=ok, other=0.0)
    d = tl.load(cur + pos + nx, mask=ok, other=0.0)
    u = tl.load(cur + pos - nx, mask=ok, other=0.0)

    val = 0.25 * (r + l + d + u)
    tl.store(nxt + pos, val, mask=ok)

    diff = (val - mid) * (val - mid)
    sm = tl.sum(tl.sum(diff, axis=1), axis=0)
    tl.atomic_add(err, sm)


# This kernel pushes edge rows straight into neighbor halo memory.
# The upper destination depends on how many rows that rank owns.
@triton.jit
def halo_kernel(
    dev_ctx,
    buf,
    nx: tl.constexpr,
    mine: tl.constexpr,
    up: tl.constexpr,
    dn: tl.constexpr,
    up_n: tl.constexpr,
    rank: tl.constexpr,
    nranks: tl.constexpr,
    BS: tl.constexpr,
):
    ctx = DeviceContext.initialize(dev_ctx, rank, nranks)
    x = 1 + tl.program_id(0) * BS + tl.arange(0, BS)
    ok = x < nx - 1

    if up >= 0:
        vals = tl.load(buf + nx + x, mask=ok, other=0.0)
        dst = (up_n + 1) * nx + x
        ctx.store(buf + dst, vals, to_rank=up, mask=ok)

    if dn >= 0:
        vals = tl.load(buf + mine * nx + x, mask=ok, other=0.0)
        ctx.store(buf + x, vals, to_rank=dn, mask=ok)


# Split only the interior rows.
# Early ranks get one extra row when the split is uneven.
def split_rows(ny, rank, nranks):
    inside = ny - 2
    if inside < 1:
        raise ValueError("ny must contain at least one interior row")
    if nranks < 1 or not 0 <= rank < nranks:
        raise ValueError("invalid rank setup")
    if nranks > inside:
        raise ValueError("nranks cannot exceed the number of interior rows")

    base, extra = divmod(inside, nranks)
    mine = base + int(rank < extra)
    first = 1 + rank * base + min(rank, extra)
    return first, first + mine - 1, mine


# Launch the remote halo writes then wait before another stencil step starts.
# Without this barrier a rank could read stale neighbor data.
def push_halos(ctx, dev_ctx, buf, nx, mine, up, dn, up_n, rank, nranks):
    bs = 256
    halo_kernel[(triton.cdiv(nx - 2, bs),)](dev_ctx, buf, nx, mine, up, dn, up_n, rank, nranks, BS=bs, num_warps=4)
    torch.cuda.synchronize()
    ctx.barrier()


# One distributed iteration is local compute then halo exchange then global error.
# Keeping those three pieces together makes the main loop much smaller.
def do_step(ctx, dev_ctx, cur, nxt, err, all_err, nx, mine, up, dn, up_n, rank, nranks):
    err.zero_()
    all_err.zero_()

    bx, by = 64, 8
    grid = (triton.cdiv(nx - 2, bx), triton.cdiv(mine, by))
    jacobi_kernel[grid](nxt, cur, err, nx, mine, BX=bx, BY=by, num_warps=4, num_stages=2)

    push_halos(ctx, dev_ctx, nxt, nx, mine, up, dn, up_n, rank, nranks)

    # Every rank needs the same residual so they all stop on the same iteration.
    ctx.ccl.all_reduce(all_err, err)
    torch.cuda.synchronize()
    return torch.sqrt(all_err).item()


# Validation needs one normal grid instead of separate padded slabs.
# All gather returns every rank slab so we can rebuild that grid on each rank.
def gather_grid(ctx, cur, nx, ny, mine, max_n, nranks):
    send = ctx.zeros((max_n, nx), dtype=torch.float32)
    send[:mine].copy_(cur[1 : mine + 1])
    got = ctx.zeros((nranks * max_n, nx), dtype=torch.float32)

    cfg = Config(
        block_size_m=32,
        block_size_n=64,
        comm_sms=64,
        num_stages=1,
        num_warps=4,
        waves_per_eu=0,
        use_gluon=False,
    )
    ctx.barrier()
    ctx.ccl.all_gather(got, send, config=cfg)
    torch.cuda.synchronize()

    full = torch.zeros((ny, nx), dtype=torch.float32, device=cur.device)
    full[:, 0] = 100.0
    full[:, -1] = 0.0
    full[0, :] = 50.0
    full[-1, :] = 0.0

    for src in range(nranks):
        first, last, n = split_rows(ny, src, nranks)
        start = src * max_n
        full[first : last + 1].copy_(got[start : start + n])
    return full


# This is an independent single-GPU answer used only for validation.
# It helps catch a bad split or halo exchange without depending on Iris RMA.
def ref_jacobi(nx, ny, nit, dev):
    cur = torch.zeros((ny, nx), dtype=torch.float32, device=dev)
    cur[:, 0] = 100.0
    cur[:, -1] = 0.0
    cur[0, :] = 50.0
    cur[-1, :] = 0.0
    nxt = cur.clone()

    for _ in range(nit):
        nxt[1:-1, 1:-1] = 0.25 * (cur[1:-1, 2:] + cur[1:-1, :-2] + cur[2:, 1:-1] + cur[:-2, 1:-1])
        cur, nxt = nxt, cur
    return cur


# main sets up the rank layout then keeps calling do_step.
# Validation is optional since gathering the whole grid is not part of the solver.
def main():
    p = argparse.ArgumentParser(description="Multi-GPU 2D Jacobi iteration with Iris")
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--ny", type=int, default=512)
    p.add_argument("--max_iterations", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--heap_size", type=int, default=1 << 30)
    p.add_argument("-v", "--validate", action="store_true")
    a = p.parse_args()

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="gloo")

    try:
        ctx = iris.iris(heap_size=a.heap_size)
        rank = ctx.get_rank()
        nranks = ctx.get_num_ranks()

        if a.nx < 3 or a.max_iterations < 1 or a.tolerance <= 0:
            raise ValueError("invalid Jacobi arguments")

        first, last, mine = split_rows(a.ny, rank, nranks)
        max_n = (a.ny - 2 + nranks - 1) // nranks
        buf_n = max_n + 2
        up = rank - 1 if rank > 0 else -1
        dn = rank + 1 if rank < nranks - 1 else -1
        up_n = split_rows(a.ny, up, nranks)[2] if up >= 0 else 0

        # Two extra rows hold the top and bottom halo.
        # Physical borders keep the values from issue 117.
        cur = ctx.zeros((buf_n, a.nx), dtype=torch.float32)
        cur[:, 0] = 100.0
        cur[:, -1] = 0.0
        if rank == 0:
            cur[0] = 50.0
        if rank == nranks - 1:
            cur[mine + 1] = 0.0
        nxt = ctx.zeros((buf_n, a.nx), dtype=torch.float32)
        nxt.copy_(cur)

        dev_ctx = ctx.get_device_context()
        err = ctx.zeros((1, 1), dtype=torch.float32)
        all_err = ctx.zeros((1, 1), dtype=torch.float32)
        ctx.info(
            f"rank={rank}/{nranks}: rows={first}..{last} owned={mine} storage={tuple(cur.shape)} neighbors=({up}, {dn})"
        )

        # Fill halos once before the first stencil read.
        push_halos(ctx, dev_ctx, cur, a.nx, mine, up, dn, up_n, rank, nranks)

        l2 = float("inf")
        done = 0
        for i in range(a.max_iterations):
            l2 = do_step(ctx, dev_ctx, cur, nxt, err, all_err, a.nx, mine, up, dn, up_n, rank, nranks)
            cur, nxt = nxt, cur
            done = i + 1
            if rank == 0 and done % 100 == 0:
                ctx.info(f"Iteration {done}: L2 norm = {l2:.6e}")
            if l2 < a.tolerance:
                break

        if rank == 0:
            msg = "Converged" if l2 < a.tolerance else "Stopped"
            ctx.info(f"{msg} after {done} iterations: L2 norm = {l2:.6e}")

        if a.validate:
            full = gather_grid(ctx, cur, a.nx, a.ny, mine, max_n, nranks)
            ref = ref_jacobi(a.nx, a.ny, done, full.device)
            mx = (full - ref).abs().max().item()
            ok = torch.allclose(full, ref, atol=1e-3, rtol=1e-4)
            if rank == 0:
                ctx.info(f"Validation {'passed' if ok else 'failed'}: max absolute error = {mx:.6e}")
            if not ok:
                raise AssertionError(f"Jacobi result does not match reference: max absolute error = {mx:.6e}")

        ctx.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
