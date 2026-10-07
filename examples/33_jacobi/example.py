#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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


LEFT_BOUNDARY = 100.0
RIGHT_BOUNDARY = 0.0
TOP_BOUNDARY = 50.0
BOTTOM_BOUNDARY = 0.0


# This kernel does the actual Jacobi update on one rank.
# Halo rows make the same four-neighbor math work at GPU boundaries.
@triton.jit
def jacobi_kernel(nxt, cur, err, nx: tl.constexpr, owned_rows: tl.constexpr, BX: tl.constexpr, BY: tl.constexpr):
    px = tl.program_id(0)
    py = tl.program_id(1)
    # Local row zero is the upper halo. Update only owned interior rows and columns.
    x = (1 + px * BX + tl.arange(0, BX))[None, :]
    y = (1 + py * BY + tl.arange(0, BY))[:, None]
    ok = (x < nx - 1) & (y <= owned_rows)
    pos = y * nx + x

    # Each updated cell reads its four neighbors, including halo cells at rank edges.
    mid = tl.load(cur + pos, mask=ok, other=0.0)
    r = tl.load(cur + pos + 1, mask=ok, other=0.0)
    l = tl.load(cur + pos - 1, mask=ok, other=0.0)
    d = tl.load(cur + pos + nx, mask=ok, other=0.0)
    u = tl.load(cur + pos - nx, mask=ok, other=0.0)

    val = 0.25 * (r + l + d + u)
    tl.store(nxt + pos, val, mask=ok)

    # Sum squared changes locally; the host reduces this residual across ranks.
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
    owned_rows: tl.constexpr,
    upper_rank: tl.constexpr,
    lower_rank: tl.constexpr,
    upper_owned_rows: tl.constexpr,
    rank: tl.constexpr,
    nranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    ctx = DeviceContext.initialize(dev_ctx, rank, nranks)
    # Border columns do not change, so exchange interior columns only.
    x = 1 + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ok = x < nx - 1

    if upper_rank >= 0:
        # Send our first owned row into the upper rank's bottom halo.
        vals = tl.load(buf + nx + x, mask=ok, other=0.0)
        dst = (upper_owned_rows + 1) * nx + x
        ctx.store(buf + dst, vals, to_rank=upper_rank, mask=ok)

    if lower_rank >= 0:
        # Send our last owned row into the lower rank's top halo.
        vals = tl.load(buf + owned_rows * nx + x, mask=ok, other=0.0)
        ctx.store(buf + x, vals, to_rank=lower_rank, mask=ok)


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
    owned_rows = base + int(rank < extra)
    first = 1 + rank * base + min(rank, extra)
    return first, first + owned_rows - 1, owned_rows


# Launch the remote halo writes then wait before another stencil step starts.
# Without this barrier a rank could read stale neighbor data.
def push_halos(ctx, dev_ctx, buf, nx, owned_rows, upper_rank, lower_rank, upper_owned_rows, rank, nranks):
    halo_block_size = 256  # Interior columns handled per Triton program.
    halo_kernel[(triton.cdiv(nx - 2, halo_block_size),)](
        dev_ctx,
        buf,
        nx,
        owned_rows,
        upper_rank,
        lower_rank,
        upper_owned_rows,
        rank,
        nranks,
        BLOCK_SIZE=halo_block_size,
        num_warps=4,
    )
    torch.cuda.synchronize()
    ctx.barrier()


# One distributed iteration is local compute then halo exchange then global error.
# Keeping those three pieces together makes the main loop much smaller.
def do_step(
    ctx, dev_ctx, cur, nxt, err, all_err, nx, owned_rows, upper_rank, lower_rank, upper_owned_rows, rank, nranks
):
    err.zero_()
    all_err.zero_()

    tile_columns, tile_rows = 64, 8
    grid = (triton.cdiv(nx - 2, tile_columns), triton.cdiv(owned_rows, tile_rows))
    jacobi_kernel[grid](nxt, cur, err, nx, owned_rows, BX=tile_columns, BY=tile_rows, num_warps=4, num_stages=2)

    push_halos(ctx, dev_ctx, nxt, nx, owned_rows, upper_rank, lower_rank, upper_owned_rows, rank, nranks)

    # Every rank needs the same residual so they all stop on the same iteration.
    ctx.ccl.all_reduce(all_err, err)
    torch.cuda.synchronize()
    return torch.sqrt(all_err).item()


# Validation needs one normal grid instead of separate padded slabs.
# All gather returns every rank slab so we can rebuild that grid on each rank.
def gather_grid(ctx, cur, nx, ny, owned_rows, max_owned_rows, nranks):
    # Pad each rank's rows to the same length for all_gather.
    send = ctx.zeros((max_owned_rows, nx), dtype=torch.float32)
    send[:owned_rows].copy_(cur[1 : owned_rows + 1])
    got = ctx.zeros((nranks * max_owned_rows, nx), dtype=torch.float32)

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
    full[:, 0] = LEFT_BOUNDARY
    full[:, -1] = RIGHT_BOUNDARY
    full[0, :] = TOP_BOUNDARY
    full[-1, :] = BOTTOM_BOUNDARY

    # Remove padding and place each slab at its global row offset.
    for src in range(nranks):
        first, last, n = split_rows(ny, src, nranks)
        start = src * max_owned_rows
        full[first : last + 1].copy_(got[start : start + n])
    return full


# This is an independent single-GPU answer used only for validation.
# It helps catch a bad split or halo exchange without depending on Iris RMA.
def ref_jacobi(nx, ny, nit, dev):
    cur = torch.zeros((ny, nx), dtype=torch.float32, device=dev)
    cur[:, 0] = LEFT_BOUNDARY
    cur[:, -1] = RIGHT_BOUNDARY
    cur[0, :] = TOP_BOUNDARY
    cur[-1, :] = BOTTOM_BOUNDARY
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

        # Each rank owns contiguous interior rows plus two halo slots.
        first, last, owned_rows = split_rows(a.ny, rank, nranks)
        max_owned_rows = (a.ny - 2 + nranks - 1) // nranks
        buffer_rows = max_owned_rows + 2
        upper_rank = rank - 1 if rank > 0 else -1
        lower_rank = rank + 1 if rank < nranks - 1 else -1
        upper_owned_rows = split_rows(a.ny, upper_rank, nranks)[2] if upper_rank >= 0 else 0

        # Two extra rows hold the top and bottom halo.
        # The physical borders use the same fixed values as the reference grid.
        cur = ctx.zeros((buffer_rows, a.nx), dtype=torch.float32)
        cur[:, 0] = LEFT_BOUNDARY
        cur[:, -1] = RIGHT_BOUNDARY
        if rank == 0:
            cur[0] = TOP_BOUNDARY
        if rank == nranks - 1:
            cur[owned_rows + 1] = BOTTOM_BOUNDARY
        nxt = ctx.zeros((buffer_rows, a.nx), dtype=torch.float32)
        nxt.copy_(cur)

        dev_ctx = ctx.get_device_context()
        err = ctx.zeros((1, 1), dtype=torch.float32)
        all_err = ctx.zeros((1, 1), dtype=torch.float32)
        ctx.info(
            f"rank={rank}/{nranks}: rows={first}..{last} owned={owned_rows} "
            f"storage={tuple(cur.shape)} neighbors=({upper_rank}, {lower_rank})"
        )

        # Fill halos once before the first stencil read.
        push_halos(ctx, dev_ctx, cur, a.nx, owned_rows, upper_rank, lower_rank, upper_owned_rows, rank, nranks)

        l2 = float("inf")
        done = 0
        for i in range(a.max_iterations):
            l2 = do_step(
                ctx,
                dev_ctx,
                cur,
                nxt,
                err,
                all_err,
                a.nx,
                owned_rows,
                upper_rank,
                lower_rank,
                upper_owned_rows,
                rank,
                nranks,
            )
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
            full = gather_grid(ctx, cur, a.nx, a.ny, owned_rows, max_owned_rows, nranks)
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
