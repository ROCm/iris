"""Benchmark every iris CCL collective against torch.distributed (RCCL).

    torchrun --nproc_per_node=4 bench_ccl.py --collectives all --json out.json

Each collective is validated against its OWN documented semantics before being
timed, and anything that fails validation is reported and never timed. A number
from an incorrect kernel is worse than no number.

The four collectives do NOT share a tensor layout, and a generic harness
silently mis-checks two of them:

    all_gather      in (M, N)          out (world*M, N)
    all_reduce      in (M, N)          out (M, N)
    all_to_all      in (M, N*world)    out (M, N*world)   sharded by COLUMN
    reduce_scatter  in (M, N)          out (M, N)         sharded by TILE

reduce_scatter is the trap: input and output are the same shape, and the rank
owns a scattered set of block_size_m x block_size_n tiles rather than a
contiguous row range. torch.distributed.reduce_scatter_tensor splits
contiguously along dim 0, so the two produce different LAYOUTS from the same
arithmetic. Bytes moved are identical, which is what makes the timing
comparable; the outputs are not elementwise comparable and must not be
allclose'd against each other.

Bandwidth follows the NCCL convention so figures line up with rccl-tests:

    all_gather      algbw = out_bytes / t    busbw = algbw * (N-1)/N
    reduce_scatter  algbw = in_bytes  / t    busbw = algbw * (N-1)/N
    all_to_all      algbw = in_bytes  / t    busbw = algbw * (N-1)/N
    all_reduce      algbw = in_bytes  / t    busbw = algbw * 2(N-1)/N

The factor of 2 on all_reduce is real -- it moves each byte twice. Compare each
collective against its own RCCL baseline, never across collectives.

RCCL ALWAYS GETS PLAIN TORCH TENSORS, never iris symmetric-heap ones. NCCL
cannot register externally-mapped VMM memory and silently stages it through a
bounce buffer, which cost it roughly an order of magnitude here and showed up
as iris winning by 4-11x. A baseline handed memory it cannot register is not a
baseline.
"""

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

import iris
from iris.ccl import Config

COLLECTIVES = ("all_gather", "all_reduce", "all_to_all", "reduce_scatter")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def bench(fn, warmup, iters, barrier):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    barrier()
    samples = []
    for _ in range(iters):
        barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    med = samples[len(samples) // 2]
    return med, (samples[-1] - samples[0]) / med if med else float("inf")


def make_case(coll, ctx, M, N, world, rank, dtype, cfg):
    """Return (iris_fn, rccl_fn, validate_fn, bytes_for_busbw, busbw_factor)."""
    f_ag_rs = (world - 1) / world
    f_ar = 2.0 * (world - 1) / world
    item = torch.empty((), dtype=dtype).element_size()

    if coll == "all_gather":
        src = ctx.zeros((M, N), dtype=dtype)
        src.fill_(float(rank + 1))
        dst = ctx.zeros((world * M, N), dtype=dtype)
        ref = torch.zeros((world * M, N), dtype=dtype, device="cuda")

        def validate():
            for r in range(world):
                blk = dst[r * M : (r + 1) * M]
                if not torch.allclose(blk, torch.full_like(blk, float(r + 1)), atol=0.5):
                    return False, f"block {r} wrong: {blk[0, 0].item()}"
            return True, ""

        rsrc = torch.full((M, N), float(rank + 1), dtype=dtype, device="cuda")
        return (
            lambda: ctx.ccl.all_gather(dst, src, config=cfg),
            lambda: dist.all_gather_into_tensor(ref, rsrc),
            validate,
            world * M * N * item,
            f_ag_rs,
        )

    if coll == "all_reduce":
        src = ctx.zeros((M, N), dtype=dtype)
        src.fill_(float(rank + 1))
        dst = ctx.zeros((M, N), dtype=dtype)
        ref = torch.zeros((M, N), dtype=dtype, device="cuda")
        want = float(sum(r + 1 for r in range(world)))

        def validate():
            ok = torch.allclose(dst, torch.full_like(dst, want), atol=0.5)
            return ok, "" if ok else f"got {dst[0, 0].item()} want {want}"

        rsrc = torch.full((M, N), float(rank + 1), dtype=dtype, device="cuda")
        return (
            lambda: ctx.ccl.all_reduce(dst, src, config=cfg),
            lambda: (ref.copy_(rsrc), dist.all_reduce(ref, op=dist.ReduceOp.SUM)),
            validate,
            M * N * item,
            f_ar,
        )

    if coll == "all_to_all":
        src = ctx.zeros((M, N * world), dtype=dtype)
        for t in range(world):
            src[:, t * N : (t + 1) * N] = float(rank * 10 + t + 1)
        dst = ctx.zeros((M, N * world), dtype=dtype)
        ref = torch.zeros((M, N * world), dtype=dtype, device="cuda")
        # RCCL equivalent moves the same bytes; it splits by ROW, iris by column.
        rsrc = torch.zeros((M * world, N), dtype=dtype, device="cuda")
        rdst = torch.zeros((M * world, N), dtype=dtype, device="cuda")

        def validate():
            for s in range(world):
                chunk = dst[:, s * N : (s + 1) * N]
                want = float(s * 10 + rank + 1)
                if not torch.allclose(chunk, torch.full_like(chunk, want), atol=0.5):
                    return False, f"chunk from {s}: got {chunk[0, 0].item()} want {want}"
            return True, ""

        return (
            lambda: ctx.ccl.all_to_all(dst, src, config=cfg),
            lambda: dist.all_to_all_single(rdst, rsrc),
            validate,
            M * N * world * item,
            f_ag_rs,
        )

    # reduce_scatter: tile-sharded. Validate only the tiles this rank owns.
    src = ctx.zeros((M, N), dtype=dtype)
    src.fill_(float(rank + 1))
    dst = ctx.zeros((M, N), dtype=dtype)
    rsrc = torch.zeros((M, N), dtype=dtype, device="cuda")
    rsrc.fill_(float(rank + 1))
    rdst = torch.zeros((M // world, N), dtype=dtype, device="cuda")
    want = float(sum(r + 1 for r in range(world)))
    bm, bn = cfg.block_size_m, cfg.block_size_n
    npm, npn = (M + bm - 1) // bm, (N + bn - 1) // bn
    total_tiles = npm * npn
    per_rank = (total_tiles + world - 1) // world
    lo, hi = rank * per_rank, min((rank + 1) * per_rank, total_tiles)

    def validate():
        for t in range(lo, hi):
            i, j = (t // npn) * bm, (t % npn) * bn
            tile = dst[i : i + bm, j : j + bn]
            if not torch.allclose(tile, torch.full_like(tile, want), atol=0.5):
                return False, f"tile {t} at ({i},{j}): got {tile[0, 0].item()} want {want}"
        return True, ""

    return (
        lambda: ctx.ccl.reduce_scatter(dst, src, config=cfg),
        lambda: dist.reduce_scatter_tensor(rdst, rsrc, op=dist.ReduceOp.SUM),
        validate,
        M * N * item,
        f_ag_rs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collectives", default="all")
    ap.add_argument(
        "--sizes", default="4194304,16777216,67108864,268435456", help="BYTES per rank of the collective's (M,N) input"
    )
    ap.add_argument("--n", type=int, default=4096, help="N columns; M derived from size")
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--comm-sms", default="64")
    ap.add_argument("--block-m", type=int, default=32)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--heap", type=int, default=8 << 30)
    ap.add_argument("--gluon", action="store_true")
    ap.add_argument(
        "--backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="process-group backend. nccl has no cross-node transport "
        "configured on this fabric, so use gloo for world>4; the "
        "RCCL baseline is then skipped rather than faked.",
    )
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    ctx = iris.iris(args.heap)
    dtype = DTYPES[args.dtype]
    item = torch.empty((), dtype=dtype).element_size()
    colls = COLLECTIVES if args.collectives == "all" else tuple(args.collectives.split(","))
    results = []

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print(
            f"# {props.gcnArchName} warp={props.warp_size} world={world} dtype={args.dtype} gluon={args.gluon}",
            flush=True,
        )
        print(
            f"# {'collective':16s}{'bytes':>11s} {'cu':>4s}  {'iris GB/s':>10s} {'rccl GB/s':>10s} {'speedup':>8s}",
            flush=True,
        )

    for coll in colls:
        for nbytes in (int(x) for x in args.sizes.split(",")):
            M = nbytes // item // args.n
            if M < world or M % world:
                M = max(world, (M // world) * world)
            for csm in (int(x) for x in args.comm_sms.split(",")):
                cfg = Config(
                    comm_sms=csm,
                    block_size_m=args.block_m,
                    block_size_n=args.block_n,
                    num_warps=args.num_warps,
                    use_gluon=args.gluon,
                )
                rec = dict(
                    collective=coll,
                    bytes_per_rank=M * args.n * item,
                    M=M,
                    N=args.n,
                    dtype=args.dtype,
                    comm_sms=csm,
                    world=world,
                    gluon=args.gluon,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    host=socket.gethostname(),
                )
                try:
                    iris_fn, rccl_fn, validate, bw_bytes, factor = make_case(
                        coll, ctx, M, args.n, world, rank, dtype, cfg
                    )
                    ctx.barrier()
                    iris_fn()
                    torch.cuda.synchronize()
                    ctx.barrier()
                    ok, why = validate()
                    rec["correct"] = bool(ok)
                    if not ok:
                        rec["why"] = why
                        if rank == 0:
                            print(
                                f"  {coll:16s}{rec['bytes_per_rank']:>11d} {csm:>4d}  INCORRECT ({why}) -- NOT TIMED",
                                flush=True,
                            )
                        results.append(rec)
                        ctx.barrier()
                        continue

                    t_i, sp_i = bench(iris_fn, args.warmup, args.iters, ctx.barrier)
                    if args.backend == "nccl":
                        t_r, sp_r = bench(rccl_fn, args.warmup, args.iters, ctx.barrier)
                    else:
                        t_r, sp_r = float("nan"), float("nan")
                    rec.update(
                        iris_ms=t_i * 1e3,
                        rccl_ms=t_r * 1e3,
                        iris_busbw_GBs=bw_bytes / t_i / 1e9 * factor,
                        rccl_busbw_GBs=bw_bytes / t_r / 1e9 * factor,
                        speedup=t_r / t_i,
                        spread_iris=sp_i,
                        spread_rccl=sp_r,
                    )
                    if rank == 0:
                        flag = "  <-- NOISY" if max(sp_i, sp_r) > 0.10 else ""
                        print(
                            f"  {coll:16s}{rec['bytes_per_rank']:>11d} {csm:>4d}  "
                            f"{rec['iris_busbw_GBs']:>10.1f} {rec['rccl_busbw_GBs']:>10.1f} "
                            f"{rec['speedup']:>7.2f}x{flag}",
                            flush=True,
                        )
                except Exception as e:
                    rec.update(correct=False, error=f"{type(e).__name__}: {e}")
                    if rank == 0:
                        print(
                            f"  {coll:16s}{rec['bytes_per_rank']:>11d} {csm:>4d}  "
                            f"FAILED {type(e).__name__}: {str(e)[:80]}",
                            flush=True,
                        )
                results.append(rec)
                ctx.barrier()

    if rank == 0 and args.json:
        json.dump(results, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
