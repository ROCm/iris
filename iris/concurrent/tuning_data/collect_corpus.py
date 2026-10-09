# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Collect a concurrent GEMM+collective **tuning corpus** for :mod:`iris.concurrent`.

For every (collective, GEMM shape, comm shape) in a fixed corpus, this benchmarks
a full grid of ``(gemm_wgs, gemm_block)`` configs on-device and records the
timing of every point plus the measured argmin (ground-truth optimum) and the
current static-default baseline. The output is the raw data behind the
universally-good candidate set and the cost model's validation set -- run it once
per architecture / world size and commit the JSON so we never have to re-run.

Launch (single node, W ranks on W GPUs)::

    torchrun --nproc_per_node=<W> -m iris.concurrent.tuning_data.collect_corpus [--quick]

Writes ``iris/concurrent/tuning_data/data/<arch>_world<W>.json`` (schema in the
sibling ``README.md``). This tool does NOT require origami -- it measures the
grid directly, independent of any cost model.
"""

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist

# ---- config grid (as fractions of num_wgs so it is arch-portable) ----------
SPLIT_FRACS = [0.55, 0.65, 0.72, 0.78, 0.84, 0.88, 0.92, 0.96, 1.0]
TILES = [(256, 256, 64), (128, 256, 64), (256, 128, 64), (128, 128, 64), (256, 64, 64)]
DEFAULT_FRAC = 0.75  # matches iris.concurrent.gemm static default (num_wgs*3//4)

# ---- shape corpus: (M, N, K, comm_m_mult (x 256 x world), comm_n) ----------
# Spans decode (M=1) -> large (16384), the three skew axes, and comm-heavy.
SHAPES = [
    # decode / tiny-M (comm-bound)
    (1, 8192, 8192, 1, 4096),
    (16, 8192, 8192, 1, 4096),
    (128, 8192, 8192, 1, 4096),
    # small / mid square
    (512, 8192, 8192, 1, 4096),
    (2048, 2048, 2048, 1, 2048),
    (4096, 4096, 4096, 1, 2048),
    # large square
    (8192, 8192, 8192, 1, 4096),
    (16384, 16384, 16384, 1, 4096),
    # skew: small N, small K, big K
    (4096, 512, 8192, 1, 2048),
    (4096, 8192, 512, 1, 2048),
    (2048, 2048, 16384, 1, 2048),
    # tall / wide
    (16384, 2048, 2048, 1, 4096),
    (4096, 16384, 2048, 1, 4096),
    # comm-heavy (big all-gather relative to GEMM)
    (4096, 4096, 4096, 4, 8192),
    (8192, 8192, 8192, 4, 16384),
]

QUICK_SHAPES = [(1, 8192, 8192, 1, 4096), (4096, 4096, 4096, 1, 2048), (16384, 16384, 16384, 1, 4096)]
QUICK_FRACS = [0.75, 0.9, 1.0]
QUICK_TILES = [(256, 256, 64), (256, 64, 64)]

COLLECTIVES = ["all_gather", "all_reduce", "reduce_scatter", "broadcast"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    ap.add_argument("--collectives", nargs="*", default=COLLECTIVES)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-repeat", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl", rank=rank, world_size=ws, device_id=torch.device(f"cuda:{lr}"))

    import iris
    import iris.concurrent.gemm as cg
    from iris.host.platform.utils import do_bench

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    shapes = QUICK_SHAPES if args.quick else SHAPES
    fracs = QUICK_FRACS if args.quick else SPLIT_FRACS
    tiles = QUICK_TILES if args.quick else TILES

    shmem = iris.iris(1 << 35)
    W = shmem.get_num_ranks()
    cu = shmem.get_cu_count()
    try:
        props = torch.cuda.get_device_properties(lr)
        arch = (getattr(props, "gcnArchName", None) or props.name).split(":")[0]
        dev_name = props.name
    except Exception:
        arch, dev_name = "unknown", "unknown"

    log(
        f"arch={arch} device={dev_name} world={W} cu={cu} grid={len(fracs)}x{len(tiles)} shapes={len(shapes)} colls={args.collectives}"
    )

    results = []
    dtype = torch.float16
    t_start = time.time()
    for coll in args.collectives:
        op = getattr(cg, coll)
        for M, N, K, mmult, cn in shapes:
            cm = 256 * W * mmult
            # skip reduce_scatter/all_to_all shapes whose comm rows don't divide W
            if coll == "reduce_scatter" and cm % W != 0:
                continue
            if coll == "all_gather":
                dm = W * cm
            elif coll == "reduce_scatter":
                dm = cm // W
            else:
                dm = cm
            A = B = src = C = dst = None
            try:
                A = shmem.randn(M, K, device="cuda", dtype=dtype)
                B = shmem.randn(N, K, device="cuda", dtype=dtype).T
                src = shmem.full((cm, cn), float(rank + 1), device="cuda", dtype=dtype)
                C = shmem.zeros((M, N), device="cuda", dtype=dtype)
                dst = shmem.zeros((dm, cn), device="cuda", dtype=dtype)

                grid = []
                fracset = sorted(set(fracs) | {DEFAULT_FRAC})
                for frac in fracset:
                    gw = max(1, min(cu, round(frac * cu)))
                    for tile in tiles:
                        shmem.barrier()
                        try:
                            ms = do_bench(
                                lambda gw=gw, tile=tile: op(
                                    shmem,
                                    A,
                                    B,
                                    src,
                                    C=C,
                                    comm_dst=dst,
                                    mode="fused",
                                    num_wgs=cu,
                                    gemm_wgs=gw,
                                    gemm_block=tile,
                                ),
                                barrier_fn=shmem.barrier,
                                n_warmup=args.n_warmup,
                                n_repeat=args.n_repeat,
                                return_mode="median",
                            )
                        except Exception as e:
                            ms = float("nan")
                            if rank == 0:
                                log(f"  {coll} {M}x{N}x{K} gw={gw} tile={tile} FAILED: {e!r}")
                        grid.append({"frac": frac, "gemm_wgs": gw, "gemm_block": list(tile), "ms": ms})

                valid = [g for g in grid if g["ms"] == g["ms"]]  # drop nan
                best = min(valid, key=lambda g: g["ms"]) if valid else None
                dgw = max(1, min(cu, round(DEFAULT_FRAC * cu)))
                dblk = [256, 64, 64] if coll == "all_gather" else [256, 256, 64]
                base = next((g for g in valid if g["gemm_wgs"] == dgw and g["gemm_block"] == dblk), None)
                row = {
                    "collective": coll,
                    "M": M,
                    "N": N,
                    "K": K,
                    "comm_m": cm,
                    "comm_n": cn,
                    "best": best,
                    "baseline": base,
                    "grid": grid,
                }
                results.append(row)
                if rank == 0 and best:
                    sp = (base["ms"] / best["ms"]) if base and best["ms"] else float("nan")
                    log(
                        f"  {coll:<14} {M:>6}x{N:<6}x{K:<6} comm {cm}x{cn}  best gw={best['gemm_wgs']} blk={tuple(best['gemm_block'])} {best['ms']:.4f}ms  vs default {sp:.2f}x"
                    )
            finally:
                # Free this shape's symmetric-heap buffers so they don't
                # accumulate across the corpus (else the heap OOMs at world>=4,
                # where the comm tensors scale with world size).
                A = B = src = C = dst = None
                gc.collect()
                torch.cuda.empty_cache()
                shmem.barrier()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        payload = {
            "arch": arch,
            "device_name": dev_name,
            "world_size": W,
            "cu_count": cu,
            # torch/ROCm build string trimmed to the public version (drop the
            # "+rocm..." local-build suffix); host intentionally omitted.
            "torch_version": torch.__version__.split("+")[0],
            "python": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "grid": {
                "split_fracs": sorted(set(fracs) | {DEFAULT_FRAC}),
                "tiles": [list(t) for t in tiles],
                "default_frac": DEFAULT_FRAC,
            },
            "n_warmup": args.n_warmup,
            "n_repeat": args.n_repeat,
            "quick": args.quick,
            "results": results,
        }
        out_path = os.path.join(args.out, f"{arch}_world{W}{'_quick' if args.quick else ''}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"\nwrote {out_path}  ({len(results)} rows, {time.time() - t_start:.1f}s)")

    shmem.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
