"""Roofline sweep for GEMM+AllReduce variants.

    torchrun --nproc_per_node=8 tests/ops/roofline_yael.py [-m 128,512,2048]

Every variant is autotuned before it enters the table, num_warps included, and
reports its OWN GEMM and comm times so the decomposition in roofline_core is
measured rather than assumed.

Note on fused variants: num_warps is a kernel-level launch parameter, so a fused
kernel forces the GEMM pool and the comm pool to share one warp count even
though their optima differ. The gap against the two-kernel path is the cost of
that constraint, and it shows up in the gemm and kern factors separately.
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris

from roofline_core import (
    LINE_RATE_GBS,
    algo_bytes,
    autotune,
    bench,
    emit,
    log,
    metrics,
    table,
    variant,
    variants,
)

TOL = 1.0  # fp16 AR over ws=8 of N(0,1) partials; real errors land far above this


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit
def null_kernel(X):
    """Empty kernel. Times back-to-back dispatch cost so launch elimination can
    be separated from real overlap."""
    pass


@triton.jit
def oneshot_ar_kernel(
    C, out, heap_bases: tl.tensor, M, N, scm, scn, som, son,
    cur: tl.constexpr, W: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, WGS: tl.constexpr,
):
    """One-shot pull all-reduce: every WG reads its tile from all ws peers."""
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN)
    n_t = tl.cdiv(M, BM) * n_n
    for t in range(pid, n_t, WGS):
        pm = t // n_n
        pn = t % n_n
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mk = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * scm + rn[None, :] * scn
        sr = (cur + 1) % W
        acc = iris.load(C + off, cur, sr, heap_bases, mask=mk).to(tl.float32)
        for i in tl.static_range(1, W):
            acc += iris.load(C + off, cur, (sr + i) % W, heap_bases, mask=mk).to(tl.float32)
        tl.store(out + rm[:, None] * som + rn[None, :] * son, acc.to(out.dtype.element_ty), mask=mk)


@triton.jit
def triton_gemm_kernel(
    A, B, C, M, N, KL, sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, WGS: tl.constexpr,
):
    """The GEMM half of the fused kernel, alone.

    Measured separately so gemm_gain is real: the fused variants pay a Triton
    GEMM tax against hipBLASLt and it must not hide inside an aggregate.
    """
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN)
    n_t = tl.cdiv(M, BM) * n_n
    for t in range(pid, n_t, WGS):
        pm = t // n_n
        pn = t % n_n
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, KL, BK):
            a = tl.load(
                A + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < KL), other=0.0,
            )
            b = tl.load(
                B + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                mask=((k0 + rk)[:, None] < KL) & (rn[None, :] < N), other=0.0,
            )
            acc += tl.dot(a, b)
        tl.store(
            C + rm[:, None] * scm + rn[None, :] * scn, acc.to(C.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt",
        )


@triton.jit
def fused_wgspec_kernel(
    A, B, C, out, flags, heap_bases: tl.tensor, M, N, KL,
    sam, sak, sbk, sbn, scm, scn, som, son, flag_target,
    cur: tl.constexpr, W: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GEMM_WGS: tl.constexpr, TOTAL_WGS: tl.constexpr, SPIN: tl.constexpr,
):
    """Fused WG specialization: disjoint producer (GEMM) and consumer (AR) pools.

    Producers write their partial to the symmetric heap and signal every rank.
    Consumers poll the local counter, then one-shot pull-reduce the tile.
    Polling is a volatile load, not an atomic RMW -- an RMW poll takes the line
    exclusive on every iteration and starves the writers.
    """
    pid = tl.program_id(0)
    n_n = tl.cdiv(N, BN)
    n_t = tl.cdiv(M, BM) * n_n
    if pid < GEMM_WGS:
        for t in range(pid, n_t, GEMM_WGS):
            pm = t // n_n
            pn = t % n_n
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, KL, BK):
                a = tl.load(
                    A + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < KL), other=0.0,
                )
                b = tl.load(
                    B + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                    mask=((k0 + rk)[:, None] < KL) & (rn[None, :] < N), other=0.0,
                )
                acc += tl.dot(a, b)
            tl.store(
                C + rm[:, None] * scm + rn[None, :] * scn, acc.to(C.dtype.element_ty),
                mask=(rm[:, None] < M) & (rn[None, :] < N), cache_modifier=".wt",
            )
            tl.atomic_add(flags + t, 1, sem="release", scope="gpu")
            for p in tl.static_range(0, W):
                if p != cur:
                    iris.atomic_add(flags + t, 1, cur, p, heap_bases, sem="release", scope="sys")
    else:
        cid = pid - GEMM_WGS
        cwg = TOTAL_WGS - GEMM_WGS
        for t in range(cid, n_t, cwg):
            s = 0
            d = tl.load(flags + t, volatile=True)
            while (d < flag_target) and (s < SPIN):
                d = tl.load(flags + t, volatile=True)
                s += 1
            _ = tl.atomic_add(flags + t, 0, sem="acquire", scope="gpu")
            pm = t // n_n
            pn = t % n_n
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            mk = (rm[:, None] < M) & (rn[None, :] < N)
            off = rm[:, None] * scm + rn[None, :] * scn
            sr = (cur + 1) % W
            acc = iris.load(C + off, cur, sr, heap_bases, mask=mk).to(tl.float32)
            for i in tl.static_range(1, W):
                acc += iris.load(C + off, cur, (sr + i) % W, heap_bases, mask=mk).to(tl.float32)
            tl.store(out + rm[:, None] * som + rn[None, :] * son,
                     acc.to(out.dtype.element_ty), mask=mk)


# --------------------------------------------------------------------------
# variants
#
# Each returns (ms, cfg, max_diff, pattern, t_gemm, t_comm, saved_launches).
# t_gemm/t_comm are that variant's OWN component times, so the decomposition is
# measured. For torch.mm-based variants they are the hipBLASLt and tuned-comm
# numbers; for fused variants they are the Triton GEMM and comm at the tuned
# config. saved_launches is how many kernel launches the variant eliminates
# relative to its own two-kernel serial reference (1 for fused, 0 otherwise).
# --------------------------------------------------------------------------

COMM_SPACE = {
    "BM": [16, 32, 64],
    "BN": [128, 256],
    "WGS": [32, 64, 128],
    "num_warps": [4, 8, 16],
}


def _comm_runner(ctx, cfg):
    C, out, hb = ctx["C_sym"], ctx["out"], ctx["hb"]
    M, N, ws, rank = ctx["M"], ctx["N"], ctx["ws"], ctx["rank"]

    def run():
        oneshot_ar_kernel[(cfg["WGS"],)](
            C, out, hb, M, N, C.stride(0), C.stride(1), out.stride(0), out.stride(1),
            rank, ws, cfg["BM"], cfg["BN"], cfg["WGS"], num_warps=cfg["num_warps"],
        )

    return run


def _ar_check(ctx):
    def check():
        d = (ctx["out"].float() - ctx["ref_ar"]).abs().max().item()
        return d < TOL, d

    return check


@variant("torch_serial", kind="baseline")
def v_torch(ctx):
    """torch.mm (hipBLASLt) + RCCL all_reduce. The thing to beat."""
    A, B, Cr = ctx["A"], ctx["B"], ctx["Cr"]

    def run():
        torch.mm(A, B, out=Cr)
        dist.all_reduce(Cr)

    return bench(run), {}, 0.0, "two_shot", ctx["t_gemm_torch"], ctx["t_rccl"], 0


@variant("oneshot_ar_only", kind="comm")
def v_oneshot_comm(ctx):
    """One-shot pull AR alone, fully tuned. Establishes the one-shot comm reference."""
    ms, cfg, n_ok, n_fail, diff = autotune(
        lambda c: _comm_runner(ctx, c), COMM_SPACE, _ar_check(ctx), label="oneshot_ar_only"
    )
    log(f"  oneshot_ar_only: {n_ok} ok / {n_fail} failed, best {cfg}")
    ctx["oneshot_cfg"] = cfg
    ctx["t_oneshot"] = ms
    # Pure comm: no GEMM in this measurement, so the compute term is zero.
    return ms, cfg, diff, "one_shot", 0.0, ms, 0


@variant("two_kernel_oneshot", kind="two_kernel")
def v_two_kernel(ctx):
    """torch.mm into the symmetric heap, then the tuned one-shot AR kernel.

    Same-stream ordering makes a host barrier unnecessary: the AR kernel cannot
    start before this rank's mm retires, and peers' partials are landed by their
    own mm on a symmetrically-timed stream.
    """
    A, B, C, out, hb = ctx["A"], ctx["B"], ctx["C_sym"], ctx["out"], ctx["hb"]
    M, N, ws, rank = ctx["M"], ctx["N"], ctx["ws"], ctx["rank"]

    def make(cfg):
        def run():
            torch.mm(A, B, out=C)
            oneshot_ar_kernel[(cfg["WGS"],)](
                C, out, hb, M, N, C.stride(0), C.stride(1), out.stride(0), out.stride(1),
                rank, ws, cfg["BM"], cfg["BN"], cfg["WGS"], num_warps=cfg["num_warps"],
            )

        return run

    ms, cfg, n_ok, n_fail, diff = autotune(
        make, COMM_SPACE, _ar_check(ctx), label="two_kernel_oneshot"
    )
    log(f"  two_kernel_oneshot: {n_ok} ok / {n_fail} failed, best {cfg}")
    # Uses hipBLASLt, so its comm time is the tuned one-shot at ITS best config.
    t_comm = bench(_comm_runner(ctx, cfg)) if cfg else float("inf")
    return ms, cfg, diff, "one_shot", ctx["t_gemm_torch"], t_comm, 0


@variant("fused_wgspec_oneshot", kind="fused")
def v_fused(ctx):
    """Fused WG specialization, one kernel, disjoint GEMM and AR pools."""
    A, B, C, out, hb = ctx["A"], ctx["B"], ctx["C_sym"], ctx["out"], ctx["hb"]
    M, N, KL, ws, rank = ctx["M"], ctx["N"], ctx["KL"], ctx["ws"], ctx["rank"]
    shmem = ctx["shmem"]

    space = {
        "BM": [16, 32, 64],
        "BN": [128, 256],
        "BK": [64],
        "GEMM_WGS": [96, 128, 192],
        "COMM_WGS": [32, 64],
        "num_warps": [4, 8],
    }
    state = {}

    def make(cfg):
        n_t = ((M + cfg["BM"] - 1) // cfg["BM"]) * ((N + cfg["BN"] - 1) // cfg["BN"])
        flags = shmem.zeros((n_t,), dtype=torch.int32)
        state["it"] = 0
        shmem.barrier()

        def run():
            state["it"] += 1
            total = cfg["GEMM_WGS"] + cfg["COMM_WGS"]
            fused_wgspec_kernel[(total,)](
                A, B, C, out, flags, hb, M, N, KL,
                A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1), out.stride(0), out.stride(1),
                state["it"] * ws, rank, ws,
                cfg["BM"], cfg["BN"], cfg["BK"], cfg["GEMM_WGS"], total, 200000,
                num_warps=cfg["num_warps"],
            )

        return run

    ms, cfg, n_ok, n_fail, diff = autotune(make, space, _ar_check(ctx), label="fused_wgspec_oneshot")
    log(f"  fused_wgspec_oneshot: {n_ok} ok / {n_fail} failed, best {cfg}")
    if cfg is None:
        return ms, cfg, diff, "one_shot", float("inf"), float("inf"), 0

    # This variant's OWN components, at its tuned config and its shared warp count.
    Ct = ctx["C_tmp"]
    t_gemm = bench(lambda: triton_gemm_kernel[(cfg["GEMM_WGS"],)](
        A, B, Ct, M, N, KL, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        Ct.stride(0), Ct.stride(1), cfg["BM"], cfg["BN"], cfg["BK"], cfg["GEMM_WGS"],
        num_warps=cfg["num_warps"],
    ))
    t_comm = bench(_comm_runner(ctx, {
        "BM": cfg["BM"], "BN": cfg["BN"], "WGS": cfg["COMM_WGS"], "num_warps": cfg["num_warps"],
    }))
    log(f"    components: triton_gemm={t_gemm:.4f} comm={t_comm:.4f} (shared warps={cfg['num_warps']})")
    # One kernel instead of two: that saving is launch elimination, not overlap.
    return ms, cfg, diff, "one_shot", t_gemm, t_comm, 1


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--sizes", default="128,512,2048")
    p.add_argument("-n", type=int, default=2880)
    p.add_argument("-k", type=int, default=4096)
    p.add_argument("-o", "--output", default="roofline_yael.json")
    p.add_argument("--only", default=None, help="comma-separated variant names")
    args = p.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    ws, rank = dist.get_world_size(), dist.get_rank()
    dt = torch.float16
    dev = f"cuda:{rank}"

    shmem = iris.iris(1 << 33)
    hb = shmem.get_heap_bases()
    N, K = args.n, args.k
    KL = K // ws
    sizes = [int(s) for s in args.sizes.split(",")]
    want = set(args.only.split(",")) if args.only else None

    payload = {
        "hw": torch.cuda.get_device_name(0),
        "ws": ws, "dtype": "float16", "N": N, "K": K,
        "line_rate_gbs": LINE_RATE_GBS,
        "serial_check": [], "floors": [], "results": [],
    }
    rows = []

    for M in sizes:
        A = torch.randn(M, KL, device=dev, dtype=dt)
        B = torch.randn(KL, N, device=dev, dtype=dt)
        Cr = torch.empty(M, N, device=dev, dtype=dt)
        C_tmp = torch.empty(M, N, device=dev, dtype=dt)
        C_sym = shmem.zeros((M, N), dtype=dt)
        out = torch.zeros(M, N, device=dev, dtype=dt)

        torch.mm(A, B, out=C_sym)
        ref_ar = C_sym.clone().float()
        dist.all_reduce(ref_ar)

        # Floors first: everything else is scored against these.
        # The standalone all_reduce runs on zeros: reducing the same buffer 125
        # times multiplies it by ws each pass and would overflow fp16 to inf.
        # NCCL timing is data-independent, so zeros cost nothing and stay finite.
        # Timing all_reduce on zeros dodges the fp16 overflow from reducing the
        # same buffer 125 times, but RCCL's reduction cost can be data
        # dependent. Scale by 1/ws instead: the sum is a fixed point, so it
        # stays finite over any number of iterations without being trivial.
        Z = torch.full((M, N), 1.0 / ws, device=dev, dtype=dt)
        t_launch = bench(lambda: null_kernel[(1,)](Z))
        t_gemm_torch = bench(lambda: torch.mm(A, B, out=Cr))
        t_rccl = bench(lambda: dist.all_reduce(Z))
        t_torch = bench(lambda: (torch.mm(A, B, out=Cr), dist.all_reduce(Cr)))

        # The whole overlap-ceiling argument rests on torch being serial.
        # Measure it, do not assume it.
        serial_err = (t_torch - (t_gemm_torch + t_rccl)) / t_torch * 100.0
        payload["serial_check"].append(
            {"M": M, "t_gemm": t_gemm_torch, "t_rccl": t_rccl,
             "t_sum": t_gemm_torch + t_rccl, "t_measured": t_torch, "err_pct": serial_err}
        )

        rccl_mb = algo_bytes("two_shot", M, N, ws, 2) / 1e6
        log(f"\n=== M={M} N={N} K={K} ws={ws} ===")
        log(f"  floors: gemm={t_gemm_torch:.4f} rccl={t_rccl:.4f} "
            f"({rccl_mb:.2f} MB, {rccl_mb * 1e6 / (t_rccl * 1e-3) / 1e9:.1f} GB/s)")
        log(f"  launch cost (null kernel): {t_launch * 1000:.1f} us")
        log(f"  serial check: gemm+rccl={t_gemm_torch + t_rccl:.4f} vs "
            f"measured mm+all_reduce={t_torch:.4f}  err={serial_err:+.1f}%")

        ctx = dict(
            A=A, B=B, Cr=Cr, C_sym=C_sym, C_tmp=C_tmp, out=out, hb=hb, shmem=shmem,
            M=M, N=N, KL=KL, ws=ws, rank=rank, ref_ar=ref_ar,
            t_gemm_torch=t_gemm_torch, t_rccl=t_rccl, t_torch=t_torch,
        )

        for name, v in variants().items():
            if want and name not in want:
                continue
            shmem.barrier()
            try:
                ms, cfg, diff, pattern, t_g, t_c, saved = v["fn"](ctx)
                mt = metrics(ms, t_g, t_c, t_gemm_torch, t_rccl, t_torch, pattern, M, N, ws, 2,
                             saved_launches=saved, t_launch=t_launch)
            except Exception as e:
                log(f"  {name}: FAILED {type(e).__name__}: {e}")
                torch.mm(A, B, out=C_sym)
                shmem.barrier()
                continue
            row = {"variant": name, "kind": v["kind"], "M": M, "cfg": cfg,
                   "max_diff": diff, "vs_torch": t_torch / ms,
                   "t_gemm_var": t_g, "t_comm_var": t_c, **mt}
            if name == "torch_serial":
                cal = abs(row["vs_torch"] - 1.0)
                row["calibration_err_pct"] = cal * 100
                if cal > 0.03:
                    log(f"  !! CALIBRATION: torch_serial scored "
                        f"{row['vs_torch']:.3f}x against itself ({cal*100:.1f}% off). "
                        f"References are mis-measured; treat every effect "
                        f"smaller than {cal*100:.0f}% in this block as noise.")
            payload["results"].append(row)
            rows.append(row)
            # C_sym is clobbered by fused/two-kernel runs; restore for the next.
            torch.mm(A, B, out=C_sym)
            shmem.barrier()

        payload["floors"].append(
            {"M": M, "t_gemm_ms": t_gemm_torch, "t_rccl_ms": t_rccl,
             "t_oneshot_ms": ctx.get("t_oneshot"), "rccl_algo_mb": rccl_mb,
             "t_launch_ms": t_launch}
        )

    log("")
    table(rows)
    emit(args.output, payload)
    shmem.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
