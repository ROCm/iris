#!/usr/bin/env python3
"""Kenji's variants for the shared roofline harness.

The metric math lives in roofline_core and is not duplicated here. Importing
roofline_yael pulls in its registered variants too, so one run produces one
table covering both of us -- the @variant decorator writes into a shared
registry and roofline_yael.main() iterates all of it.

Adds the two rows that harness is missing:

  twoshot_ar_only    tuned standalone RS+AG all-reduce. This is the comm
                     reference the fused two-shot must be scored against;
                     without it the fused row would be compared to a one-shot
                     reference and would book the algorithm switch as overlap.

  fused_hbm_twoshot  fused GEMM + two-shot AR, three disjoint WG pools with
                     HBM staging. saved_launches=1: it issues one kernel where
                     its own serial reference issues two, and that saving is
                     launch elimination, not overlap.

Run (8 ranks):
    torchrun --nproc_per_node=8 tests/ops/roofline_kenji.py -m 128,512,2048
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from roofline_core import LINE_RATE_GBS, autotune, bench, log, variant

# Importing the sibling runner registers its variants and gives us main().
import roofline_yael as ry


@triton.jit
def _gemm_pool_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """The fused kernel's GEMM pool, standalone.

    Needed so gemm_gain measures the Triton GEMM the fused path actually runs.
    Substituting torch.mm here would hide the Triton GEMM tax inside
    component_gain -- the term this study already mis-blamed once.
    """
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_SIZE_N)
    total = tl.cdiv(M, BLOCK_SIZE_M) * num_n
    for tid in range(pid, total, NUM_SMS):
        pid_m, pid_n = tid // num_n, tid % num_n
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        rk = tl.arange(0, BLOCK_SIZE_K)
        ap = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        bp = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            rem = K - k * BLOCK_SIZE_K
            acc += tl.dot(tl.load(ap, mask=rk[None, :] < rem, other=0.0),
                          tl.load(bp, mask=rk[:, None] < rem, other=0.0))
            ap += BLOCK_SIZE_K * stride_ak
            bp += BLOCK_SIZE_K * stride_bk
        off = rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptr + off, acc.to(c_ptr.type.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


# num_warps is mandatory in every space -- it was worth 4.3x on the one-shot
# kernel and was unswept on every fused variant in this study.
TWOSHOT_SPACE = {
    "BM": [32, 64, 128, 256],
    "BN": [64, 128],
    "WGS": [32, 64, 128, 196, 256],
    "num_warps": [1, 2, 4, 8, 16],
}

FUSED_SPACE = {
    "BM": [16, 32, 64, 128],
    "BN": [64, 128],
    "num_warps": [1, 2, 4, 8, 16],
    "split": [(192, 32, 32), (128, 64, 64), (96, 96, 64), (64, 96, 96)],
    "tpf": [1, 2, 4],
}


@variant("twoshot_ar_only", kind="comm")
def v_twoshot_comm(ctx):
    """Tuned standalone two-shot (RS+AG) all-reduce. The comm reference."""
    from iris.ops.all_reduce_fast import two_shot_all_reduce

    shmem, C, out = ctx["shmem"], ctx["C_sym"], ctx["out"]
    M, ws = ctx["M"], ctx["ws"]
    state = {"scratch": None}

    def make(cfg):
        if (M // ws) % cfg["BM"]:
            raise ValueError("shard not divisible by BM")

        def run():
            state["scratch"] = two_shot_all_reduce(
                shmem, out, C, scratch=state["scratch"],
                block_m=cfg["BM"], block_n=cfg["BN"],
                num_sms=cfg["WGS"], num_warps=cfg["num_warps"])

        return run

    ms, cfg, n_ok, n_fail, diff = autotune(
        make, TWOSHOT_SPACE, ry._ar_check(ctx), label="twoshot_ar_only")
    log(f"  twoshot_ar_only: {n_ok} ok / {n_fail} failed, best {cfg}")
    ctx["twoshot_cfg"] = cfg
    ctx["t_twoshot"] = ms
    # Pure comm measurement: no GEMM in it, so the compute term is zero.
    return ms, cfg, diff, "two_shot", 0.0, ms, 0


@variant("fused_hbm_twoshot", kind="fused")
def v_fused_hbm(ctx):
    """Fused GEMM + two-shot AR, 3 disjoint WG pools with HBM staging.

    A single-pool two-shot deadlocks: all-gather needs the reduce-scatter
    output of every rank, so a phase-2 WG can block the phase-1 WG that would
    unblock it. Three pools with a staging buffer make the dependency graph
    linear, which is what lets two-shot run without a host barrier at all.
    """
    from iris.ops.matmul_all_reduce_hbm_buffer import (
        matmul_all_reduce_hbm_buffer,
        matmul_all_reduce_hbm_buffer_preamble,
    )

    shmem, A, B, out = ctx["shmem"], ctx["A"], ctx["B"], ctx["out"]
    M, N, KL, ws = ctx["M"], ctx["N"], ctx["KL"], ctx["ws"]
    dt = A.dtype
    wcache = {}

    def make(cfg):
        if triton.cdiv(M, cfg["BM"]) % ws:
            raise ValueError("M-tiles not divisible across ranks")
        key = (cfg["BM"], cfg["BN"])
        if key not in wcache:
            wcache[key] = matmul_all_reduce_hbm_buffer_preamble(
                shmem, M, N, dt, cfg["BM"], cfg["BN"])
            shmem.barrier()
        wsp = wcache[key]
        g, r, a = cfg["split"]

        def run():
            matmul_all_reduce_hbm_buffer(
                shmem, out, A, B, workspace=wsp,
                block_m=cfg["BM"], block_n=cfg["BN"], block_k=64,
                num_gemm_sms=g, num_rs_sms=r, num_ag_sms=a,
                num_warps=cfg["num_warps"], mfma=32, tiles_per_flag=cfg["tpf"])

        return run

    ms, cfg, n_ok, n_fail, diff = autotune(
        make, FUSED_SPACE, ry._ar_check(ctx), label="fused_hbm_twoshot")
    log(f"  fused_hbm_twoshot: {n_ok} ok / {n_fail} failed, best {cfg}")
    if cfg is None:
        raise RuntimeError("no valid fused config")

    # Measure the components at the winning config so gemm_gain reflects the
    # Triton GEMM this variant actually runs, not hipBLASLt.
    Cg = torch.empty(M, N, device=A.device, dtype=dt)
    g = cfg["split"][0]
    t_gemm_var = bench(lambda: _gemm_pool_kernel[(g,)](
        A, B, Cg, M, N, KL,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        Cg.stride(0), Cg.stride(1),
        BLOCK_SIZE_M=cfg["BM"], BLOCK_SIZE_N=cfg["BN"], BLOCK_SIZE_K=64,
        NUM_SMS=g, num_warps=cfg["num_warps"], matrix_instr_nonkdim=32))

    # Comm reference must be THIS kernel's comm, not the closest standalone we
    # have. twoshot_ar_only pays a host barrier the fused path never pays, and
    # scoring against it credits that saving as overlap -- which the ceiling
    # assert in roofline_core now catches. SKIP_GEMM drops only the math and
    # keeps the barriers, launch count and flag protocol identical.
    g, r, a = cfg["split"]
    key = (cfg["BM"], cfg["BN"])
    t_comm_var = bench(lambda: matmul_all_reduce_hbm_buffer(
        shmem, out, A, B, workspace=wcache[key],
        block_m=cfg["BM"], block_n=cfg["BN"], block_k=64,
        num_gemm_sms=g, num_rs_sms=r, num_ag_sms=a,
        num_warps=cfg["num_warps"], mfma=32, tiles_per_flag=cfg["tpf"],
        skip_gemm=True))
    log(f"  fused_hbm_twoshot: comm-only (SKIP_GEMM) {t_comm_var:.4f}ms "
        f"vs standalone twoshot {ctx.get('t_twoshot', float('nan')):.4f}ms")

    # One kernel where the serial reference issues two.
    return ms, cfg, diff, "two_shot", t_gemm_var, t_comm_var, 1


if __name__ == "__main__":
    ry.main()
