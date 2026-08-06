"""Shared roofline harness for GEMM+collective variants.

Every measured variant is decomposed so that "fast" and "slow" always name a
cause. The headline speedup factors exactly into four terms:

    vs_torch = serial_overhead * component_gain * launch_gain * true_overlap

    serial_overhead  how much torch's mm+all_reduce exceeds the sum of its parts
    component_gain   are our GEMM and our collective faster than torch's
    launch_gain      did fusion eliminate a kernel launch
    true_overlap     did we actually hide comm behind compute

component_gain is driven by gemm_gain and comm_gain, but is their weighted
harmonic mean, NOT their product -- it is a ratio of sums. Only comm_gain
factors cleanly, into algorithm (bytes moved) times kernel (how fast we move
them). All four identities are asserted at runtime; a row that does not add up
crashes rather than printing a plausible number.

The split between launch_gain and true_overlap matters: the small-M fused wins
in this study turned out to be launch-overhead elimination with zero overlap
behind them, and a single overlap number hid that for days.

Also reported per row: overlap_ratio (0 = serial, 1 = comm fully hidden) and
efficiency (T_ideal / T_meas). A variant can overlap perfectly and still be slow
because the kernel is bad, or be fast with no overlap because the collective is
better. Reporting only one of these conflates them.

Register a variant with @variant; it is autotuned over its config space and only
the tuned best enters the table. num_warps is mandatory in every config space --
it was worth 4.3x on the one-shot pull kernel and was unswept everywhere else.
"""

import itertools
import json
import os
import traceback

import torch
import torch.distributed as dist

LINE_RATE_GBS = 448.0  # MI355X XGMI per-GPU

_VARIANTS = {}


def variant(name, kind="fused"):
    """Register a variant. kind is one of: baseline, floor, comm, two_kernel, fused."""

    def deco(fn):
        _VARIANTS[name] = {"name": name, "kind": kind, "fn": fn}
        return fn

    return deco


def variants():
    return _VARIANTS


def is_rank0():
    return dist.get_rank() == 0


def log(*a):
    if is_rank0():
        print(*a, flush=True)


def bench(fn, iters=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def configs(space):
    """Cartesian product of a dict of lists -> list of dicts."""
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def algo_bytes(pattern, M, N, ws, itemsize):
    """Bytes each rank moves over the fabric, by algorithm.

    one_shot: every rank reads the full MxN partial from all ws peers.
    two_shot: reduce-scatter then all-gather, the ring/NCCL convention.
    """
    mn = M * N * itemsize
    if pattern == "one_shot":
        return ws * mn
    if pattern == "two_shot":
        return 2 * (ws - 1) / ws * mn
    raise ValueError(pattern)


def metrics(t_meas, t_gemm_var, t_comm_var, t_gemm_torch, t_comm_rccl,
            t_torch, pattern, M, N, ws, itemsize, saved_launches=0, t_launch=0.0):
    """Roofline numbers for one measured variant.

    The speedup factors exactly into three terms, so nothing is unattributed:

        T_torch / T_meas = serial_overhead * component_gain * overlap_gain

        serial_overhead = T_torch       / (T_gemm_torch + T_comm_rccl)
        component_gain  = T_serial_torch / T_serial_var
        overlap_gain    = T_serial_var  / T_meas

    serial_overhead is torch's own launch cost -- how far mm+all_reduce exceeds
    the sum of its parts. It is the measured basis for the claim that torch is
    serial, and it belongs in the product rather than being rounded away.

    component_gain does NOT factor into gemm_gain * comm_gain: it is a ratio of
    sums, and a ratio of sums is not a product of component ratios. It is the
    weighted harmonic mean of the two, weighted by how torch splits its time:

        gemm_gain = T_gemm_torch / T_gemm_var
        comm_gain = T_comm_rccl  / T_comm_var
        w_g, w_c  = torch's GEMM and comm share of T_serial_torch
        component_gain = 1 / (w_g / gemm_gain + w_c / comm_gain)

    That identity is asserted. Only comm_gain factors cleanly, into the bytes
    the algorithm moves and how fast the kernel moves them:

        algorithm = rccl_bytes  / variant_bytes
        kernel    = variant_GBs / rccl_GBs
        comm_gain = algorithm * kernel

    Keeping gemm_gain separate is what stops a Triton GEMM tax from hiding
    inside an aggregate.

    overlap_gain splits once more. T_serial_var is the sum of two separately
    launched kernels, so a fused variant saves a launch relative to its own
    serial reference. That saving is not overlap, and this study has already
    mistaken one for the other -- the small-M fused wins turned out to be
    launch-overhead elimination with zero overlap behind them:

        launch_gain  = T_serial_var / (T_serial_var - saved_launches * t_launch)
        true_overlap = overlap_gain / launch_gain

    launch_gain is 1.0 for any variant that still issues two kernels, so the
    factoring is unchanged for them and only fused rows are affected.
    """
    t_serial_var = t_gemm_var + t_comm_var
    t_ideal = max(t_gemm_var, t_comm_var)
    denom = t_serial_var - t_ideal
    # denom is 0 when one component is free; overlap is then undefined, not zero.
    overlap = (t_serial_var - t_meas) / denom if denom > 1e-9 else float("nan")

    ab_var = algo_bytes(pattern, M, N, ws, itemsize)
    ab_rccl = algo_bytes("two_shot", M, N, ws, itemsize)
    gbs_var = ab_var / (t_comm_var * 1e-3) / 1e9 if t_comm_var > 1e-9 else float("inf")
    gbs_rccl = ab_rccl / (t_comm_rccl * 1e-3) / 1e9

    t_serial_torch = t_gemm_torch + t_comm_rccl
    serial_overhead = t_torch / t_serial_torch
    component_gain = t_serial_torch / t_serial_var
    overlap_gain = t_serial_var / t_meas
    gemm_gain = t_gemm_torch / t_gemm_var if t_gemm_var > 1e-9 else float("inf")
    algorithm = ab_rccl / ab_var
    kernel = gbs_var / gbs_rccl
    comm_gain = t_comm_rccl / t_comm_var if t_comm_var > 1e-9 else float("inf")

    # A fused variant issues fewer kernels than its own serial reference; that
    # saving is launch elimination, not overlap.
    saved = saved_launches * t_launch
    launch_gain = t_serial_var / (t_serial_var - saved) if t_serial_var - saved > 1e-9 else float("inf")
    true_overlap = overlap_gain / launch_gain if launch_gain > 0 else float("nan")

    # This study has already produced three confident-and-wrong diagnoses.
    # Make the arithmetic fail loudly rather than print a plausible number.
    assert abs(serial_overhead * component_gain * overlap_gain - t_torch / t_meas) < 1e-6, \
        "serial_overhead * component_gain * overlap_gain != vs_torch"
    assert abs(launch_gain * true_overlap - overlap_gain) < 1e-9, \
        "launch_gain * true_overlap != overlap_gain"
    assert abs(algorithm * kernel - comm_gain) / comm_gain < 0.02, \
        "comm_gain != algorithm * kernel"
    # Overlap cannot beat perfect hiding. If it does, t_comm_var came from a
    # DIFFERENT implementation than the variant runs -- e.g. scoring a
    # barrier-free fused kernel against a standalone reference that pays a
    # host barrier. The difference is then credited as overlap, which is the
    # same mislabeling as the launch leak, one level up. Fail loudly.
    if denom > 1e-9:
        ceiling = t_serial_var / t_ideal
        assert true_overlap <= ceiling * 1.05, (
            f"true_overlap {true_overlap:.3f} exceeds the physical ceiling "
            f"{ceiling:.3f} -- t_comm_var ({t_comm_var:.4f}ms) is not the comm "
            f"cost of this variant's own implementation"
        )

    if t_gemm_var > 1e-9:
        w_g = t_gemm_torch / t_serial_torch
        w_c = t_comm_rccl / t_serial_torch
        blend = 1.0 / (w_g / gemm_gain + w_c / comm_gain)
        assert abs(blend - component_gain) / component_gain < 1e-6, \
            "component_gain is not the weighted harmonic mean of gemm_gain and comm_gain"

    return {
        "t_ms": t_meas,
        "t_serial_ms": t_serial_var,
        "t_ideal_ms": t_ideal,
        "overlap_ratio": overlap,
        "efficiency": t_ideal / t_meas,
        "algo_mb": ab_var / 1e6,
        "comm_gbs": gbs_var,
        "pct_line": gbs_var / LINE_RATE_GBS * 100.0,
        "serial_overhead": serial_overhead,
        "component_gain": component_gain,
        "overlap_gain": overlap_gain,
        "launch_gain": launch_gain,
        "true_overlap": true_overlap,
        "saved_launches": saved_launches,
        "gemm_gain": gemm_gain,
        "algorithm": algorithm,
        "kernel": kernel,
        "comm_gain": comm_gain,
    }


def autotune(fn, space, check, iters=50, warmup=15, label=""):
    """Sweep the config space, keep the fastest config that is correct.

    Returns (best_ms, best_cfg, n_ok, n_fail). A config that raises (register
    spill, bad tile, launch rejection) is skipped, not fatal -- config spaces
    are deliberately wide and some corners will not compile.
    """
    best_ms, best_cfg, n_ok, n_fail = float("inf"), None, 0, 0
    for cfg in configs(space):
        try:
            run = fn(cfg)
            run()
            torch.cuda.synchronize()
            ok, diff = check()
            if not ok:
                n_fail += 1
                continue
            ms = bench(run, iters=iters, warmup=warmup)
            n_ok += 1
            if ms < best_ms:
                best_ms, best_cfg, best_diff = ms, cfg, diff
        except Exception:
            n_fail += 1
            if os.environ.get("ROOFLINE_DEBUG"):
                log(f"  [{label}] cfg {cfg} failed:\n{traceback.format_exc()}")
            continue
    if best_cfg is None:
        return float("inf"), None, n_ok, n_fail, float("nan")
    return best_ms, best_cfg, n_ok, n_fail, best_diff


def emit(path, payload):
    if is_rank0():
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"\nwrote {path}")


def table(rows):
    """Render the roofline table.

    vs_torch = serial x comp x ovlap exactly. comp is driven by gemm and comm
    (harmonic blend, not a product), and comm = algo x kern. So a slow variant
    always names the factor that sank it.
    """
    hdr = (
        f"{'variant':<26} {'M':>5} {'ms':>9} {'vs torch':>9} | "
        f"{'serial':>6} {'comp':>6} {'lnch':>6} {'ovlap':>6} | "
        f"{'gemm':>6} {'comm':>6} {'algo':>6} {'kern':>6} | "
        f"{'GB/s':>7} {'%line':>6} {'diff':>7}"
    )
    log(hdr)
    log("  vs_torch = serial x comp x lnch x ovlap  (these four multiply exactly)")
    log("  comp is a harmonic blend of gemm and comm, NOT their product; comm = algo x kern")
    log("-" * len(hdr))
    for r in rows:
        log(
            f"{r['variant']:<26} {r['M']:>5} {r['t_ms']:>9.4f} {r['vs_torch']:>8.2f}x | "
            f"{r['serial_overhead']:>6.3f} {r['component_gain']:>6.2f} "
            f"{r['launch_gain']:>6.2f} {r['true_overlap']:>6.2f} | "
            f"{r['gemm_gain']:>6.2f} {r['comm_gain']:>6.2f} "
            f"{r['algorithm']:>6.2f} {r['kernel']:>6.2f} | "
            f"{r['comm_gbs']:>7.1f} {r['pct_line']:>5.0f}% {r['max_diff']:>7.4f}"
        )
