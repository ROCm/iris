# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Analytical CU-allocation predictor for the fused work-stealing GEMM+collective.

Uses the origami GEMM cost model and the origami comm cost model to pick the
GEMM/comm CU split (``gemm_wgs``) for ``iris.concurrent.gemm.*``.

Model (global work stealing => CUs are fungible workers; GEMM is compute-bound,
comm is xGMI-bandwidth-bound, so they overlap on disjoint resources):

    T_fused(gemm_wgs) ~= max( T_gemm(gemm_wgs), T_comm(num_wgs - gemm_wgs) )

with T_gemm(cus) from origami ``compute_total_latency`` and T_comm(wgs) from the
comm model ``predict_row``. The argmin over ``gemm_wgs`` is the predicted split;
global stealing makes the achieved makespan a bit better than this static max,
so it is used purely as a *ranker*.

Requires the origami extension (LD_PRELOAD a libstdc++ with GLIBCXX_3.4.32 if the
interpreter's libstdc++ is older than ROCm's).

This module is import-safe without that dependency; the cost functions raise a clear
error if their backend is missing.
"""

import functools

# ---- GEMM backend: origami -------------------------------------------------
try:
    import origami as _origami

    _HAVE_ORIGAMI = True
except Exception:  # pragma: no cover
    _HAVE_ORIGAMI = False

_ARCH = {"gfx942": "gfx942"}


@functools.lru_cache(maxsize=4)
def _hw(arch="gfx942", n_cu=304, clock_khz=1500000):
    if not _HAVE_ORIGAMI:
        raise ImportError("origami extension not importable")
    a = getattr(_origami.architecture_t, arch)
    return _origami.get_hardware_for_arch(a, n_cu, 65536, 4194304, clock_khz), clock_khz / 1e3  # (hw, clock_mhz)


def _cdiv(a, b):
    return (a + b - 1) // b


# ---- GEMM cost ------------------------------------------------------------
# origami full-GEMM is a near-CONSTANT 0.730x of measured hipBLASLt across the
# 4096-shape corpus (a compute-clock/frequency gap: origami assumes ~1.5 GHz,
# effective sustained ~0.73x). Correcting by 1/0.730 collapses the GEMM error
# from 27.2% -> 6.1% MdAPE; a shape-dependent (log-linear) fit does WORSE
# (overfits), confirming it is a constant, not shape structure. The residual
# ~+-20% per-shape scatter is intrinsic to origami. This factor is validated
# standalone against measured GEMM, so `FUSED_GEMM_MULT` below is now ONLY the
# fused-triton-vs-hipBLASLt overhead (previously it conflated the two).
ORIGAMI_GEMM_CAL = 1.37


@functools.lru_cache(maxsize=8192)
def gemm_full_ms(M, N, K, mt=(256, 256, 64), arch="gfx942", n_cu=304, clock_khz=1500000):
    """origami full-GEMM latency (ms) at full occupancy, times ORIGAMI_GEMM_CAL
    (the validated ~1.37x clock-gap correction: origami vs measured GEMM is a
    near-constant 0.73x; corrected -> 6.1% MdAPE across 4096 shapes)."""
    hw, clk_mhz = _hw(arch, n_cu, clock_khz)
    p = _origami.problem_t()
    p.size = _origami.dim3_t(M, N, K)
    p.batch = 1
    p.a_transpose = _origami.transpose_t.T
    p.b_transpose = _origami.transpose_t.N
    for at in ("a_dtype", "b_dtype", "c_dtype", "d_dtype", "mi_dtype"):
        setattr(p, at, _origami.data_type_t.Half)
    c = _origami.config_t()
    c.mt = _origami.dim3_t(*mt)
    c.mi = _origami.dim3_t(16, 16, 16)
    c.occupancy = 2
    c.workgroup_mapping = 8
    return ORIGAMI_GEMM_CAL * _origami.compute_total_latency(p, hw, c, int(n_cu)) / (clk_mhz * 1e3)


def gemm_tile_ms(M, N, K, mt=(256, 256, 64), arch="gfx942", n_cu=304, clock_khz=1500000):
    """Per-tile (one-workgroup) GEMM node latency (ms), derived from the
    CALIBRATED full-GEMM number: T_full * n_cu / n_tiles (CU-seconds per tile),
    so `n_tiles` such nodes over `n_cu` workers reproduce T_full. NOTE: pricing a
    1-tile problem directly underestimates ~10x (origami gives it full-chip
    resources), so we scale down from the full GEMM instead."""
    n_tiles = _cdiv(M, mt[0]) * _cdiv(N, mt[1])
    return gemm_full_ms(M, N, K, tuple(mt), arch, n_cu, clock_khz) * n_cu / max(1, n_tiles)


# ---- fused-kernel GEMM interference model ---------------------------------
# Trace-area calibration: the per-tile GEMM cost INSIDE the fused kernel is
# higher than standalone origami and grows with how much comm bandwidth competes
# with it (GEMM and comm contend for HBM/xGMI):
#   * FUSED_GEMM_MULT: fused-triton-vs-(clock-corrected-origami) overhead. The
#     old 1.97 conflated the origami clock gap (now folded into ORIGAMI_GEMM_CAL
#     = 1.37) with the genuine fused overhead, so 1.97/1.37 = 1.44 is the pure
#     fused-triton overhead over origami-after-calibration.
#   * GEMM_INTERFERENCE_ALPHA: interference rises with the comm-home worker
#     fraction f_comm AND with comm VOLUME/INTENSITY = comm_full/gemm_full (how
#     bandwidth-heavy the comm is). Per-collective trace areas showed AG/BC
#     (comm=full tiles, N-1 transfers each => high intensity) have a strong
#     GEMM-area slope, while AR/RS (comm=full/W tiles => low intensity) have an
#     almost FLAT GEMM area -- so interference MUST scale with comm volume, not
#     just worker count. Refit with the calibrated gemm_full: a ~= 0.685.
# Lg_eff = origami_tile * FUSED_GEMM_MULT * (1 + ALPHA * f_comm * comm_intensity).
FUSED_GEMM_MULT = 1.44
GEMM_INTERFERENCE_ALPHA = 0.40

# Comm is xGMI-bandwidth-bound and saturates at a small CU count -- beyond this
# extra channels/stealers add contention, not throughput (MI300X sweet spots:
# RS/A2A ~32, AllReduce ~60, AllGather ~64; see mi300x-architecture). This is
# the effective comm concurrency: comm CU-area ~= comm_full * COMM_SAT_CHANNELS
# (not * num_wgs), which matches the measured comm trace area at the optimum.
COMM_SAT_CHANNELS = 64


def _comm_sat_channels(collective, world, base=COMM_SAT_CHANNELS, num_wgs=None):
    """CUs comm needs to SATURATE xGMI, scaled by per-tile transfer count (the
    physics, not a per-collective fit): a tile that pushes/pulls more remote
    blocks needs more CUs to hit link bandwidth. Relative to all_gather's N-1
    transfers/tile: all_reduce = 2(N-1) (read-all + write-all) -> 2x; all_to_all
    = 1 (one peer) -> ~1/(N-1)x; reduce_scatter/broadcast = N-1 -> 1x. This
    penalizes starving a heavy-comm collective (AR) of CUs -- the fix for the
    measured gw=272/comm_wgs=32 cliff the model otherwise walks into -- WITHOUT
    over-flooring the light A2A (which a blanket cap did)."""
    # Only all_reduce needs the floor: its comm is FEW tiles (full/W) each HEAVY
    # (2(N-1) read+write transfers), so starving it to ~32 CUs is catastrophic
    # (measured 8-13x cliff at gw=272). AG/RS/BC/A2A have many light comm tiles,
    # so a floor there only mis-shifts their (already good) picks -- an A/B on the
    # fine grid showed a blanket floor drops AG exact 78->65%, BC 75->63%. So
    # floor AR at 2x base; leave the rest unfloored.
    floor = int(round(base * 2.0)) if collective == "all_reduce" else 1
    if num_wgs is not None:
        floor = min(floor, num_wgs)
    return max(1, floor)


def gemm_tile_ms_fused(
    M,
    N,
    K,
    num_wgs,
    gemm_wgs,
    comm_intensity=0.0,
    mt=(256, 256, 64),
    arch="gfx942",
    clock_khz=1500000,
    fused_mult=FUSED_GEMM_MULT,
    alpha=GEMM_INTERFERENCE_ALPHA,
):
    """Interference-aware per-tile GEMM cost (ms) for the FUSED kernel: the
    origami CU-seconds tile, scaled by the fused-occupancy multiplier and by a
    term linear in the concurrent-comm fraction TIMES the comm intensity
    (``comm_intensity = comm_full/gemm_full`` -- comm's bandwidth demand). A
    comm-light collective (AR/RS, small comm_intensity) barely slows GEMM even
    with many comm-home workers; a bandwidth-heavy one (AG/BC) slows it a lot.
    See FUSED_GEMM_MULT / GEMM_INTERFERENCE_ALPHA."""
    base = gemm_tile_ms(M, N, K, tuple(mt), arch, num_wgs, clock_khz)
    f_comm = max(0, num_wgs - gemm_wgs) / max(1, num_wgs)
    return base * fused_mult * (1.0 + alpha * f_comm * max(0.0, comm_intensity))


# Effective per-GPU xGMI bandwidth (GB/s) the fused comm achieves at saturation.
# ONE hardware constant for ALL collectives -- the per-collective cost difference
# comes entirely from the real byte counts in `comm_remote_bytes_per_rank`, not
# from per-collective tuning. MI300X has 7 xGMI links ~= 47 GB/s each (~330 GB/s
# aggregate egress); calibrated on the traced 8192x4608 all_gather (528 MB/rank
# over ~1.8 ms comm wall => ~293 GB/s).
# 120, not the ~300 tail bandwidth: comm shares HBM/xGMI with GEMM during
# overlap, so its EFFECTIVE fused bandwidth is well below the uncontended
# roofline (the comm-side of the GEMM<->comm interference). Two corpus regret
# sweeps (parallel hypothesis subagents) found comm ~2.5x under-priced as the
# DOMINANT regret lever: 300->120 shifts the predicted split toward more comm
# CUs and matches the measured optima (exact-pick 46->57%, >=8w 52->72%, median
# 1.65->0.0%, mean_abs 656->597us). NOTE: this over-prices the truly uncontended
# tail, so large comm-heavy shapes (AG/BC) get higher absolute makespan error; an
# overlap-GATED variant is physically cleaner but did not reproduce the ranking
# win (the tail over-pricing is part of what corrects the corpus ranking).
XGMI_BW_GBPS = 120.0
COMM_LAUNCH_US = 5.0  # fixed per-collective launch/sync floor (small-msg regime)


def comm_remote_bytes_per_rank(collective, comm_m, comm_n, world, cb=(256, 64), dtype_bytes=2):
    """Per-rank REMOTE (xGMI) bytes moved by the fused kernel for this collective,
    read straight from the kernel's tile traffic (`_kernels.py`):

      all_gather   : `full` tiles, each pushed to N-1 peers        -> full·(N-1)
      all_reduce   : full/W tiles owned, each reads N-1 + writes N-1 -> (full/W)·2(N-1)
      reduce_scatter: full/W output tiles, each reads N-1 remote    -> (full/W)·(N-1)
      all_to_all   : full tiles, each to ONE peer ((N-1)/N remote)  -> full·(N-1)/N
      broadcast    : root pushes `full` tiles to N-1 peers          -> full·(N-1)

    (tb = BM·BN·dtype = one comm tile's bytes). This is the physics that
    distinguishes the collectives -- no per-collective fit constant."""
    tb = cb[0] * cb[1] * dtype_bytes
    full = _cdiv(comm_m, cb[0]) * _cdiv(comm_n, cb[1])
    W = max(1, world)
    Nm1 = W - 1
    if collective == "all_reduce":
        return _cdiv(full, W) * 2 * Nm1 * tb
    if collective == "reduce_scatter":
        return _cdiv(full, W) * Nm1 * tb
    if collective == "all_to_all":
        return full * Nm1 // W * tb
    # all_gather, broadcast (root)
    return full * Nm1 * tb


@functools.lru_cache(maxsize=8192)
def comm_full_ms(collective, comm_m, comm_n, world, wgs, dtype_bytes=2, oneshot=True):
    """Full-collective wall-time (ms) for the FUSED kernel, as a byte-roofline:
    (per-rank remote xGMI bytes) / (effective xGMI bandwidth) + launch floor.

    Bandwidth-bound and saturating in channel count (`wgs` unused -- comm is
    already saturated at `COMM_SAT_CHANNELS`), so the per-collective cost is set
    purely by the real traffic `comm_remote_bytes_per_rank`, not the RCCL comm
    model (which prices ring algorithms the fused kernel does not run) nor any
    per-collective constant."""
    B = comm_remote_bytes_per_rank(collective, comm_m, comm_n, world, dtype_bytes=dtype_bytes)
    return B / (XGMI_BW_GBPS * 1e9) * 1e3 + COMM_LAUNCH_US / 1e3


def _n_comm_tiles(collective, comm_m, comm_n, world, cb=(256, 64)):
    """Number of comm work items in the kernel's queue for this collective."""
    full = _cdiv(comm_m, cb[0]) * _cdiv(comm_n, cb[1])
    if collective == "all_reduce":
        return _cdiv(full, world)  # block-partitioned per rank
    if collective == "reduce_scatter":
        return _cdiv(comm_m // world, cb[0]) * _cdiv(comm_n, cb[1])
    return full  # all_gather / all_to_all / broadcast


# ---- work-stealing scheduler ----------------------------------------------
# Measured contended dequeue latency of the single device-wide atomic_add per
# queue (304 WGs sharing one counter), p50, ms. See iris_fused_tracing:
#   GEMM p50 3960 ns, comm p50 9440 ns (comm tiles are finer => 2x more
#   contention). For comm the per-item EXEC cost is ~0.8 us, so this ~9.4 us
#   dequeue DOMINATES each comm work item ~10x -- omitting it is the other half
#   of the absolute under-prediction (alongside pricing the ring vs one-shot).
# Measured p50 dequeue latencies are 3.96/9.44 us, but the effective slice on
# the makespan-critical path is much smaller (regret sweeps put the sweet spot
# at ~0.25x the p50; zeroing overshoots) -- the p50 over-counts because most
# pulls overlap other work. So use ~1.0/2.4 us.
C_ATOMIC_GEMM_MS = 0.99e-3
C_ATOMIC_COMM_MS = 2.36e-3


def schedule_makespan(
    n_gemm, Lg, n_comm, Lc_unit, num_wgs, gemm_wgs, c_atomic_gemm=0.0, c_atomic_comm=None, comm_cap=None, comm_sat=1
):
    """Discrete-event simulation of the fused kernel's GLOBAL work-stealing over
    two queues. `gemm_wgs` workers home on the GEMM queue, the rest on comm;
    an idle worker steals the other. Returns makespan (ms).

    Comm is xGMI-BANDWIDTH-BOUND: `k` workers doing comm concurrently SHARE the
    fixed link bandwidth, so each comm tile costs ``k * Lc_unit`` where
    ``Lc_unit = comm_full / n_comm``. This single physical rule (no cap, no
    per-collective constant) gives both:
      * wall-time = comm_full regardless of how many workers pile on (bandwidth),
      * comm CU-area = comm_full * (avg concurrent comm workers) -- small when
        comm overlaps GEMM (optimal split), large in the serial tail
        (over-subscription) -> reproduces the measured comm-area 'V'.
    GEMM tiles are compute-bound (fixed `Lg`, parallel). Each dequeue also pays
    its queue's contended atomic latency (`c_atomic_*`). `comm_cap` is accepted
    for API compat but ignored (the bandwidth-sharing rule subsumes it)."""
    import heapq

    if c_atomic_comm is None:
        c_atomic_comm = c_atomic_gemm
    g_left, c_left = n_gemm, n_comm
    active_comm = 0
    # worker heap entries: (free_time, worker, last_task)  last_task in {"g","c","-"}
    h = [(0.0, w, "-") for w in range(num_wgs)]
    heapq.heapify(h)
    home_of = lambda w: 0 if w < gemm_wgs else 1
    makespan = 0.0
    while g_left > 0 or c_left > 0:
        t, w, last = heapq.heappop(h)
        if last == "c":
            active_comm -= 1
        # prefer home queue, then steal the other
        if home_of(w) == 0:
            order = [("g", g_left > 0), ("c", c_left > 0)]
        else:
            order = [("c", c_left > 0), ("g", g_left > 0)]
        pick = next((k for k, ok in order if ok), None)
        if pick == "g":
            g_left -= 1
            nt = t + Lg + c_atomic_gemm
            last = "g"
        elif pick == "c":
            c_left -= 1
            active_comm += 1
            # bandwidth shared among the `active_comm` concurrent comm workers
            # (Lc_unit already reflects the fused effective xGMI bandwidth).
            # `comm_sat` (default 1 = off) floors the effective worker count:
            # comm needs >= comm_sat CUs to saturate xGMI; below that it's
            # CU-limited (models the low-comm_wgs starvation cliff).
            nt = t + max(active_comm, comm_sat) * Lc_unit + c_atomic_comm
            last = "c"
        else:
            continue
        makespan = max(makespan, nt)
        heapq.heappush(h, (nt, w, last))
    return makespan


def _best_split_for_tile(
    M,
    N,
    K,
    cm,
    cn,
    coll,
    W,
    num_wgs,
    candidates,
    mt,
    cb,
    c_atomic_gemm,
    c_atomic_comm,
    oneshot,
    comm_sat,
):
    """Best CU split for ONE fixed GEMM tile `mt`. Returns
    (gw, makespan, t_gemm, comm_full, Lc_unit, Lg, n_gemm, curve)."""
    n_gemm = _cdiv(M, mt[0]) * _cdiv(N, mt[1])
    n_comm = _n_comm_tiles(coll, cm, cn, W, cb)
    comm_full_sat = comm_full_ms(coll, cm, cn, W, num_wgs, oneshot=oneshot)
    Lc_unit = comm_full_sat / max(1, n_comm)
    comm_intensity = comm_full_sat / max(1e-9, gemm_full_ms(M, N, K, tuple(mt), n_cu=num_wgs))
    curve = []
    best = None
    csat = _comm_sat_channels(coll, W, base=comm_sat, num_wgs=num_wgs)
    for gw in candidates:
        Lg = gemm_tile_ms_fused(M, N, K, num_wgs, gw, comm_intensity, tuple(mt))
        t_gemm = Lg * _cdiv(n_gemm, gw)
        mk = schedule_makespan(
            n_gemm,
            Lg,
            n_comm,
            Lc_unit,
            num_wgs,
            gw,
            c_atomic_gemm=c_atomic_gemm,
            c_atomic_comm=c_atomic_comm,
            comm_sat=csat,
        )
        curve.append((gw, t_gemm, comm_full_sat, mk))
        if best is None or mk < best[1]:
            best = (gw, mk, t_gemm, comm_full_sat, Lc_unit, Lg)
    return (*best, n_gemm, curve)


def predict_split(
    shape,
    num_wgs=304,
    candidates=None,
    mt=(256, 256, 64),
    cb=(256, 64),
    c_atomic_gemm=C_ATOMIC_GEMM_MS,
    c_atomic_comm=C_ATOMIC_COMM_MS,
    oneshot=True,
    comm_sat=COMM_SAT_CHANNELS,
    mt_candidates=None,
    mt_prefer=(256, 256, 64),
    mt_margin=0.20,
):
    """Predict the best CU split ``gemm_wgs`` (and, if ``mt_candidates`` is given,
    the best GEMM tile shape) for one shape via the work-stealing scheduler.

    The absolute makespan comes from the discrete-event scheduler
    (``schedule_makespan``) with the one-shot comm layout + atomic dequeue.

    ``mt_candidates``: optional list of GEMM tile shapes ``(BM, BN, BK)`` to
    CO-SELECT alongside the split (origami is tile-aware). The joint-param sweep
    showed the split is the dominant knob but tile choice helps ~23% of shapes
    (skinny/small GEMMs where 256x256 is oversized). Defaults to ``[mt]`` (split
    only, backward compatible).

    shape = dict(gemm_m, gemm_n, gemm_k, comm_m, comm_n, collective, world)
    Returns dict(gemm_wgs, mt, pred_ms, n_gemm, Lg, Lc, curve).
    """
    M, N, K = shape["gemm_m"], shape["gemm_n"], shape["gemm_k"]
    cm, cn = shape["comm_m"], shape["comm_n"]
    coll, W = shape["collective"], shape["world"]
    if candidates is None:
        candidates = list(range(max(16, num_wgs // 8), num_wgs + 1, 8))
    tiles = [tuple(t) for t in (mt_candidates if mt_candidates else [mt])]
    # best split per candidate tile
    per_tile = {}
    for tmt in tiles:
        per_tile[tmt] = _best_split_for_tile(
            M,
            N,
            K,
            cm,
            cn,
            coll,
            W,
            num_wgs,
            candidates,
            tmt,
            cb,
            c_atomic_gemm,
            c_atomic_comm,
            oneshot,
            comm_sat,
        )  # (gw, mk, tg, cf, lc, lg, ng, curve)
    # margin gate: keep the preferred/default tile unless another is predicted
    # clearly (> mt_margin) faster. Expanding the tile space rescues the
    # skinny-shape tail, but the tile ranking is noisy on shapes where the
    # default is already fine, so an unconditional argmin over tiles hurts the
    # median. The gate keeps default for the majority and only switches on a
    # confident win.
    pref = tuple(mt_prefer)
    chosen = min(per_tile, key=lambda t: per_tile[t][1])  # global argmin tile (by makespan)
    if pref in per_tile and chosen != pref:
        if per_tile[chosen][1] >= per_tile[pref][1] * (1.0 - mt_margin):
            chosen = pref  # not a confident win -> stay default
    gw, mk, tg, cf, lc, lg, ng, curve = per_tile[chosen]
    return {
        "gemm_wgs": gw,
        "mt": chosen,
        "pred_ms": mk,
        "T_gemm": tg,
        "T_comm": cf,
        "n_gemm": ng,
        "Lg": lg,
        "Lc": lc,
        "curve": curve,
    }
