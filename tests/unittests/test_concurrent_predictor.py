# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for iris.concurrent.predictor (CU-allocation cost model).

Covers the pure-Python cost/scheduler logic (no GPU): per-collective comm byte
counts, comm-tile counts, the byte-roofline comm cost, the work-stealing
bandwidth-sharing scheduler, and the comm-saturation floor. Tests that need the
origami GEMM extension (gemm_full_ms / gemm_tile_ms_fused / predict_split) are
skipped when it is unavailable.
"""

import os

import pytest

# The predictor is pure Python (origami/comm-model are lazy/optional) and does
# not need the full iris+triton+tritonblas stack. Prefer the package import, but
# fall back to loading the module by path so these logic tests run in a minimal
# (CPU/CI) environment too.
try:
    from iris.concurrent import predictor as P
except Exception:
    import importlib.util

    _pp = os.path.join(os.path.dirname(__file__), "..", "..", "iris", "concurrent", "predictor.py")
    _spec = importlib.util.spec_from_file_location("iris_concurrent_predictor", os.path.abspath(_pp))
    P = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(P)

requires_origami = pytest.mark.skipif(not getattr(P, "_HAVE_ORIGAMI", False), reason="origami extension not importable")

W = 8
CM, CN = 8192, 4608
CB = (256, 64)
TB = CB[0] * CB[1] * 2  # one comm tile's bytes (bf16/fp16)


def _full(cm=CM, cn=CN, cb=CB):
    return P._cdiv(cm, cb[0]) * P._cdiv(cn, cb[1])


# ---------------------------------------------------------------------------
# comm byte counts (the physics that distinguishes collectives)
# ---------------------------------------------------------------------------
def test_comm_bytes_all_gather_equals_broadcast():
    ag = P.comm_remote_bytes_per_rank("all_gather", CM, CN, W)
    bc = P.comm_remote_bytes_per_rank("broadcast", CM, CN, W)
    assert ag == bc == _full() * (W - 1) * TB


def test_comm_bytes_all_reduce_is_twice_reduce_scatter():
    ar = P.comm_remote_bytes_per_rank("all_reduce", CM, CN, W)
    rs = P.comm_remote_bytes_per_rank("reduce_scatter", CM, CN, W)
    # AR = (full/W)*2(N-1), RS = (full/W)*(N-1)
    assert ar == 2 * rs
    assert rs == P._cdiv(_full(), W) * (W - 1) * TB


def test_comm_bytes_all_to_all_smallest():
    a2a = P.comm_remote_bytes_per_rank("all_to_all", CM, CN, W)
    ag = P.comm_remote_bytes_per_rank("all_gather", CM, CN, W)
    assert a2a == _full() * (W - 1) // W * TB
    assert a2a < ag  # one transfer/tile vs N-1


def test_comm_bytes_scale_with_world():
    b4 = P.comm_remote_bytes_per_rank("all_gather", CM, CN, 4)
    b8 = P.comm_remote_bytes_per_rank("all_gather", CM, CN, 8)
    assert b8 > b4  # more peers -> more bytes


# ---------------------------------------------------------------------------
# comm tile counts (must match the kernel's work-item partitioning)
# ---------------------------------------------------------------------------
def test_n_comm_tiles_partitioning():
    full = _full()
    assert P._n_comm_tiles("all_gather", CM, CN, W, CB) == full
    assert P._n_comm_tiles("all_to_all", CM, CN, W, CB) == full
    assert P._n_comm_tiles("broadcast", CM, CN, W, CB) == full
    assert P._n_comm_tiles("all_reduce", CM, CN, W, CB) == P._cdiv(full, W)
    assert P._n_comm_tiles("reduce_scatter", CM, CN, W, CB) == P._cdiv(CM // W, CB[0]) * P._cdiv(CN, CB[1])


# ---------------------------------------------------------------------------
# byte-roofline comm cost
# ---------------------------------------------------------------------------
def test_comm_full_ms_positive_and_bytes_proportional():
    ag = P.comm_full_ms("all_gather", CM, CN, W, 304)
    rs = P.comm_full_ms("reduce_scatter", CM, CN, W, 304)
    assert ag > 0 and rs > 0
    # subtract the fixed launch floor, then it should track the byte ratio
    launch = P.COMM_LAUNCH_US / 1e3
    ratio_time = (ag - launch) / (rs - launch)
    ratio_bytes = P.comm_remote_bytes_per_rank("all_gather", CM, CN, W) / P.comm_remote_bytes_per_rank(
        "reduce_scatter", CM, CN, W
    )
    assert ratio_time == pytest.approx(ratio_bytes, rel=1e-6)


def test_comm_full_ms_channel_independent():
    # comm is bandwidth-bound: the `wgs` arg does not change the cost
    a = P.comm_full_ms("all_gather", CM, CN, W, 32)
    b = P.comm_full_ms("all_gather", CM, CN, W, 304)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# per-collective comm-saturation floor
# ---------------------------------------------------------------------------
def test_comm_sat_only_all_reduce():
    assert P._comm_sat_channels("all_reduce", W) == 2 * P.COMM_SAT_CHANNELS
    assert P._comm_sat_channels("all_reduce", W, num_wgs=64) == 64
    for c in ("all_gather", "reduce_scatter", "broadcast", "all_to_all"):
        assert P._comm_sat_channels(c, W) == 1


def test_best_split_uses_comm_sat_override(monkeypatch):
    seen = []
    monkeypatch.setattr(P, "comm_full_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(P, "gemm_full_ms", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(P, "gemm_tile_ms_fused", lambda *args, **kwargs: 1.0)

    def schedule(*args, **kwargs):
        seen.append(kwargs["comm_sat"])
        return 1.0

    monkeypatch.setattr(P, "schedule_makespan", schedule)
    P._best_split_for_tile(
        256,
        256,
        64,
        256,
        64,
        "all_reduce",
        W,
        80,
        [40],
        (256, 256, 64),
        CB,
        0.0,
        0.0,
        True,
        10,
    )
    assert seen == [20]


# ---------------------------------------------------------------------------
# work-stealing scheduler
# ---------------------------------------------------------------------------
def test_schedule_gemm_only_wave_quantized():
    # all workers home GEMM, no comm: makespan = ceil(n_gemm/num_wgs) * Lg
    Lg = 1.0
    assert P.schedule_makespan(304, Lg, 0, 0.0, 304, 304) == pytest.approx(1.0)
    assert P.schedule_makespan(608, Lg, 0, 0.0, 304, 304) == pytest.approx(2.0)
    assert P.schedule_makespan(305, Lg, 0, 0.0, 304, 304) == pytest.approx(2.0)


def test_schedule_comm_is_bandwidth_bound():
    # comm-only: more workers should NOT linearly speed it up (bandwidth-bound).
    # Doubling num_wgs keeps the makespan ~constant, not halved.
    Lc = 1.0
    mk_304 = P.schedule_makespan(0, 0.0, 600, Lc, 304, 0)
    mk_152 = P.schedule_makespan(0, 0.0, 600, Lc, 152, 0)
    assert 0.7 < mk_304 / mk_152 < 1.4  # ~constant (bandwidth), not 0.5


def test_schedule_comm_sat_floor_penalizes_starvation():
    # With few concurrent comm workers, a high saturation floor raises makespan.
    Lc = 0.01
    base = P.schedule_makespan(0, 0.0, 600, Lc, 32, 0, comm_sat=1)
    floored = P.schedule_makespan(0, 0.0, 600, Lc, 32, 0, comm_sat=128)
    assert floored > base
    # once enough workers are active, the floor is inert
    plenty_base = P.schedule_makespan(0, 0.0, 600, Lc, 304, 0, comm_sat=1)
    plenty_floor = P.schedule_makespan(0, 0.0, 600, Lc, 304, 0, comm_sat=64)
    assert plenty_floor == pytest.approx(plenty_base, rel=0.05)


def test_schedule_atomic_delay_adds_to_makespan():
    m0 = P.schedule_makespan(304, 1.0, 0, 0.0, 304, 304, c_atomic_gemm=0.0)
    m1 = P.schedule_makespan(304, 1.0, 0, 0.0, 304, 304, c_atomic_gemm=0.1)
    assert m1 > m0


# ---------------------------------------------------------------------------
# origami-dependent (GEMM cost + full predictor)
# ---------------------------------------------------------------------------
@requires_origami
def test_gemm_interference_monotonic():
    # Lg rises with the concurrent-comm fraction (lower gemm_wgs => more comm).
    hi = P.gemm_tile_ms_fused(8192, 4608, 8192, 304, 160, comm_intensity=2.0)
    lo = P.gemm_tile_ms_fused(8192, 4608, 8192, 304, 304, comm_intensity=2.0)
    assert hi > lo
    # and with comm intensity
    a = P.gemm_tile_ms_fused(8192, 4608, 8192, 304, 160, comm_intensity=0.0)
    b = P.gemm_tile_ms_fused(8192, 4608, 8192, 304, 160, comm_intensity=2.0)
    assert b > a


@requires_origami
def test_gemm_full_ms_calibrated():
    # standalone GEMM cost is the origami number x the validated clock correction
    assert P.ORIGAMI_GEMM_CAL > 1.0
    ms = P.gemm_full_ms(8192, 4608, 8192)
    assert ms > 0


@requires_origami
@pytest.mark.parametrize("coll", ["all_gather", "all_reduce", "reduce_scatter", "all_to_all", "broadcast"])
def test_predict_split_returns_valid_config(coll):
    shape = dict(gemm_m=8192, gemm_n=4608, gemm_k=8192, comm_m=8192, comm_n=4608, collective=coll, world=8)
    cands = [160, 208, 240, 272]
    r = P.predict_split(shape, candidates=cands)
    assert r["gemm_wgs"] in cands
    assert r["pred_ms"] > 0
    assert r["mt"] == (256, 256, 64)  # default tile when no mt_candidates


@requires_origami
def test_predict_split_tile_margin_gate_prefers_default():
    # with a tiny margin=0 it may switch tiles; with a huge margin it must keep
    # the preferred/default tile.
    shape = dict(gemm_m=8192, gemm_n=4608, gemm_k=8192, comm_m=8192, comm_n=4608, collective="all_gather", world=8)
    mts = [(256, 256, 64), (128, 128, 64), (256, 128, 64)]
    r = P.predict_split(shape, candidates=[208, 240, 272], mt_candidates=mts, mt_prefer=(256, 256, 64), mt_margin=10.0)
    assert r["mt"] == (256, 256, 64)


@requires_origami
def test_predict_split_all_reduce_floor_shifts_toward_comm():
    # The AR comm-saturation floor makes starving comm of CUs costlier, so it can
    # only shift the pick toward MORE comm (<= gemm_wgs), never less. On a
    # comm-heavy AR shape it should strictly reduce the chosen gemm_wgs.
    shape = dict(gemm_m=8192, gemm_n=4608, gemm_k=2048, comm_m=8192, comm_n=4608, collective="all_reduce", world=8)
    cands = [160, 208, 240, 272]
    r_on = P.predict_split(shape, candidates=cands)
    orig = P._comm_sat_channels
    P._comm_sat_channels = lambda *a, **k: 1  # disable the floor
    try:
        r_off = P.predict_split(shape, candidates=cands)
    finally:
        P._comm_sat_channels = orig
    assert r_on["gemm_wgs"] <= r_off["gemm_wgs"]
