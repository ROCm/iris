# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
``iris.concurrent.autotune`` -- top-k autotuner for the concurrent GEMM+collective.

The concurrent kernels expose a large config space (the GEMM/comm CU split
``gemm_wgs`` and the GEMM tile ``gemm_block`` being the dominant knobs). Sweeping
it exhaustively per shape is expensive, so this module uses the analytical
:mod:`iris.concurrent.predictor` (origami GEMM + comm cost models) as a *ranker*
to shortlist the top-``k`` predicted configs, then benchmarks only those on the
device and keeps the fastest.

The winner is persisted to a JSON tuning database so subsequent launches of the
same shape are a dict lookup with no benchmarking. Layout mirrors Triton's own
cache convention:

* ``$IRIS_CONCURRENT_CACHE``            -- explicit file path (highest priority)
* ``$XDG_CACHE_HOME/iris/concurrent/autotune.json``
* ``~/.cache/iris/concurrent/autotune.json``   (default)
* ``$TMPDIR/iris_concurrent_autotune.json``    (fallback if HOME unwritable)

Usage is via ``tune=True`` on any :mod:`iris.concurrent.gemm` op; this module is
the machinery behind that flag and is not usually called directly.

Environment overrides (all optional):

* ``IRIS_CONCURRENT_TUNE_TOPK``   -- number of predicted configs to benchmark (default 8)
* ``IRIS_CONCURRENT_TUNE_FORCE``  -- ``1`` to re-tune even on a cache hit
* ``IRIS_CONCURRENT_TUNE_NREP``   -- timing repeats per candidate (default 20)
* ``IRIS_CONCURRENT_TUNE_NWARMUP``-- warmup iters per candidate (default 5)
* ``IRIS_CONCURRENT_CACHE``       -- tuning-db file path
"""

import fcntl
import json
import math
import os
import tempfile
import threading

__all__ = ["tune_config", "topk_configs", "cache_path", "load_db", "clear"]

# Default GEMM tile candidates co-explored with the split. The op's own default
# ``gemm_block`` is always added on top of these at tune time.
_DEFAULT_MT_CANDIDATES = [
    (256, 256, 64),
    (128, 256, 64),
    (256, 128, 64),
    (128, 128, 64),
    (256, 64, 64),
]

_DEFAULT_TOPK = 8

_db_lock = threading.Lock()
_db_cache = None  # in-process copy of the on-disk db


# ---------------------------------------------------------------------------
# tuning database (json on disk)
# ---------------------------------------------------------------------------
def cache_path():
    """Resolve the tuning-database file path (see module docstring for order)."""
    explicit = os.environ.get("IRIS_CONCURRENT_CACHE")
    if explicit:
        return explicit
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "iris", "concurrent", "autotune.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    except OSError:
        return os.path.join(tempfile.gettempdir(), "iris_concurrent_autotune.json")


def load_db():
    """Load (and memoize) the tuning db as a dict."""
    global _db_cache
    if _db_cache is not None:
        return _db_cache
    path = cache_path()
    try:
        with open(path, "r") as f:
            _db_cache = json.load(f)
    except (OSError, ValueError):
        _db_cache = {}
    return _db_cache


def _save_db(db):
    path = cache_path()
    tmp = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(f"{path}.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(path, "r") as f:
                    merged = json.load(f)
            except (OSError, ValueError):
                merged = {}
            merged.update(db)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
            tmp = None
            db.clear()
            db.update(merged)
    except OSError:
        pass
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass


def clear():
    """Drop the in-memory db and delete the on-disk file."""
    global _db_cache
    _db_cache = None
    try:
        os.remove(cache_path())
    except OSError:
        pass


def _make_key(shape, mode, num_wgs, arch, extra):
    """Stable string key for a (shape, mode, hardware) tuning point."""
    arch = str(arch).split(":", 1)[0] or "unknown"
    parts = [
        f"arch={arch}",
        f"world={shape['world']}",
        f"nwg={num_wgs}",
        f"mode={mode}",
        f"coll={shape['collective']}",
        f"g={shape['gemm_m']}x{shape['gemm_n']}x{shape['gemm_k']}",
        f"c={shape['comm_m']}x{shape['comm_n']}",
    ]
    if extra:
        parts.append(str(extra))
    return "|".join(parts)


# ---------------------------------------------------------------------------
# candidate generation (the "heuristic" that seeds top-k)
# ---------------------------------------------------------------------------
def _heuristic_grid(num_wgs, default_gemm_block):
    """Origami-free fallback: a coarse split grid at the op's default tile."""
    fracs = (0.5, 0.625, 0.75, 0.875, 1.0)
    seen = []
    for f in fracs:
        gw = max(1, min(num_wgs, int(round(num_wgs * f))))
        if gw not in seen:
            seen.append(gw)
    return [{"gemm_wgs": gw, "gemm_block": tuple(default_gemm_block)} for gw in seen]


def topk_configs(
    shape, num_wgs, k=_DEFAULT_TOPK, candidates=None, mt_candidates=None, default_gemm_block=(256, 256, 64)
):
    """Return the top-``k`` predicted ``{gemm_wgs, gemm_block}`` configs.

    Ranks the full ``(gemm_wgs x gemm_block)`` grid with the analytical predictor
    (:func:`iris.concurrent.predictor.predict_split`) and returns the ``k``
    lowest predicted-makespan configs. Falls back to a coarse heuristic split
    grid if origami / the comm model is unavailable.
    """
    mts = list(mt_candidates) if mt_candidates is not None else list(_DEFAULT_MT_CANDIDATES)
    dgb = tuple(default_gemm_block)
    if dgb not in mts:
        mts.insert(0, dgb)

    ranked = []
    try:
        from . import predictor

        for mt in mts:
            r = predictor.predict_split(shape, num_wgs=num_wgs, candidates=candidates, mt=mt)
            for gw, _t_gemm, _comm_full, mk in r["curve"]:
                mk = float(mk)
                # Drop non-finite predictions: origami's makespan degenerates for
                # tiny GEMMs (comm_intensity -> inf), which would otherwise poison
                # the ranking. A degenerate shape simply falls through to the
                # heuristic grid (+ the always-included default in tune_config).
                if math.isfinite(mk):
                    ranked.append((mk, int(gw), tuple(mt)))
    except Exception:
        # origami extension / comm model unavailable -> heuristic seed
        return _heuristic_grid(num_wgs, dgb)

    if not ranked:
        return _heuristic_grid(num_wgs, dgb)

    ranked.sort(key=lambda x: x[0])
    out, seen = [], set()
    for mk, gw, mt in ranked:
        key = (gw, mt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"gemm_wgs": gw, "gemm_block": mt, "pred_ms": mk})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# device benchmarking + persistence
# ---------------------------------------------------------------------------
def _int_env(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def tune_config(
    *,
    run,
    shape,
    mode,
    num_wgs,
    barrier_fn=lambda: None,
    default_gemm_block=(256, 256, 64),
    default_gemm_wgs=None,
    arch="unknown",
    k=None,
    candidates=None,
    mt_candidates=None,
    extra_key=None,
    force=None,
    cache=True,
    n_warmup=None,
    n_repeat=None,
):
    """Resolve the best ``{gemm_wgs, gemm_block}`` for one shape, benchmarking
    only the top-``k`` predicted candidates and caching the winner.

    Args:
        run: ``callable(cfg_dict)`` that launches the op with ``cfg["gemm_wgs"]``
            and ``cfg["gemm_block"]`` applied (and ``tune=False``). Used both to
            benchmark candidates and left to the caller for the final launch.
        shape: dict with ``gemm_m/n/k``, ``comm_m/n``, ``collective``, ``world``.
        mode: ``"fused"`` or ``"concurrent"`` (part of the cache key).
        num_wgs: total persistent workgroups (the split's denominator).
        barrier_fn: cross-rank sync used by the benchmark (``shmem.barrier``).
        arch: device arch string, folded into the cache key.

    Returns:
        The chosen config dict ``{gemm_wgs, gemm_block, pred_ms?, measured_ms?}``.
    """
    k = k if k is not None else _int_env("IRIS_CONCURRENT_TUNE_TOPK", _DEFAULT_TOPK)
    force = force if force is not None else bool(_int_env("IRIS_CONCURRENT_TUNE_FORCE", 0))
    n_warmup = n_warmup if n_warmup is not None else _int_env("IRIS_CONCURRENT_TUNE_NWARMUP", 5)
    n_repeat = n_repeat if n_repeat is not None else _int_env("IRIS_CONCURRENT_TUNE_NREP", 20)

    key = _make_key(shape, mode, num_wgs, arch, extra_key)

    with _db_lock:
        db = load_db()
        if cache and not force and key in db:
            hit = db[key]["config"]
            return {
                "gemm_wgs": int(hit["gemm_wgs"]),
                "gemm_block": tuple(hit["gemm_block"]),
                "measured_ms": db[key].get("measured_ms"),
                "cached": True,
            }

    cfgs = topk_configs(
        shape, num_wgs, k=k, candidates=candidates, mt_candidates=mt_candidates, default_gemm_block=default_gemm_block
    )

    # Always benchmark the op's STATIC default too, so the tuner can never pick a
    # config slower than the untuned baseline (monotonicity guarantee). This is
    # what rescues degenerate shapes -- e.g. tiny decode GEMMs where origami's
    # makespan model produces non-finite / unreliable rankings.
    dwgs = default_gemm_wgs if default_gemm_wgs is not None else (num_wgs * 3) // 4
    default_cfg = {"gemm_wgs": int(dwgs), "gemm_block": tuple(default_gemm_block), "pred_ms": None}
    if not any(
        c["gemm_wgs"] == default_cfg["gemm_wgs"] and tuple(c["gemm_block"]) == default_cfg["gemm_block"] for c in cfgs
    ):
        cfgs = list(cfgs) + [default_cfg]

    from ..host.platform.utils import do_bench

    results = []
    for cfg in cfgs:
        launch = {"gemm_wgs": cfg["gemm_wgs"], "gemm_block": tuple(cfg["gemm_block"])}
        try:
            ms = do_bench(
                lambda c=launch: run(c),
                barrier_fn=barrier_fn,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                return_mode="median",
            )
        except Exception:
            continue
        results.append((float(ms), launch, cfg.get("pred_ms")))

    if not results:
        # everything failed to launch -> fall back to the op's static default.
        return {"gemm_wgs": None, "gemm_block": tuple(default_gemm_block), "cached": False}

    results.sort(key=lambda x: x[0])
    best_ms, best_cfg, best_pred = results[0]

    if cache:
        with _db_lock:
            db = load_db()
            db[key] = {
                "config": {"gemm_wgs": best_cfg["gemm_wgs"], "gemm_block": list(best_cfg["gemm_block"])},
                "measured_ms": best_ms,
                "pred_ms": best_pred,
                "n_candidates": len(results),
                "candidates_ms": [
                    {"gemm_wgs": c["gemm_wgs"], "gemm_block": list(c["gemm_block"]), "ms": ms} for ms, c, _ in results
                ],
            }
            _save_db(db)

    return {
        "gemm_wgs": best_cfg["gemm_wgs"],
        "gemm_block": tuple(best_cfg["gemm_block"]),
        "measured_ms": best_ms,
        "pred_ms": best_pred,
        "cached": False,
    }
