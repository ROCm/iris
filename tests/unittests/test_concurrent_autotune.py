# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for iris.concurrent.autotune (top-k tuner + tuning db).

Covers the pure-Python machinery that does not need a GPU: cache-path
resolution, the origami-free heuristic candidate grid, the JSON tuning-db
round-trip, cache-key stability, and the cache-hit fast path of ``tune_config``
(which must not benchmark). The device-benchmarking path needs 2 GPUs and is
exercised by the example tests.
"""

import importlib.util
import json
import os

import pytest

# Load the module directly by path so the pure-Python logic runs in a minimal
# CPU/CI environment (the package import pulls in torch/triton). Relative
# imports inside the module are only hit on the GPU paths we don't test here.
_ap = os.path.join(os.path.dirname(__file__), "..", "..", "iris", "concurrent", "autotune.py")
_spec = importlib.util.spec_from_file_location("iris_concurrent_autotune", os.path.abspath(_ap))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


SHAPE = dict(gemm_m=8192, gemm_n=4608, gemm_k=8192, comm_m=8192, comm_n=4608, collective="all_gather", world=8)


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "autotune.json")
    monkeypatch.setenv("IRIS_CONCURRENT_CACHE", path)
    A._db_cache = None
    yield path
    A.clear()


def test_cache_path_env_override(tmp_db):
    assert A.cache_path() == tmp_db


def test_cache_path_default_under_home(monkeypatch):
    monkeypatch.delenv("IRIS_CONCURRENT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    p = A.cache_path()
    assert p.endswith(os.path.join("iris", "concurrent", "autotune.json"))


def test_topk_heuristic_fallback():
    # origami/comm-model unavailable in CI -> coarse split grid at default tile.
    cfgs = A.topk_configs(SHAPE, num_wgs=304, k=6, default_gemm_block=(256, 64, 64))
    assert 1 <= len(cfgs) <= 6
    assert all(c["gemm_block"] == (256, 64, 64) for c in cfgs)
    gws = [c["gemm_wgs"] for c in cfgs]
    assert gws == sorted(set(gws))  # unique + ascending fractions
    assert all(1 <= g <= 304 for g in gws)
    assert 304 in gws  # full-GEMM split is always a candidate


def test_db_roundtrip(tmp_db):
    A.clear()
    db = A.load_db()
    assert db == {}
    db["mykey"] = {"config": {"gemm_wgs": 240, "gemm_block": [256, 64, 64]}, "measured_ms": 1.5}
    A._save_db(db)
    A._db_cache = None  # force a fresh read from disk
    assert A.load_db()["mykey"]["config"]["gemm_wgs"] == 240


def test_db_save_merges_entries_added_by_another_process(tmp_db):
    db = A.load_db()
    db["local"] = {"config": {"gemm_wgs": 240}}
    with open(tmp_db, "w") as f:
        json.dump({"remote": {"config": {"gemm_wgs": 160}}}, f)
    A._save_db(db)
    A._db_cache = None
    assert set(A.load_db()) == {"local", "remote"}


def test_clear_removes_file(tmp_db):
    A._save_db({"x": 1})
    assert os.path.exists(tmp_db)
    A.clear()
    assert not os.path.exists(tmp_db)


def test_make_key_stable_and_distinct():
    k1 = A._make_key(SHAPE, "fused", 304, "gfx942", None)
    k2 = A._make_key(SHAPE, "fused", 304, "gfx942", None)
    assert k1 == k2
    k3 = A._make_key(SHAPE, "concurrent", 304, "gfx942", None)
    assert k1 != k3  # mode is part of the key
    k4 = A._make_key({**SHAPE, "collective": "all_reduce"}, "fused", 304, "gfx942", None)
    assert k1 != k4  # collective is part of the key
    assert A._make_key(SHAPE, "fused", 304, "gfx942:sramecc+:xnack-", None) == k1


def test_tune_config_cache_hit_skips_benchmark(tmp_db):
    key = A._make_key(SHAPE, "fused", 304, "gfx942", None)
    A._save_db({key: {"config": {"gemm_wgs": 216, "gemm_block": [128, 128, 64]}, "measured_ms": 0.9}})
    A._db_cache = None

    def run(cfg):
        raise AssertionError("cache hit must not benchmark")

    out = A.tune_config(run=run, shape=SHAPE, mode="fused", num_wgs=304, arch="gfx942")
    assert out["cached"] is True
    assert out["gemm_wgs"] == 216
    assert out["gemm_block"] == (128, 128, 64)
