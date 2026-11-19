# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only checks that reported improvements cannot silently hide regressions."""

import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "workspace_report", Path(__file__).resolve().parents[2] / "benchmark/ops/summarize_workspace_benchmark.py"
)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def reports():
    base = {
        "schema_version": 1,
        "label": "base",
        "commit": "base-sha",
        "world_size": 2,
        "gpu": "AMD test fixture",
        "torch": "test",
        "hip": "test",
        "triton": "test",
        "tritonblas": "test",
        "dtype": "float16",
        "config": {"block_size_m": 64, "block_size_n": 64, "block_size_k": 32},
        "warmup": 5,
        "iterations": 20,
        "trials": 3,
        "results": [
            {"m": 128, "n": 1024, "k_local": 512, "implementation": variant, "wall_ms": value, "validated": True}
            for variant, value in (("atomic", 2.0), ("one_shot", 2.0), ("torch_mm_rccl", 10.0))
        ],
    }
    head = copy.deepcopy(base)
    head.update(label="head", commit="head-sha")
    return base, head


def test_comparison_retains_slowdowns_and_excludes_control_from_geometric_mean():
    base, head = reports()
    head["results"][0]["wall_ms"] = 1.0  # 2x faster
    head["results"][1]["wall_ms"] = 4.0  # 2x slower
    head["results"][2]["wall_ms"] = 0.1  # control must not inflate the aggregate
    text = reporter.compare(base, head)
    assert "2.000x | -50.0%" in text
    assert "0.500x | +100.0%" in text
    assert "all 2 fused cases: **1.000x**" in text


@pytest.mark.parametrize(
    "key", ["gpu", "world_size", "torch", "hip", "triton", "tritonblas", "dtype", "config", "iterations"]
)
def test_mismatched_conditions_are_rejected(key):
    base, head = reports()
    head[key] = "different"
    with pytest.raises(ValueError, match=f"different {key}"):
        reporter.compare(base, head)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_measurements_are_rejected(value):
    base, head = reports()
    head["results"][0]["wall_ms"] = value
    with pytest.raises(ValueError, match="Invalid or unvalidated"):
        reporter.compare(base, head)


def test_missing_case_is_rejected():
    base, head = reports()
    head["results"].pop()
    with pytest.raises(ValueError, match="same nonempty set"):
        reporter.compare(base, head)


def test_duplicate_case_is_rejected():
    base, head = reports()
    head["results"].append(head["results"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        reporter.compare(base, head)


def test_unvalidated_case_is_rejected():
    base, head = reports()
    head["results"][0]["validated"] = False
    with pytest.raises(ValueError, match="Invalid or unvalidated"):
        reporter.compare(base, head)
