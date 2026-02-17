#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Basic tests for iris.bench module that don't require GPU or iris runtime.
Tests the new decorator-only approach.
"""

import json
import sys
from pathlib import Path

# Import bench module directly without going through iris.__init__
bench_path = Path(__file__).parent.parent.parent / "iris" / "bench.py"
import importlib.util

spec = importlib.util.spec_from_file_location("bench", bench_path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

import torch


def test_benchmark_result_creation():
    """Test creating a BenchmarkResult object."""
    result = bench.BenchmarkResult(
        name="test_benchmark",
        mean_ms=10.5,
        median_ms=10.2,
        p50_ms=10.2,
        p99_ms=15.3,
        min_ms=8.1,
        max_ms=16.2,
        n_warmup=5,
        n_repeat=50,
        params={"size": 1024},
        metadata={"gpu": "MI300X"},
        raw_times=[10.1, 10.2, 10.3],
    )

    assert result.name == "test_benchmark"
    assert result.mean_ms == 10.5
    assert result.median_ms == 10.2
    assert result.p50_ms == 10.2
    assert result.p99_ms == 15.3
    assert result.min_ms == 8.1
    assert result.max_ms == 16.2
    assert result.n_warmup == 5
    assert result.n_repeat == 50
    assert result.params == {"size": 1024}
    assert result.metadata == {"gpu": "MI300X"}
    assert result.raw_times == [10.1, 10.2, 10.3]
    print("✓ test_benchmark_result_creation passed")


def test_benchmark_result_to_dict():
    """Test converting BenchmarkResult to dictionary."""
    result = bench.BenchmarkResult(
        name="test",
        mean_ms=10.0,
        median_ms=10.0,
        p50_ms=10.0,
        p99_ms=12.0,
        min_ms=9.0,
        max_ms=13.0,
        n_warmup=5,
        n_repeat=10,
        raw_times=[9.0, 10.0, 11.0, 12.0, 13.0],
    )

    # Without raw times
    d = result.to_dict(include_raw_times=False)
    assert "raw_times" not in d
    assert d["name"] == "test"
    assert d["mean_ms"] == 10.0

    # With raw times
    d = result.to_dict(include_raw_times=True)
    assert "raw_times" in d
    assert d["raw_times"] == [9.0, 10.0, 11.0, 12.0, 13.0]
    print("✓ test_benchmark_result_to_dict passed")


def test_benchmark_result_to_json():
    """Test converting BenchmarkResult to JSON."""
    result = bench.BenchmarkResult(
        name="test",
        mean_ms=10.0,
        median_ms=10.0,
        p50_ms=10.0,
        p99_ms=12.0,
        min_ms=9.0,
        max_ms=13.0,
        n_warmup=5,
        n_repeat=10,
    )

    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["name"] == "test"
    assert parsed["mean_ms"] == 10.0
    print("✓ test_benchmark_result_to_json passed")


def test_compute_percentile():
    """Test percentile computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    p50 = bench._compute_percentile(values, 50)
    assert 5.0 <= p50 <= 6.0

    p99 = bench._compute_percentile(values, 99)
    assert p99 > 9.0

    # Edge cases
    assert bench._compute_percentile([], 50) == 0.0
    assert bench._compute_percentile([5.0], 50) == 5.0
    print("✓ test_compute_percentile passed")


def test_torch_dtype_from_str():
    """Test torch_dtype_from_str utility."""
    assert bench.torch_dtype_from_str("int8") == torch.int8
    assert bench.torch_dtype_from_str("fp16") == torch.float16
    assert bench.torch_dtype_from_str("bf16") == torch.bfloat16
    assert bench.torch_dtype_from_str("fp32") == torch.float32

    try:
        bench.torch_dtype_from_str("invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown datatype" in str(e)
    print("✓ test_torch_dtype_from_str passed")


def test_compute_bandwidth_gbps():
    """Test bandwidth computation."""
    # 1 GiB in 1 second = 1 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(2**30, 1000)
    assert abs(bandwidth - 1.0) < 0.001

    # 2 GiB in 0.5 seconds = 4 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(2 * 2**30, 500)
    assert abs(bandwidth - 4.0) < 0.001

    # 512 MiB in 100ms = 5 GiB/s
    bandwidth = bench.compute_bandwidth_gbps(512 * 2**20, 100)
    assert abs(bandwidth - 5.0) < 0.01
    print("✓ test_compute_bandwidth_gbps passed")


if __name__ == "__main__":
    test_benchmark_result_creation()
    test_benchmark_result_to_dict()
    test_benchmark_result_to_json()
    test_compute_percentile()
    test_torch_dtype_from_str()
    test_compute_bandwidth_gbps()
    print("\n✅ All tests passed!")
