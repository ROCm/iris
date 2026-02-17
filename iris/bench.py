# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unified benchmarking harness for Iris.

This module provides a decorator-based infrastructure for benchmarking operations:
- Automatic iris instance creation and management
- Warmup and iteration handling
- Timing and synchronization
- Statistics computation (mean, p50, p99)
- Structured result output (JSON or dict)

The harness automatically constructs the iris instance and passes it to your
benchmark function, allowing you to annotate different parts of your code:
- @setup: Runs once before any timing (e.g., tensor allocation)
- @preamble: Runs before each iteration (e.g., resetting flags)
- @measure: The code to actually benchmark (e.g., kernel launch)

Example usage:

    from iris.bench import benchmark

    @benchmark(name="gemm_kernel", warmup=5, iters=50, heap_size=1<<33)
    def run_benchmark(shmem, m=8192, n=4608, k=36864):
        # shmem is automatically created by the decorator

        @setup
        def allocate_tensors():
            # Runs once before timing starts
            A = shmem.randn(m, k, dtype=torch.float16)
            B = shmem.randn(k, n, dtype=torch.float16)
            C = shmem.zeros(m, n, dtype=torch.float16)
            return A, B, C

        @preamble
        def reset_output(C):
            # Runs before each timed iteration
            C.zero_()

        @measure
        def run_kernel(A, B, C):
            # This is what gets timed
            gemm_kernel[grid](A, B, C, m, n, k)

    result = run_benchmark(m=8192, n=4608, k=36864)
    result.print_summary()
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List
import functools
import torch


def _compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


@dataclass
class BenchmarkResult:
    """
    Stores results from a benchmark run.

    Attributes:
        name: Name of the benchmark
        mean_ms: Mean time in milliseconds
        median_ms: Median time in milliseconds
        p50_ms: 50th percentile time in milliseconds
        p99_ms: 99th percentile time in milliseconds
        min_ms: Minimum time in milliseconds
        max_ms: Maximum time in milliseconds
        n_warmup: Number of warmup iterations
        n_repeat: Number of timing iterations
        params: Additional parameters passed to the benchmark
        metadata: Additional metadata
        raw_times: Raw timing measurements in milliseconds
    """

    name: str
    mean_ms: float
    median_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    n_warmup: int
    n_repeat: int
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_times: List[float] = field(default_factory=list)

    def to_dict(self, include_raw_times: bool = False) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Args:
            include_raw_times: Whether to include raw timing measurements

        Returns:
            Dictionary representation of the result
        """
        result = asdict(self)
        if not include_raw_times:
            result.pop("raw_times", None)
        return result

    def to_json(self, include_raw_times: bool = False, indent: int = 2) -> str:
        """
        Convert result to JSON string.

        Args:
            include_raw_times: Whether to include raw timing measurements
            indent: JSON indentation level

        Returns:
            JSON string representation of the result
        """
        return json.dumps(self.to_dict(include_raw_times=include_raw_times), indent=indent)

    def print_summary(self):
        """Print a human-readable summary of the benchmark result."""
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {self.name}")
        if self.params:
            print(f"Parameters: {self.params}")
        print(f"{'-' * 60}")
        print(f"Mean:   {self.mean_ms:10.4f} ms")
        print(f"Median: {self.median_ms:10.4f} ms")
        print(f"P50:    {self.p50_ms:10.4f} ms")
        print(f"P99:    {self.p99_ms:10.4f} ms")
        print(f"Min:    {self.min_ms:10.4f} ms")
        print(f"Max:    {self.max_ms:10.4f} ms")
        print(f"{'-' * 60}")
        print(f"Warmup iterations: {self.n_warmup}")
        print(f"Timing iterations: {self.n_repeat}")
        if self.metadata:
            print(f"Metadata: {self.metadata}")
        print(f"{'=' * 60}\n")


class _BenchmarkContext:
    """Internal context for collecting setup, preamble, and measure functions."""

    def __init__(self):
        self.setup_fn = None
        self.preamble_fn = None
        self.measure_fn = None

    def setup(self, fn):
        """Mark a function as setup code (runs once before timing)."""
        self.setup_fn = fn
        return fn

    def preamble(self, fn):
        """Mark a function as preamble code (runs before each timed iteration)."""
        self.preamble_fn = fn
        return fn

    def measure(self, fn):
        """Mark a function as the code to measure (gets timed)."""
        self.measure_fn = fn
        return fn


def benchmark(
    name: str,
    warmup: int = 25,
    iters: int = 100,
    heap_size: int = 1 << 33,
    auto_print: bool = False,
):
    """
    Decorator for benchmarking functions with automatic iris instance management.

    The decorator creates an iris instance and passes it to your benchmark function.
    Within your function, use @setup, @preamble, and @measure decorators to annotate
    different parts of your benchmark code.

    Args:
        name: Name of the benchmark
        warmup: Number of warmup iterations
        iters: Number of timing iterations
        heap_size: Size of iris symmetric heap
        auto_print: Whether to automatically print results

    Returns:
        Decorated function that returns BenchmarkResult

    Example:
        @benchmark(name="my_kernel", warmup=5, iters=50)
        def run(shmem, size=1024):
            @setup
            def allocate():
                buffer = shmem.zeros(size, size)
                return buffer

            @measure
            def kernel_launch(buffer):
                my_kernel[grid](buffer)

        result = run(size=2048)
        result.print_summary()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Import iris here to avoid circular dependencies
            from . import iris as iris_module

            # Create iris instance
            shmem = iris_module.iris(heap_size)

            # Create benchmark context for collecting annotated functions
            ctx = _BenchmarkContext()

            # Make decorators available in the function scope
            import builtins

            original_setup = getattr(builtins, "setup", None)
            original_preamble = getattr(builtins, "preamble", None)
            original_measure = getattr(builtins, "measure", None)

            try:
                # Inject decorators into builtins temporarily
                builtins.setup = ctx.setup
                builtins.preamble = ctx.preamble
                builtins.measure = ctx.measure

                # Call user function to collect setup/preamble/measure functions
                func(shmem, *args, **kwargs)

            finally:
                # Restore original builtins
                if original_setup is not None:
                    builtins.setup = original_setup
                elif hasattr(builtins, "setup"):
                    delattr(builtins, "setup")

                if original_preamble is not None:
                    builtins.preamble = original_preamble
                elif hasattr(builtins, "preamble"):
                    delattr(builtins, "preamble")

                if original_measure is not None:
                    builtins.measure = original_measure
                elif hasattr(builtins, "measure"):
                    delattr(builtins, "measure")

            # Validate that measure function was provided
            if ctx.measure_fn is None:
                raise ValueError(f"Benchmark '{name}' must have a @measure decorated function")

            # Run setup once if provided
            setup_results = ()
            if ctx.setup_fn is not None:
                result = ctx.setup_fn()
                # Convert to tuple for consistent handling
                if result is None:
                    setup_results = ()
                elif isinstance(result, tuple):
                    setup_results = result
                else:
                    setup_results = (result,)

            # Define preamble_fn for do_bench
            def preamble_fn():
                if ctx.preamble_fn is not None:
                    ctx.preamble_fn(*setup_results)

            # Define measure_fn for do_bench
            def measure_fn():
                ctx.measure_fn(*setup_results)

            # Import do_bench at runtime
            from .util import do_bench

            # Run benchmark with automatic barrier
            raw_times = do_bench(
                measure_fn,
                barrier_fn=shmem.barrier,
                preamble_fn=preamble_fn,
                n_warmup=warmup,
                n_repeat=iters,
                return_mode="all",
            )

            # Compute statistics
            mean_ms = sum(raw_times) / len(raw_times) if raw_times else 0.0
            median_ms = _compute_percentile(raw_times, 50)
            p50_ms = median_ms  # P50 is the same as median
            p99_ms = _compute_percentile(raw_times, 99)
            min_ms = min(raw_times) if raw_times else 0.0
            max_ms = max(raw_times) if raw_times else 0.0

            # Extract function parameters for metadata
            func_params = {**kwargs}
            for i, arg in enumerate(args):
                if i < len(func.__code__.co_varnames) - 1:  # -1 to skip 'shmem'
                    param_name = func.__code__.co_varnames[i + 1]  # +1 to skip 'shmem'
                    func_params[param_name] = arg

            result = BenchmarkResult(
                name=name,
                mean_ms=mean_ms,
                median_ms=median_ms,
                p50_ms=p50_ms,
                p99_ms=p99_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                n_warmup=warmup,
                n_repeat=iters,
                params=func_params,
                raw_times=raw_times,
            )

            if auto_print:
                result.print_summary()

            return result

        return wrapper

    return decorator


# Utility functions for common patterns


def torch_dtype_from_str(datatype: str) -> torch.dtype:
    """
    Convert string datatype to torch.dtype.

    Args:
        datatype: String representation of datatype

    Returns:
        torch.dtype object

    Raises:
        ValueError: If datatype is not recognized
    """
    dtype_map = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unknown datatype: {datatype}. Expected one of {list(dtype_map.keys())}")
    return dtype_map[datatype]


def compute_bandwidth_gbps(
    total_bytes: int,
    time_ms: float,
) -> float:
    """
    Compute bandwidth in GiB/s.

    Args:
        total_bytes: Total number of bytes transferred
        time_ms: Time in milliseconds

    Returns:
        Bandwidth in GiB/s
    """
    time_sec = time_ms * 1e-3
    return total_bytes / time_sec / (2**30)


__all__ = [
    "BenchmarkResult",
    "benchmark",
    "torch_dtype_from_str",
    "compute_bandwidth_gbps",
]
