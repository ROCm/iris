#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example demonstrating the unified benchmarking harness (iris.bench).

This example shows how to use the @benchmark decorator with @setup, @preamble,
and @measure annotations. The decorator automatically creates an iris instance
and passes it to your function.

Note: setup, preamble, and measure are injected by the @benchmark decorator
at runtime and are not imported. This is intentional.
"""

# ruff: noqa: F821

import torch
from iris.bench import benchmark, torch_dtype_from_str, compute_bandwidth_gbps


# Example 1: Simple benchmark with setup and measure
@benchmark(name="simple_operation", warmup=2, iters=5, auto_print=True)
def benchmark_simple(shmem, size=1024):
    """Simple benchmark using decorator with setup and measure."""

    @setup
    def allocate_tensors():
        # Runs once before timing starts
        tensor = shmem.zeros(size, size, dtype=torch.float32)
        return tensor

    @measure
    def run_operation(tensor):
        # This is what gets timed
        result = tensor + 1.0


# Example 2: Benchmark with preamble for resetting state
@benchmark(name="with_preamble", warmup=2, iters=5)
def benchmark_with_preamble(shmem, size=2048):
    """Benchmark demonstrating preamble usage."""

    @setup
    def allocate():
        tensor = shmem.ones(size, size, dtype=torch.float16)
        output = shmem.zeros(size, size, dtype=torch.float16)
        return tensor, output

    @preamble
    def reset_output(tensor, output):
        # Runs before each timed iteration
        output.zero_()

    @measure
    def compute(tensor, output):
        # This gets timed
        output.copy_(tensor * 2.0)


# Example 3: Bandwidth calculation
@benchmark(name="bandwidth_test", warmup=2, iters=5)
def benchmark_bandwidth(shmem, size=1024 * 1024 * 256, dtype_str="fp16"):
    """Benchmark with bandwidth calculation."""
    dtype = torch_dtype_from_str(dtype_str)
    element_size = torch.tensor([], dtype=dtype).element_size()

    @setup
    def allocate():
        tensor = shmem.zeros(size, dtype=dtype)
        result = shmem.zeros(size, dtype=dtype)
        return tensor, result

    @measure
    def copy_data(tensor, result):
        result.copy_(tensor)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-enabled GPU.")
        exit(1)

    print("=" * 70)
    print("Iris Benchmarking Harness Examples (Decorator-Only)")
    print("=" * 70)

    print("\n### Example 1: Simple operation ###")
    result1 = benchmark_simple(size=1024)
    # Note: auto_print=True so summary is printed automatically

    print("\n### Example 2: With preamble ###")
    result2 = benchmark_with_preamble(size=2048)
    result2.print_summary()

    print("\n### Example 3: Bandwidth test ###")
    result3 = benchmark_bandwidth(size=1024 * 1024 * 256, dtype_str="fp16")

    # Compute bandwidth
    dtype = torch_dtype_from_str("fp16")
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 1024 * 1024 * 256 * element_size
    bandwidth = compute_bandwidth_gbps(total_bytes, result3.mean_ms)

    print(f"\nBandwidth: {bandwidth:.2f} GiB/s")
    print(f"Size: {total_bytes / 2**30:.2f} GiB")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
