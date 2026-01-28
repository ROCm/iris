#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import iris

import importlib.util
from pathlib import Path

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/02_all_load/all_load_bench.py").resolve()
module_name = "all_load_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        ((1 << 20), (1 << 30)),  # 1 MiB buffer, 1 GiB heap
        ((1 << 22), (1 << 31)),  # 4 MiB buffer, 2 GiB heap
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_all_load_bench(dtype, buffer_size, heap_size, block_size):
    # TODO: Benchmark is not accurate. See: https://github.com/ROCm/iris/issues/119
    pytest.skip("Benchmark is not accurate. See: https://github.com/ROCm/iris/issues/119")
    shmem = None
    try:
        shmem = iris.iris(heap_size)
        num_ranks = shmem.get_num_ranks()

        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        n_elements = buffer_size // element_size_bytes
        buffer = shmem.zeros(n_elements, dtype=dtype)

        # Create arguments similar to what all_load_bench.py expects
        args = {
            "datatype": _torch_dtype_to_str(dtype),
            "block_size": block_size,
            "active_ranks": num_ranks,
            "num_warmup": 4,
            "num_experiments": 8,
            "verbose": False,
            "validate": False,
        }

        shmem.barrier()

        # Run the experiment and measure bandwidth
        bandwidth_gbps = module.run_experiment(shmem, args, buffer)

        shmem.barrier()

        # Verify that we got a reasonable bandwidth measurement
        assert isinstance(bandwidth_gbps, float)
        assert bandwidth_gbps >= 0.0  # Bandwidth should be non-negative
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass  # Ignore errors during cleanup
            # Explicitly delete the shmem instance to trigger cleanup
            del shmem
            # Force garbage collection to ensure IPC handles are cleaned up
            import gc

            gc.collect()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,  # Test with one dtype for validation
    ],
)
def test_all_load_bench_with_validation(dtype):
    """Test all_load_bench with validation enabled to ensure correctness"""
    heap_size = 1 << 30  # 1 GiB heap
    buffer_size = 1 << 20  # 1 MiB buffer
    block_size = 512

    shmem = None
    try:
        shmem = iris.iris(heap_size)
        num_ranks = shmem.get_num_ranks()

        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        n_elements = buffer_size // element_size_bytes
        buffer = shmem.zeros(n_elements, dtype=dtype)

        # Create arguments with validation enabled
        args = {
            "datatype": _torch_dtype_to_str(dtype),
            "block_size": block_size,
            "active_ranks": num_ranks,
            "num_warmup": 1,
            "num_experiments": 1,
            "verbose": False,
            "validate": True,  # Enable validation
        }

        shmem.barrier()

        # Run the experiment and measure bandwidth
        bandwidth_gbps = module.run_experiment(shmem, args, buffer)

        shmem.barrier()

        # Verify that we got a reasonable bandwidth measurement
        assert isinstance(bandwidth_gbps, float)
        assert bandwidth_gbps >= 0.0  # Bandwidth should be non-negative
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass  # Ignore errors during cleanup
            # Explicitly delete the shmem instance to trigger cleanup
            del shmem
            # Force garbage collection to ensure IPC handles are cleaned up
            import gc

            gc.collect()


def _torch_dtype_to_str(dtype):
    """Convert torch dtype to string format expected by all_load_bench.py"""
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
