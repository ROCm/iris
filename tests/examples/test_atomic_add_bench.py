#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import iris

import importlib.util
from pathlib import Path

current_dir = Path(__file__).parent
file_path = (current_dir / "../../examples/04_atomic_add/atomic_add_bench.py").resolve()
module_name = "atomic_add_bench"
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
        (20480, (1 << 33)),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_atomic_add_bench(dtype, buffer_size, heap_size, block_size):
    """Test that atomic_add benchmark runs and validation passes."""
    shmem = iris.iris(heap_size)
    num_ranks = shmem.get_num_ranks()

    bandwidth_matrix = np.zeros((num_ranks, num_ranks), dtype=np.float32)
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = buffer_size // element_size_bytes
    source_buffer = shmem.arange(n_elements, dtype=dtype)
    result_buffer = shmem.zeros_like(source_buffer)

    shmem.barrier()

    # Test with validation enabled and return_result flag
    args = {
        "datatype": "fp32" if dtype == torch.float32 else "fp16",
        "block_size": block_size,
        "verbose": True,
        "validate": True,
        "num_experiments": 1,
        "num_warmup": 1,
        "return_result": True,
    }

    for source_rank in range(num_ranks):
        for destination_rank in range(num_ranks):
            result = module.run_experiment(shmem, args, source_rank, destination_rank, source_buffer, result_buffer)

            # Unpack result based on return format
            if isinstance(result, tuple):
                bandwidth_gbps, buffer_result = result
            else:
                bandwidth_gbps = result
                buffer_result = None

            # Bandwidth should be positive
            assert bandwidth_gbps > 0, f"Bandwidth should be positive, got {bandwidth_gbps}"
            bandwidth_matrix[source_rank, destination_rank] = bandwidth_gbps

            # Test expected values when we have buffer result
            if buffer_result is not None and shmem.get_rank() == destination_rank:
                # After all atomic_add operations, each element should be num_ranks
                expected = torch.ones(n_elements, dtype=dtype, device="cuda") * num_ranks
                torch.testing.assert_close(buffer_result, expected, rtol=0, atol=1)

            shmem.barrier()

    # All bandwidth measurements should be positive
    assert np.all(bandwidth_matrix > 0), "All bandwidth measurements should be positive"


def test_atomic_add_kernel_stores_result():
    """Test that atomic_add_kernel includes the store operation."""
    import inspect

    # Get the source code of the atomic_add_kernel
    source = inspect.getsource(module.atomic_add_kernel)

    # Check that it includes tl.store operation
    assert "tl.store" in source, "atomic_add_kernel should store the result in result_buffer"
    assert "result_buffer + offsets" in source, "atomic_add_kernel should store to result_buffer"
    assert "result" in source, "atomic_add_kernel should store the atomic_add result"


def test_validation_logic():
    """Test that validation logic properly resets buffers and validates expected values."""
    import inspect

    # Get the source code of the run_experiment function
    source = inspect.getsource(module.run_experiment)

    # Check that validation includes buffer reset
    assert "source_buffer.copy_" in source, "Validation should reset source_buffer"
    assert "result_buffer.zero_" in source, "Validation should reset result_buffer"
    assert "torch.arange(n_elements" in source, "Validation should check against arange pattern"
