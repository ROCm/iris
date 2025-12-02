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
import sys
from pathlib import Path

current_dir = Path(__file__).parent

# Add examples directory to sys.path so that example files can import from examples.common
# Note: Examples use "from examples.common.utils import ..." which requires examples/ in sys.path
examples_dir = (current_dir / "../..").resolve()
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

# Load utils module from file path (not package import)
# Note: We use path-based imports instead of "from examples.common.utils import ..."
# because examples/ is not included in the installed package. This allows tests to
# work with both editable install (pip install -e .) and regular install (pip install git+...).
utils_path = (current_dir / "../../examples/common/utils.py").resolve()
utils_spec = importlib.util.spec_from_file_location("utils", utils_path)
utils_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_module)
torch_dtype_to_str = utils_module.torch_dtype_to_str

# Load benchmark module
file_path = (current_dir / "../../examples/04_atomic_add/atomic_add_bench.py").resolve()
module_name = "atomic_add_bench"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "dtype",
    [
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
def test_atomic_bandwidth(dtype, buffer_size, heap_size, block_size):
    """Test that atomic_add benchmark runs and produces positive bandwidth."""
    shmem = None
    try:
        shmem = iris.iris(heap_size)
        num_ranks = shmem.get_num_ranks()

        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        n_elements = buffer_size // element_size_bytes
        source_buffer = shmem.arange(n_elements, dtype=dtype)

        shmem.barrier()

        args = {
            "datatype": torch_dtype_to_str(dtype),
            "block_size": block_size,
            "verbose": False,
            "validate": False,
            "num_experiments": 10,
            "num_warmup": 5,
        }

        source_rank = 0
        destination_rank = 1 if num_ranks > 1 else 0

        bandwidth_gbps, _ = module.run_experiment(shmem, args, source_rank, destination_rank, source_buffer)

        assert bandwidth_gbps > 0, f"Bandwidth should be positive, got {bandwidth_gbps}"

        shmem.barrier()
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
def test_atomic_correctness(dtype, buffer_size, heap_size, block_size):
    """Test that atomic_add benchmark runs and produces positive bandwidth."""
    shmem = None
    try:
        shmem = iris.iris(heap_size)
        num_ranks = shmem.get_num_ranks()

        element_size_bytes = torch.tensor([], dtype=dtype).element_size()
        n_elements = buffer_size // element_size_bytes
        source_buffer = shmem.arange(n_elements, dtype=dtype)

        shmem.barrier()

        args = {
            "datatype": torch_dtype_to_str(dtype),
            "block_size": block_size,
            "verbose": False,
            "validate": False,
            "num_experiments": 1,
            "num_warmup": 0,
        }

        source_rank = 0
        destination_rank = 1 if num_ranks > 1 else 0

        _, result_buffer = module.run_experiment(shmem, args, source_rank, destination_rank, source_buffer)

        if shmem.get_rank() == destination_rank:
            expected = torch.ones(n_elements, dtype=dtype, device="cuda")

            assert torch.allclose(result_buffer, expected), "Result buffer should be equal to expected"

        shmem.barrier()
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
