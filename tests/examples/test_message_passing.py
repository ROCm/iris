#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import iris

import importlib.util
from pathlib import Path

current_dir = Path(__file__).parent

# Import message_passing_load_store module
load_store_file_path = (current_dir / "../../examples/06_message_passing/message_passing_load_store.py").resolve()
load_store_module_name = "message_passing_load_store"
load_store_spec = importlib.util.spec_from_file_location(load_store_module_name, load_store_file_path)
load_store_module = importlib.util.module_from_spec(load_store_spec)
load_store_spec.loader.exec_module(load_store_module)

# Import message_passing_put module
put_file_path = (current_dir / "../../examples/06_message_passing/message_passing_put.py").resolve()
put_module_name = "message_passing_put"
put_spec = importlib.util.spec_from_file_location(put_module_name, put_file_path)
put_module = importlib.util.module_from_spec(put_spec)
put_spec.loader.exec_module(put_module)


def create_test_args(dtype_str, buffer_size, heap_size, block_size):
    """Create args dict that matches what parse_args() returns."""
    return {"datatype": dtype_str, "buffer_size": buffer_size, "heap_size": heap_size, "block_size": block_size}


def run_message_passing_kernels(module, args):
    """Run the core message passing logic without command line argument parsing."""
    shmem = None
    try:
        shmem = iris.iris(args["heap_size"])
        dtype = module.torch_dtype_from_str(args["datatype"])
        cur_rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()

        # Check that we have exactly 2 ranks as required by message passing examples
        if world_size != 2:
            pytest.skip("Message passing examples require exactly two processes.")

        # Allocate source and destination buffers on the symmetric heap - match original examples
        source_buffer = shmem.zeros(args["buffer_size"], device="cuda", dtype=dtype)
        if dtype.is_floating_point:
            destination_buffer = shmem.randn(args["buffer_size"], device="cuda", dtype=dtype)
        else:
            ii = torch.iinfo(dtype)
            destination_buffer = shmem.randint(ii.min, ii.max, (args["buffer_size"],), device="cuda", dtype=dtype)

        producer_rank = 0
        consumer_rank = 1

        n_elements = source_buffer.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        num_blocks = triton.cdiv(n_elements, args["block_size"])

        # Allocate flags on the symmetric heap
        flags = shmem.zeros((num_blocks,), device="cuda", dtype=torch.int32)

        if cur_rank == producer_rank:
            # Run producer kernel
            module.producer_kernel[grid](
                source_buffer,
                destination_buffer,
                flags,
                n_elements,
                producer_rank,
                consumer_rank,
                args["block_size"],
                shmem.get_heap_bases(),
            )
        else:
            # Run consumer kernel
            module.consumer_kernel[grid](
                destination_buffer, flags, n_elements, consumer_rank, args["block_size"], shmem.get_heap_bases()
            )

        shmem.barrier()

        # Validation - only consumer rank validates (matches original examples)
        success = True
        if cur_rank == consumer_rank:
            expected = source_buffer * 2
            if not torch.allclose(destination_buffer, expected, atol=1):
                success = False

        shmem.barrier()
        return success
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
    "dtype_str",
    [
        "int8",
        "fp16",
        "bf16",
        "fp32",
    ],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        (4096, 1 << 20),  # Smaller sizes for testing
        (8192, 1 << 21),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_message_passing_load_store(dtype_str, buffer_size, heap_size, block_size):
    """Test message passing with load/store operations."""
    args = create_test_args(dtype_str, buffer_size, heap_size, block_size)
    success = run_message_passing_kernels(load_store_module, args)
    assert success, "Message passing load/store validation failed"


@pytest.mark.parametrize(
    "dtype_str",
    [
        "int8",
        "fp16",
        "bf16",
        "fp32",
    ],
)
@pytest.mark.parametrize(
    "buffer_size, heap_size",
    [
        (4096, 1 << 20),  # Smaller sizes for testing
        (8192, 1 << 21),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        512,
        1024,
    ],
)
def test_message_passing_put(dtype_str, buffer_size, heap_size, block_size):
    """Test message passing with put operations."""
    args = create_test_args(dtype_str, buffer_size, heap_size, block_size)
    success = run_message_passing_kernels(put_module, args)
    assert success, "Message passing put validation failed"
