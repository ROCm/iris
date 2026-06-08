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


def load_example_module(relative_path: str, module_name: str):
    file_path = (current_dir / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import message passing example modules
load_store_module = load_example_module(
    "../../examples/06_message_passing/message_passing_load_store.py", "message_passing_load_store"
)
put_module = load_example_module("../../examples/06_message_passing/message_passing_put.py", "message_passing_put")
host_initiated_module = load_example_module(
    "../../examples/06_message_passing/message_passing_host_initiated.py", "message_passing_host_initiated"
)


def create_test_args(dtype_str, buffer_size, heap_size, block_size):
    """Create args dict that matches what parse_args() returns."""
    return {"datatype": dtype_str, "buffer_size": buffer_size, "heap_size": heap_size, "block_size": block_size}


def run_message_passing_kernels(module, args, *, use_copy_engine: bool = False):
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

        copy_engine_ctx = shmem.get_copy_engine_ctx()

        producer_fn = getattr(module.producer_kernel, "fn", None)
        producer_params = (
            producer_fn.__code__.co_varnames if producer_fn and hasattr(producer_fn, "__code__") else tuple()
        )
        needs_copy_engine_arg = any(param in producer_params for param in ("copy_engine_ctx"))
        has_use_copy_engine = "USE_COPY_ENGINE" in producer_params

        if cur_rank == producer_rank:
            # Run producer kernel
            kernel_args = [
                source_buffer,
                destination_buffer,
                flags,
                n_elements,
                producer_rank,
                consumer_rank,
                args["block_size"],
                shmem.get_heap_bases(),
            ]
            if needs_copy_engine_arg:
                kernel_args.append(copy_engine_ctx)

            launch_kwargs = {"USE_COPY_ENGINE": use_copy_engine} if has_use_copy_engine else {}
            module.producer_kernel[grid](*kernel_args, **launch_kwargs)
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


@pytest.mark.parametrize("dtype_str", ["fp16", "fp32"])
@pytest.mark.parametrize("buffer_size, heap_size", [(4096, 1 << 20)])
@pytest.mark.parametrize("block_size", [512])
def test_message_passing_copy_engine(dtype_str, buffer_size, heap_size, block_size):
    """Test message passing with device-initiated copy engine."""
    args = create_test_args(dtype_str, buffer_size, heap_size, block_size)
    success = run_message_passing_kernels(put_module, args, use_copy_engine=True)
    assert success, "Message passing copy-engine validation failed"


def run_host_initiated_copy_engine(module, args):
    """Execute the host-initiated message passing example logic."""
    shmem = None
    try:
        shmem = iris.iris(args["heap_size"])
        dtype = module.torch_dtype_from_str(args["datatype"])
        cur_rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()

        if world_size != args.get("num_ranks", 2):
            pytest.skip("Host-initiated message passing example requires two ranks.")

        # Allocate buffers
        destination_buffer = shmem.zeros(args["buffer_size"], device="cuda", dtype=dtype)
        if dtype.is_floating_point:
            source_buffer = shmem.randn(args["buffer_size"], device="cuda", dtype=dtype)
        else:
            ii = torch.iinfo(dtype)
            source_buffer = shmem.randint(ii.min, ii.max, (args["buffer_size"],), device="cuda", dtype=dtype)

        producer_rank = 0
        consumer_rank = 1

        n_elements = source_buffer.numel()
        block_size = args["block_size"]
        num_blocks = triton.cdiv(n_elements, block_size)
        grid = (num_blocks,)

        flags = shmem.zeros((num_blocks,), device="cuda", dtype=torch.int32)

        # Use the example's producer function for the producer rank
        if cur_rank == producer_rank:
            module.host_initiated_producer(
                shmem, source_buffer, destination_buffer, flags, consumer_rank, block_size, verbose=False
            )
        else:
            # Consumer uses the kernel (same as other tests)
            module.consumer_kernel[grid](
                destination_buffer, flags, n_elements, consumer_rank, block_size, shmem.get_heap_bases()
            )

        shmem.barrier()

        # Validation
        success = True
        if cur_rank == consumer_rank:
            expected = source_buffer * 2
            if not torch.allclose(destination_buffer, expected, atol=1):
                success = False

        shmem.barrier()
        return success
    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            import gc

            del shmem
            gc.collect()


@pytest.mark.parametrize("dtype_str", ["fp16", "fp32"])
@pytest.mark.parametrize("buffer_size, heap_size", [(4096, 1 << 20)])
@pytest.mark.parametrize("block_size", [512])
def test_message_passing_host_initiated(dtype_str, buffer_size, heap_size, block_size):
    """Test host-initiated copy engine example."""
    args = {
        "datatype": dtype_str,
        "buffer_size": buffer_size,
        "heap_size": heap_size,
        "block_size": block_size,
        "num_ranks": 2,
    }
    success = run_host_initiated_copy_engine(host_initiated_module, args)
    assert success, "Host-initiated message passing validation failed"
