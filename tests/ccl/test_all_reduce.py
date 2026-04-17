# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-reduce collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        # "ring",
        "two_shot",
        "one_shot",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size (partial-width, multi-block per rank)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_reduce(variant, dtype, M, N, block_size_m, block_size_n):
    """Test all-reduce functionality by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    # PyTorch's all_reduce format: each rank has M x N data
    # All ranks compute the sum of all tensors
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_reduce to get reference output
    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Now set up Iris all_reduce format
    # Iris format: same as PyTorch - input and output are both (M, N)
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Run Iris all_reduce with specified variant
    shmem.barrier()
    config = Config(all_reduce_variant=variant, block_size_m=block_size_m, block_size_n=block_size_n)
    if variant == "two_shot":
        # Test both distribution modes for two_shot
        config.all_reduce_distribution = 0  # striding
    if variant == "ring":
        config.all_reduce_num_rings = min(2, config.comm_sms)

    # Explicitly call preamble to ensure proper initialization and synchronization
    # This helps with test isolation when tests run sequentially
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris output doesn't match PyTorch's all_reduce (variant={variant})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_all_reduce_two_shot_distribution(distribution, dtype=torch.float32, M=1024, N=256):
    """Test two-shot all-reduce with different distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(all_reduce_variant="two_shot", all_reduce_distribution=distribution)

    # Explicitly call preamble to ensure proper initialization and synchronization
    workspace = shmem.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    shmem.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris two-shot output doesn't match PyTorch (distribution={distribution})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


def test_all_reduce_spinlock_lock_too_small():
    """Test that ValueError is raised when the spinlock lock array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized lock array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)

    M, N = 512, 512

    iris_input = shmem.zeros((M, N), dtype=torch.float32)
    iris_output = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller lock array
    config_large = Config(all_reduce_variant="spinlock", block_size_m=128, block_size_n=128)
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized lock array is detected.
    config_small = Config(all_reduce_variant="spinlock", block_size_m=64, block_size_n=64)
    with pytest.raises(ValueError, match="Lock array too small"):
        shmem.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()


def test_all_reduce_ring_flags_too_small():
    """Test that ValueError is raised when the ring flags array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized flags array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    world_size = shmem.get_num_ranks()

    M, N = 512, 512

    # Choose block_size_n values divisible by world_size for both configs
    # Use 128 and 64 which are divisible by typical world sizes (1, 2, 4, 8)
    block_size_n_large = (128 // world_size) * world_size
    block_size_n_small = (64 // world_size) * world_size
    if block_size_n_large == 0 or block_size_n_small == 0 or block_size_n_large == block_size_n_small:
        del shmem
        pytest.skip(f"Cannot create two distinct block sizes divisible by world_size={world_size}")

    iris_input = shmem.zeros((M, N), dtype=torch.float32)
    iris_output = shmem.zeros((M, N), dtype=torch.float32)

    shmem.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller flags array
    config_large = Config(
        all_reduce_variant="ring",
        block_size_m=128,
        block_size_n=block_size_n_large,
    )
    workspace = shmem.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized flags array is detected.
    config_small = Config(
        all_reduce_variant="ring",
        block_size_m=64,
        block_size_n=block_size_n_small,
    )
    with pytest.raises(ValueError, match="Flags array too small"):
        shmem.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    shmem.barrier()
    del shmem
    import gc

    gc.collect()
