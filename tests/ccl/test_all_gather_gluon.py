# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation using Gluon.
"""

import os

import pytest
import torch
import torch.distributed as dist

# Try to import Gluon, skip tests if not available
try:
    import iris
    from iris.ccl import Config
    from triton.experimental import gluon  # noqa: F401

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n, comm_sms",
    [
        # block_size_n must be a multiple of (threads_per_warp * num_warps).
        # With num_warps=4 the minimum is 256 on a 64-wide wavefront, 128 on a 32-wide one.
        # elems_per_thread = block_size_n / (threads_per_warp * num_warps): higher = wider
        # vector loads. The per-case counts below are for a 64-wide wavefront; double them at 32.
        (256, 256, 32, 256, 64),  # Small: elems_per_thread=1 (scalar loads)
        (1024, 512, 32, 512, 64),  # Medium: elems_per_thread=2 (dword loads)
        (8192, 8192, 32, 1024, 64),  # Large: elems_per_thread=4 (dwordx4, optimal)
        (13, 300, 8, 256, 1),  # Reuse peer addresses across tiles with partial rows and columns.
        (7, 64, 8, 256, 1),
    ],
)
def test_all_gather_gluon(dtype, M, N, block_size_m, block_size_n, comm_sms):
    """Test all-gather functionality using Gluon by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    # Size heap to fit input (M*N) + output (max_ranks*M*N) with headroom
    max_ranks = int(os.environ.get("WORLD_SIZE", 8))
    elem_size = torch.tensor([], dtype=dtype).element_size()
    needed = (1 + max_ranks) * M * N * elem_size
    heap_size = max(2**30, int(needed * 2))  # 2x headroom, minimum 1GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank has an M x N input tensor
    # Output is (world_size * M, N) - concatenated along dimension 0
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")

    # Create output tensor for PyTorch: (world_size * M, N)
    pytorch_output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    # Run PyTorch's all_gather_into_tensor to get reference output
    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor)
    torch.cuda.synchronize()

    # Now set up Iris Gluon all_gather
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    # Run Iris Gluon all_gather
    shmem.barrier()
    config = Config(use_gluon=True, block_size_m=block_size_m, block_size_n=block_size_n, comm_sms=comm_sms)
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Compare results
    try:
        torch.testing.assert_close(iris_output_tensor, pytorch_output_tensor, rtol=0, atol=0)
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
