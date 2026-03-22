# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation using Gluon.
"""

import pytest
import torch
import torch.distributed as dist

# Try to import Gluon, skip tests if not available
try:
    import iris.experimental.iris_gluon as iris_gluon
    from iris.ccl import Config
    from iris.ccl.all_gather import all_gather

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
    "M, N, block_size_m, block_size_n",
    [
        # Flat-2D kernel: BLOCK_SIZE_M * BLOCK_SIZE_N must be a multiple of
        # (threads_per_warp * num_warps) = 256. Optimal: 2048-4096 total elems.
        (8192, 8192, 8, 256),   # Optimal flat-2D tile (2048 elems, 8/thread)
        (8192, 8192, 4, 512),   # Alternative optimal (2048 elems, 8/thread)
        (8192, 8192, 8, 512),   # Larger tile (4096 elems, 16/thread)
        (256, 256, 8, 256),     # Small tensor with optimal tile
        (1024, 512, 4, 256),    # Medium tensor
        (8192, 8192, 32, 1024), # Legacy-sized tile (32768 elems)
    ],
)
def test_all_gather_gluon(dtype, M, N, block_size_m, block_size_n):
    """Test all-gather functionality using Gluon by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris_gluon.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank has an M x N input tensor
    # Output is (world_size * M, N) - concatenated along dimension 0
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

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
    config = Config(use_gluon=True, block_size_m=block_size_m, block_size_n=block_size_n)
    all_gather(iris_output_tensor, iris_input_tensor, shmem, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris Gluon output doesn't match PyTorch's all_gather_into_tensor"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        shmem.barrier()
        del shmem
        import gc
        gc.collect()
