# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for ring-based all-gather collective operation.
Tests the RCCL-ported ring AllGather algorithm.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


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
        (128, 128, 32, 32),  # BLOCK_N < N/world_size
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_gather_ring(dtype, M, N, block_size_m, block_size_n):
    """Test ring all-gather by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # PyTorch reference: arange(size) * rank_id + rank_id
    # Use float32 intermediate to avoid float16 overflow (max 65504) for large tensors
    arange_vals = torch.arange(M * N, dtype=torch.float32, device=f"cuda:{rank}").reshape(M, N)
    pytorch_input_tensor = ((arange_vals % 1000) * rank + rank).to(dtype)
    pytorch_output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor)
    torch.cuda.synchronize()

    # Iris ring all-gather
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)
    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    config = Config(
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        all_gather_variant="ring",
    )
    workspace = shmem.ccl.all_gather_preamble(iris_output_tensor, iris_input_tensor, config=config)
    shmem.barrier()
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Bit-exact comparison: AllGather is pure data movement with zero arithmetic,
    # so any non-zero difference indicates a correctness bug.
    match = torch.equal(iris_output_tensor, pytorch_output_tensor)

    try:
        if not match:
            max_diff = torch.abs(iris_output_tensor.float() - pytorch_output_tensor.float()).max().item()
            mismatches = (iris_output_tensor != pytorch_output_tensor).sum().item()
            total = iris_output_tensor.numel()
            pytest.fail(
                f"Rank {rank}: {mismatches}/{total} mismatches, max_diff={max_diff} for dtype={dtype}, M={M}, N={N}"
            )
        assert match
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "size_bytes",
    [
        1024,  # 1 KB
        4096,  # 4 KB
        16384,  # 16 KB
        65536,  # 64 KB
        262144,  # 256 KB
        1048576,  # 1 MB
        4194304,  # 4 MB
        16777216,  # 16 MB
        67108864,  # 64 MB
    ],
)
def test_all_gather_ring_sizes(dtype, size_bytes):
    """Test ring all-gather across message sizes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = size_bytes // elem_size
    # Make 2D: M rows, N columns (choose N=128 for good vectorization)
    N = min(128, num_elements)
    M = num_elements // N
    if M == 0:
        M = 1
        N = num_elements

    # Each element is unique: arange(size) * rank_id + rank_id
    # Use float32 intermediate with modular arithmetic to avoid float16 overflow (max 65504)
    arange_vals = torch.arange(M * N, dtype=torch.float32, device=f"cuda:{rank}").reshape(M, N)
    pytorch_input = ((arange_vals % 1000) * rank + rank).to(dtype)
    pytorch_output = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output, pytorch_input)
    torch.cuda.synchronize()

    # Iris
    iris_input = shmem.zeros((M, N), dtype=dtype)
    iris_input.copy_(pytorch_input)
    iris_output = shmem.zeros((world_size * M, N), dtype=dtype)

    config = Config(
        block_size_m=32,
        block_size_n=min(64, N),
        all_gather_variant="ring",
    )
    workspace = shmem.ccl.all_gather_preamble(iris_output, iris_input, config=config)
    shmem.barrier()
    shmem.ccl.all_gather(iris_output, iris_input, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Bit-exact comparison (AllGather is pure data movement)
    match = torch.equal(iris_output, pytorch_output)

    try:
        if not match:
            max_diff = torch.abs(iris_output.float() - pytorch_output.float()).max().item()
            mismatches = (iris_output != pytorch_output).sum().item()
            total = iris_output.numel()
            pytest.fail(
                f"Rank {rank}: {mismatches}/{total} mismatches, max_diff={max_diff} "
                f"for size={size_bytes}B dtype={dtype}"
            )
        assert match
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
