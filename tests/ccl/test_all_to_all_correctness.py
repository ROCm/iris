# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive correctness tests for AllToAll and AllToAllv.

Tests iris implementations against torch.distributed (RCCL backend) across:
  - Multiple message sizes (1KB through 1GB)
  - All primary dtypes (float16, bfloat16, float32)
  - Variable split sizes (AllToAllv)
  - Rank-specific input patterns for full correctness verification

Run with:
    torchrun --standalone --nproc_per_node=8 tests/run_tests_distributed.py \
        tests/ccl/test_all_to_all_correctness.py -v
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


def _compute_M_N(total_bytes, dtype, world_size):
    """Compute M, N such that M * N * world_size * element_size = total_bytes."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_elements = total_bytes // elem_size
    # Elements per rank
    elements_per_rank = total_elements // world_size
    # Choose M = 1 for small sizes, scale up for large
    if elements_per_rank <= 256:
        M = 1
    elif elements_per_rank <= 4096:
        M = 16
    elif elements_per_rank <= 65536:
        M = 64
    else:
        M = 256
    N = elements_per_rank // M
    if N < 1:
        N = 1
        M = elements_per_rank
    return M, N


# Test sizes: 1KB through 1GB
ALL_SIZES = [
    1024,  # 1KB
    4 * 1024,  # 4KB
    16 * 1024,  # 16KB
    64 * 1024,  # 64KB
    256 * 1024,  # 256KB
    1024 * 1024,  # 1MB
    4 * 1024 * 1024,  # 4MB
    16 * 1024 * 1024,  # 16MB
    64 * 1024 * 1024,  # 64MB
    256 * 1024 * 1024,  # 256MB
    1024 * 1024 * 1024,  # 1GB
]

# Reduced sizes for CI (skip very large ones)
CI_SIZES = [1024, 4096, 16384, 65536, 262144, 1048576]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("total_bytes", CI_SIZES)
def test_all_to_all_correctness_ci(dtype, total_bytes):
    """Test AllToAll correctness against torch.distributed reference for various sizes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = _compute_M_N(total_bytes, dtype, world_size)
    if M * N * world_size * torch.tensor([], dtype=dtype).element_size() < 16:
        pytest.skip("Size too small for this dtype")

    # Create rank-specific input pattern: rank_id * 1000 + element_index % 100
    torch.manual_seed(42 + rank)

    # Iris AllToAll
    iris_input = shmem.zeros((M, N * world_size), dtype=dtype)
    iris_output = shmem.zeros((M, N * world_size), dtype=dtype)

    for t in range(world_size):
        iris_input[:, t * N : (t + 1) * N] = float(rank * 1000 + t)

    # Reference: torch.distributed.all_to_all
    torch_input_list = [iris_input[:, t * N : (t + 1) * N].clone() for t in range(world_size)]
    torch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    shmem.barrier()
    dist.all_to_all(torch_output_list, torch_input_list)
    torch.cuda.synchronize()
    torch_ref = torch.cat(torch_output_list, dim=1)

    # Iris AllToAll
    shmem.barrier()
    config = Config(block_size_m=min(32, M), block_size_n=min(64, N))
    shmem.ccl.all_to_all(iris_output, iris_input, config=config)
    torch.cuda.synchronize()

    # Compare
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-3, 1e-3

    max_diff = (iris_output - torch_ref).abs().max().item()

    try:
        assert torch.allclose(iris_output, torch_ref, atol=atol, rtol=rtol), (
            f"AllToAll mismatch at size={total_bytes}B dtype={dtype}: max_diff={max_diff}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("total_bytes", ALL_SIZES)
def test_all_to_all_correctness(dtype, total_bytes):
    """Test AllToAll correctness for all sizes including 256MB and 1GB.

    Marked @pytest.mark.slow — skipped in CI by default, run with:
        pytest -m slow tests/ccl/test_all_to_all_correctness.py
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**34  # 16GB for large sizes
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = _compute_M_N(total_bytes, dtype, world_size)
    if M * N * world_size * torch.tensor([], dtype=dtype).element_size() < 16:
        pytest.skip("Size too small for this dtype")

    # Create rank-specific input pattern: rank_id * 1000 + element_index % 100
    torch.manual_seed(42 + rank)

    # Iris AllToAll
    iris_input = shmem.zeros((M, N * world_size), dtype=dtype)
    iris_output = shmem.zeros((M, N * world_size), dtype=dtype)

    for t in range(world_size):
        iris_input[:, t * N : (t + 1) * N] = float(rank * 1000 + t)

    # Reference: torch.distributed.all_to_all
    torch_input_list = [iris_input[:, t * N : (t + 1) * N].clone() for t in range(world_size)]
    torch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    shmem.barrier()
    dist.all_to_all(torch_output_list, torch_input_list)
    torch.cuda.synchronize()
    torch_ref = torch.cat(torch_output_list, dim=1)

    # Iris AllToAll
    shmem.barrier()
    config = Config(block_size_m=min(32, M), block_size_n=min(64, N))
    shmem.ccl.all_to_all(iris_output, iris_input, config=config)
    torch.cuda.synchronize()

    # Compare
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:
        atol, rtol = 1e-3, 1e-3

    max_diff = (iris_output - torch_ref).abs().max().item()

    try:
        assert torch.allclose(iris_output, torch_ref, atol=atol, rtol=rtol), (
            f"AllToAll mismatch at size={total_bytes}B dtype={dtype}: max_diff={max_diff}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size
        (256, 128, 32, 16),  # Minimum BLOCK_N=16
        (1024, 256, 32, 64),  # Medium
    ],
)
def test_all_to_all_shapes(dtype, M, N, block_size_m, block_size_n):
    """Test AllToAll with various tensor shapes and block sizes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(heap_size=2**33)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    iris_input = shmem.zeros((M, N * world_size), dtype=dtype)
    iris_output = shmem.zeros((M, N * world_size), dtype=dtype)

    for t in range(world_size):
        iris_input[:, t * N : (t + 1) * N] = float(rank * 1000 + t)

    torch_input_list = [iris_input[:, t * N : (t + 1) * N].clone() for t in range(world_size)]
    torch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    shmem.barrier()
    dist.all_to_all(torch_output_list, torch_input_list)
    torch.cuda.synchronize()
    torch_ref = torch.cat(torch_output_list, dim=1)

    shmem.barrier()
    config = Config(block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.all_to_all(iris_output, iris_input, config=config)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    max_diff = (iris_output - torch_ref).abs().max().item()

    try:
        assert torch.allclose(iris_output, torch_ref, atol=atol), f"Max diff: {max_diff}, expected < {atol}"
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_all_to_all_v_equal_splits(dtype):
    """Test AllToAllv with equal split sizes (should match AllToAll)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(heap_size=2**33)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M, N = 128, 64
    input_split_sizes = [N] * world_size
    output_split_sizes = [N] * world_size

    total_cols = N * world_size
    iris_input = shmem.zeros((M, total_cols), dtype=dtype)
    iris_output = shmem.zeros((M, total_cols), dtype=dtype)

    for t in range(world_size):
        iris_input[:, t * N : (t + 1) * N] = float(rank * 1000 + t)

    # Reference
    torch_input_list = [iris_input[:, t * N : (t + 1) * N].clone() for t in range(world_size)]
    torch_output_list = [torch.zeros(M, N, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]
    shmem.barrier()
    dist.all_to_all(torch_output_list, torch_input_list)
    torch.cuda.synchronize()
    torch_ref = torch.cat(torch_output_list, dim=1)

    # Iris AllToAllv
    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=64)
    shmem.ccl.all_to_all_v(iris_output, iris_input, output_split_sizes, input_split_sizes, config=config)
    torch.cuda.synchronize()

    max_diff = (iris_output - torch_ref).abs().max().item()

    try:
        assert torch.allclose(iris_output, torch_ref, atol=1e-5), f"AllToAllv with equal splits: max_diff={max_diff}"
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_to_all_v_variable_splits(dtype):
    """Test AllToAllv with variable split sizes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = iris.iris(heap_size=2**33)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    M = 64
    # Each rank sends the same size to all ranks: (rank+1)*16 cols per dest
    input_split_sizes = [(rank + 1) * 16 for _ in range(world_size)]
    # From rank s, we receive (s+1)*16 cols
    output_split_sizes = [(s + 1) * 16 for s in range(world_size)]

    total_input_cols = sum(input_split_sizes)
    total_output_cols = sum(output_split_sizes)

    # Allocate with max size to maintain heap symmetry
    max_input_cols = (world_size) * 16 * world_size
    iris_input_buf = shmem.zeros((M, max_input_cols), dtype=dtype)
    iris_output = shmem.zeros((M, total_output_cols), dtype=dtype)

    iris_input = iris_input_buf[:, :total_input_cols]
    offset = 0
    for t in range(world_size):
        size = input_split_sizes[t]
        iris_input[:, offset : offset + size] = float(rank * 100 + t)
        offset += size

    # Reference
    torch_input_list = []
    offset = 0
    for t in range(world_size):
        size = input_split_sizes[t]
        torch_input_list.append(iris_input[:, offset : offset + size].clone())
        offset += size

    torch_output_list = []
    for t in range(world_size):
        size = output_split_sizes[t]
        torch_output_list.append(torch.zeros(M, size, dtype=dtype, device=f"cuda:{rank}"))

    shmem.barrier()
    dist.all_to_all(torch_output_list, torch_input_list)
    torch.cuda.synchronize()
    torch_ref = torch.cat(torch_output_list, dim=1)

    # Iris AllToAllv
    shmem.barrier()
    config = Config(block_size_m=32, block_size_n=16)
    shmem.ccl.all_to_all_v(
        iris_output,
        iris_input_buf[:, :total_input_cols],
        output_split_sizes,
        input_split_sizes,
        config=config,
    )
    torch.cuda.synchronize()

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    max_diff = (iris_output - torch_ref).abs().max().item()

    try:
        assert torch.allclose(iris_output, torch_ref, atol=atol), (
            f"AllToAllv variable splits: max_diff={max_diff} dtype={dtype}"
        )
    finally:
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
