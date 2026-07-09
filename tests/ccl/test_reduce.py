# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for reduce collective operation.
Validates against torch.distributed.reduce (RCCL backend) as golden reference.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "variant",
    [
        "one_shot",
        "two_shot",
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
        (128, 128, 32, 32),  # Multi-block per rank
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_reduce(variant, dtype, M, N, block_size_m, block_size_n):
    """Test reduce functionality by comparing against PyTorch's implementation."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    root = 0

    # Create deterministic input
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's reduce to get reference output
    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_output_tensor, dst=root, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Now set up Iris reduce
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    # Run Iris reduce
    shmem.barrier()
    config = Config(reduce_variant=variant, block_size_m=block_size_m, block_size_n=block_size_n)

    workspace = shmem.ccl.reduce_preamble(iris_output_tensor, iris_input_tensor, root=root, config=config)
    shmem.barrier()

    shmem.ccl.reduce(iris_output_tensor, iris_input_tensor, root=root, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Only root rank should have the correct result
    if rank == root:
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank} (root): Iris output doesn't match PyTorch's reduce (variant={variant})"
            )
        finally:
            shmem.barrier()
            del shmem
            import gc
            gc.collect()
    else:
        # Non-root ranks: no check on output (RCCL semantics: undefined)
        shmem.barrier()
        del shmem
        import gc
        gc.collect()


@pytest.mark.parametrize("root", [0, 1, 3, 7])
def test_reduce_different_roots(root, dtype=torch.float32, M=1024, N=256):
    """Test reduce with different root ranks."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    if root >= world_size:
        shmem.barrier()
        del shmem
        import gc
        gc.collect()
        pytest.skip(f"root={root} >= world_size={world_size}")

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_output_tensor, dst=root, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(reduce_variant="two_shot")

    workspace = shmem.ccl.reduce_preamble(iris_output_tensor, iris_input_tensor, root=root, config=config)
    shmem.barrier()

    shmem.ccl.reduce(iris_output_tensor, iris_input_tensor, root=root, config=config, workspace=workspace)
    torch.cuda.synchronize()

    if rank == root:
        atol = 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank} (root={root}): Iris output doesn't match PyTorch's reduce"
            )
        finally:
            shmem.barrier()
            del shmem
            import gc
            gc.collect()
    else:
        shmem.barrier()
        del shmem
        import gc
        gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_reduce_two_shot_distribution(distribution, dtype=torch.float32, M=1024, N=256):
    """Test two-shot reduce with different distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    root = 0

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_output_tensor = pytorch_input_tensor.clone()
    shmem.barrier()
    dist.reduce(pytorch_output_tensor, dst=root, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((M, N), dtype=dtype)

    shmem.barrier()
    config = Config(reduce_variant="two_shot", all_reduce_distribution=distribution)

    workspace = shmem.ccl.reduce_preamble(iris_output_tensor, iris_input_tensor, root=root, config=config)
    shmem.barrier()

    shmem.ccl.reduce(iris_output_tensor, iris_input_tensor, root=root, config=config, workspace=workspace)
    torch.cuda.synchronize()

    if rank == root:
        atol = 1e-5
        max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

        try:
            assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
                f"Max difference: {max_diff}, expected < {atol}\n"
                f"Rank {rank}: Iris two-shot reduce doesn't match PyTorch (distribution={distribution})"
            )
        finally:
            shmem.barrier()
            del shmem
            import gc
            gc.collect()
    else:
        shmem.barrier()
        del shmem
        import gc
        gc.collect()
