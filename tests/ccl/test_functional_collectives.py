# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for iris torch.compile functional collectives.

Tests cover:
1. Eager mode correctness vs torch.distributed RCCL baseline
2. torch.compile integration (tracing, compilation, execution)
3. Fake tensor mode compatibility
4. Multiple dtypes and message sizes
5. Compiled vs eager consistency
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import Config
from iris.compile import functional_collectives as fc


# ============================================================================
# Helper utilities
# ============================================================================

def _skip_if_not_distributed():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _get_rccl_allreduce_reference(input_tensor, rank):
    """Run RCCL all-reduce to get reference output."""
    ref = input_tensor.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return ref


# ============================================================================
# Eager mode correctness tests
# ============================================================================

@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 64),      # Small (~32KB for fp32)
        (256, 128),     # Medium (~128KB)
        (1024, 256),    # Large (~1MB)
        (4096, 1024),   # XLarge (~16MB)
    ],
)
def test_all_reduce_eager_correctness(dtype, M, N):
    """Test iris functional all-reduce in eager mode against RCCL baseline."""
    _skip_if_not_distributed()

    heap_size = 2 ** 33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    try:
        # Initialize compile context
        config = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
        )
        compile_ctx = fc.setup(shmem, config=config)

        # Create input tensor with deterministic values
        input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
        input_tensor.fill_(float(rank + 1))

        # Get RCCL reference
        rccl_output = _get_rccl_allreduce_reference(input_tensor, rank)

        # Run iris functional all-reduce
        shmem.barrier()
        iris_output = fc.all_reduce(input_tensor)
        torch.cuda.synchronize()

        # Compare results
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-5
        else:
            rtol, atol = 1e-3, 1e-3

        max_diff = torch.abs(iris_output - rccl_output).max().item()
        assert torch.allclose(iris_output, rccl_output, rtol=rtol, atol=atol), (
            f"Max difference: {max_diff}\n"
            f"Rank {rank}: iris functional all-reduce doesn't match RCCL "
            f"(dtype={dtype}, shape=({M},{N}))"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


# ============================================================================
# torch.compile integration tests
# ============================================================================

def test_compile_tracing():
    """Test that torch.compile can trace a model with iris collectives."""
    _skip_if_not_distributed()

    heap_size = 2 ** 33
    shmem = iris.iris(heap_size)

    try:
        config = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
        )
        fc.setup(shmem, config=config)

        # Define a simple model with iris collective
        def model_fn(x):
            return torch.ops.iris.all_reduce(x)

        # Compile with inductor backend
        compiled_fn = torch.compile(model_fn, backend="eager")

        # Run the compiled function
        input_tensor = torch.randn(128, 64, dtype=torch.float32, device=f"cuda:{shmem.get_rank()}")
        shmem.barrier()
        output = compiled_fn(input_tensor)
        torch.cuda.synchronize()

        assert output.shape == input_tensor.shape
        assert output.dtype == input_tensor.dtype
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_compiled_matches_eager(dtype):
    """Test that compiled iris all-reduce matches eager mode output."""
    _skip_if_not_distributed()

    heap_size = 2 ** 33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    try:
        config = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
        )
        fc.setup(shmem, config=config)

        M, N = 256, 128

        # Create identical inputs
        input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
        input_tensor.fill_(float(rank + 1))

        # Eager mode
        shmem.barrier()
        eager_output = fc.all_reduce(input_tensor)
        torch.cuda.synchronize()

        # Compiled mode
        def compiled_allreduce(x):
            return torch.ops.iris.all_reduce(x)

        compiled_fn = torch.compile(compiled_allreduce, backend="eager")
        shmem.barrier()
        compiled_output = compiled_fn(input_tensor)
        torch.cuda.synchronize()

        # Both should match
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-5
        else:
            rtol, atol = 1e-3, 1e-3

        assert torch.allclose(eager_output, compiled_output, rtol=rtol, atol=atol), (
            f"Compiled output doesn't match eager output "
            f"(dtype={dtype}, max_diff={torch.abs(eager_output - compiled_output).max().item()})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


def test_compile_mlp_model():
    """Test torch.compile with a simple MLP model containing iris all-reduce."""
    _skip_if_not_distributed()

    heap_size = 2 ** 33
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()

    try:
        config = Config(
            block_size_m=32,
            block_size_n=64,
            all_reduce_variant="two_shot",
            all_reduce_distribution=1,
        )
        fc.setup(shmem, config=config)

        # Simple MLP with all-reduce between layers
        class MLPWithAllReduce(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim, bias=False)
                self.fc2 = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.ops.iris.all_reduce(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        model = MLPWithAllReduce(64).to(f"cuda:{rank}")
        compiled_model = torch.compile(model, backend="eager")

        input_tensor = torch.randn(32, 64, dtype=torch.float32, device=f"cuda:{rank}")

        shmem.barrier()
        # Eager forward
        eager_out = model(input_tensor)
        torch.cuda.synchronize()

        shmem.barrier()
        # Compiled forward
        compiled_out = compiled_model(input_tensor)
        torch.cuda.synchronize()

        assert eager_out.shape == compiled_out.shape
        assert torch.allclose(eager_out, compiled_out, rtol=1e-5, atol=1e-5), (
            f"MLP compiled output doesn't match eager "
            f"(max_diff={torch.abs(eager_out - compiled_out).max().item()})"
        )
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


# ============================================================================
# Fake tensor mode tests
# ============================================================================

def test_fake_tensor_mode_all_reduce():
    """Test that all-reduce works in fake tensor mode (for tracing)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        input_tensor = torch.randn(128, 64, device="meta")
        output = torch.ops.iris.all_reduce(input_tensor)
        assert output.shape == input_tensor.shape
        assert output.dtype == input_tensor.dtype


def test_fake_tensor_mode_all_gather():
    """Test that all-gather works in fake tensor mode."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        input_tensor = torch.randn(128, 64, device="meta")
        output = torch.ops.iris.all_gather(input_tensor, 8)
        assert output.shape == (128 * 8, 64)
        assert output.dtype == input_tensor.dtype


def test_fake_tensor_mode_reduce_scatter():
    """Test that reduce-scatter works in fake tensor mode."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        input_tensor = torch.randn(128, 64, device="meta")
        output = torch.ops.iris.reduce_scatter(input_tensor, 8)
        assert output.shape == (128 // 8, 64)
        assert output.dtype == input_tensor.dtype


# ============================================================================
# Graph capture test (verifies ops in compiled graph)
# ============================================================================

def test_graph_contains_iris_ops():
    """Verify that the compiled graph contains iris collective ops."""

    def model_fn(x):
        y = torch.ops.iris.all_reduce(x)
        return y + 1.0

    # Use make_fx with symbolic tracing mode to use fake implementations
    from torch.fx.experimental.proxy_tensor import make_fx

    # Create a concrete input
    input_tensor = torch.randn(64, 32)

    # make_fx with tracing_mode="fake" uses the registered fake kernels
    graph_module = make_fx(model_fn, tracing_mode="fake")(input_tensor)

    # Check that iris.all_reduce appears in the graph
    found_iris_op = False
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and "all_reduce" in str(node.target):
            found_iris_op = True
            break

    assert found_iris_op, (
        f"iris.all_reduce not found in compiled graph. "
        f"Graph nodes: {[str(n) for n in graph_module.graph.nodes]}"
    )
