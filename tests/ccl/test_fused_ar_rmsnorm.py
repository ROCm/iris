# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for fused AllReduce + Residual Add + RMSNorm collective operation.

Validates correctness against a PyTorch reference implementation that does
the three operations separately: dist.all_reduce → residual += reduced → RMSNorm.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm in float32 for numerical stability."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return (x_normed * weight.to(torch.float32)).to(x.dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tokens, hidden",
    [
        (1, 64),  # Single token, small hidden (decode-like)
        (1, 128),  # Single token, medium hidden
        (4, 256),  # Small batch
        (32, 512),  # Medium batch
        (128, 1024),  # Larger batch
        (256, 4096),  # Typical LLM hidden size (LLaMA 7B)
        (512, 5120),  # LLaMA 13B hidden size
        (64, 8192),  # LLaMA 65B / 70B hidden size
    ],
)
def test_fused_ar_rmsnorm(dtype, tokens, hidden):
    """Test fused AR+RMSNorm against separate PyTorch all_reduce + RMSNorm."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    eps = 1e-6

    # Create deterministic inputs per rank
    # Partial is different per rank (each rank has its own GEMM shard)
    torch.manual_seed(42 + rank)
    partial_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")

    # Residual and weight are replicated across ranks (tensor parallelism invariant)
    torch.manual_seed(42)
    residual_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")
    weight_data = torch.randn(hidden, dtype=dtype, device=f"cuda:{rank}")

    # === PyTorch reference ===
    ref_partial = partial_data.clone()
    ref_residual = residual_data.clone()

    # Step 1: AllReduce partial
    dist.all_reduce(ref_partial, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Step 2: Residual add
    ref_residual = ref_residual + ref_partial

    # Step 3: RMSNorm
    ref_norm_out = _rmsnorm_reference(ref_residual, weight_data, eps)

    # === Iris fused implementation ===
    iris_partial = ctx.zeros((tokens, hidden), dtype=dtype)
    iris_partial.copy_(partial_data)

    iris_residual = ctx.zeros((tokens, hidden), dtype=dtype)
    iris_residual.copy_(residual_data)

    ctx.barrier()

    config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=1,
    )
    iris_norm_out = ctx.ccl.all_reduce_rmsnorm(iris_partial, iris_residual, weight_data, eps=eps, config=config)
    torch.cuda.synchronize()

    # === Compare ===
    # Tolerances for norm output: fused kernel accumulates in fp32 so should be close
    if dtype == torch.float16:
        atol = 1e-2
        rtol = 1e-2
    elif dtype == torch.bfloat16:
        atol = 2e-2
        rtol = 2e-2
    else:
        atol = 1e-4
        rtol = 1e-4

    # Residual tolerances are wider because the allreduce accumulation path differs
    # between iris (fp32 accumulation in Triton) and RCCL (implementation-defined).
    # With world_size=8 and bf16, the sum of 8 values can differ by multiple bf16 ULPs.
    if dtype == torch.float16:
        res_atol = 5e-2
    elif dtype == torch.bfloat16:
        res_atol = 2e-1
    else:
        res_atol = 1e-4

    # Check norm_out
    max_diff_norm = torch.abs(iris_norm_out - ref_norm_out).max().item()
    try:
        assert torch.allclose(iris_norm_out, ref_norm_out, atol=atol, rtol=rtol), (
            f"norm_out max diff: {max_diff_norm}, atol={atol}, rtol={rtol}\n"
            f"Rank {rank}: Iris fused AR+RMSNorm output doesn't match reference"
        )
    except AssertionError:
        # Print debug info
        print(f"Rank {rank}: norm_out max_diff={max_diff_norm}")
        print(f"  ref_norm_out[:3,:5]={ref_norm_out[:3, :5]}")
        print(f"  iris_norm_out[:3,:5]={iris_norm_out[:3, :5]}")
        raise

    # Check residual updated in-place
    max_diff_res = torch.abs(iris_residual - ref_residual).max().item()
    try:
        assert torch.allclose(iris_residual, ref_residual, atol=res_atol, rtol=res_atol), (
            f"residual max diff: {max_diff_res}, atol={res_atol}\n"
            f"Rank {rank}: Iris residual doesn't match reference after fused op"
        )
    except AssertionError:
        print(f"Rank {rank}: residual max_diff={max_diff_res}")
        raise

    # Cleanup
    ctx.barrier()
    del ctx
    import gc

    gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_fused_ar_rmsnorm_distribution(distribution):
    """Test fused AR+RMSNorm with both distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype = torch.float32
    tokens, hidden = 128, 1024
    eps = 1e-6

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    # Partial is different per rank
    torch.manual_seed(42 + rank)
    partial_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")

    # Residual and weight are replicated (tensor parallelism invariant)
    torch.manual_seed(42)
    residual_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")
    weight_data = torch.randn(hidden, dtype=dtype, device=f"cuda:{rank}")

    # Reference
    ref_partial = partial_data.clone()
    ref_residual = residual_data.clone()
    dist.all_reduce(ref_partial, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    ref_residual = ref_residual + ref_partial
    ref_norm_out = _rmsnorm_reference(ref_residual, weight_data, eps)

    # Iris
    iris_partial = ctx.zeros((tokens, hidden), dtype=dtype)
    iris_partial.copy_(partial_data)
    iris_residual = ctx.zeros((tokens, hidden), dtype=dtype)
    iris_residual.copy_(residual_data)
    ctx.barrier()

    config = Config(
        all_reduce_variant="two_shot",
        all_reduce_distribution=distribution,
    )
    iris_norm_out = ctx.ccl.all_reduce_rmsnorm(iris_partial, iris_residual, weight_data, eps=eps, config=config)
    torch.cuda.synchronize()

    atol = 1e-4
    max_diff = torch.abs(iris_norm_out - ref_norm_out).max().item()
    try:
        assert torch.allclose(iris_norm_out, ref_norm_out, atol=atol, rtol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: distribution={distribution} doesn't match reference"
        )
    except AssertionError:
        print(f"Rank {rank}: distribution={distribution}, max_diff={max_diff}")
        raise

    ctx.barrier()
    del ctx
    import gc

    gc.collect()


def test_fused_ar_rmsnorm_deterministic():
    """Test that repeated calls produce identical results."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    dtype = torch.float32
    tokens, hidden = 64, 512
    eps = 1e-6

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    # Partial is different per rank
    torch.manual_seed(42 + rank)
    partial_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")

    # Residual and weight are replicated
    torch.manual_seed(42)
    residual_data = torch.randn(tokens, hidden, dtype=dtype, device=f"cuda:{rank}")
    weight_data = torch.randn(hidden, dtype=dtype, device=f"cuda:{rank}")

    # Run twice with fresh inputs
    results = []
    for _ in range(2):
        iris_partial = ctx.zeros((tokens, hidden), dtype=dtype)
        iris_partial.copy_(partial_data)
        iris_residual = ctx.zeros((tokens, hidden), dtype=dtype)
        iris_residual.copy_(residual_data)
        ctx.barrier()

        config = Config(all_reduce_variant="two_shot", all_reduce_distribution=1)
        norm_out = ctx.ccl.all_reduce_rmsnorm(iris_partial, iris_residual, weight_data, eps=eps, config=config)
        torch.cuda.synchronize()
        results.append((norm_out.clone(), iris_residual.clone()))

    try:
        assert torch.equal(results[0][0], results[1][0]), "norm_out not deterministic"
        assert torch.equal(results[0][1], results[1][1]), "residual not deterministic"
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


def test_fused_ar_rmsnorm_shape_validation():
    """Test that invalid shapes raise appropriate errors."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    # 1D partial should fail
    partial_1d = ctx.zeros((128,), dtype=torch.float32)
    residual = ctx.zeros((4, 128), dtype=torch.float32)
    weight = torch.randn(128, dtype=torch.float32, device=f"cuda:{rank}")

    with pytest.raises(ValueError, match="partial must be 2D"):
        ctx.ccl.all_reduce_rmsnorm(partial_1d, residual, weight)

    # Mismatched shapes
    partial = ctx.zeros((4, 128), dtype=torch.float32)
    residual_wrong = ctx.zeros((4, 256), dtype=torch.float32)

    with pytest.raises(ValueError, match="doesn't match"):
        ctx.ccl.all_reduce_rmsnorm(partial, residual_wrong, weight)

    # Wrong weight size
    weight_wrong = torch.randn(256, dtype=torch.float32, device=f"cuda:{rank}")
    with pytest.raises(ValueError, match="doesn't match hidden"):
        ctx.ccl.all_reduce_rmsnorm(partial, residual, weight_wrong)

    ctx.barrier()
    del ctx
    import gc

    gc.collect()
