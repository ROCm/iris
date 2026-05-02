# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for fused GEMM + reduce-scatter collective operation.

Reference computation:
    Each rank holds:
        input_shard = full_input[:, rank*H_shard : (rank+1)*H_shard]
        weight_shard of shape [H_shard, K]

    The full matmul is: full_result = full_input @ full_weight   (shape [tokens, K])
    where full_weight is the vertical stack of all ranks' weight_shards.

    After reduce-scatter along columns, rank j should hold:
        full_result[:, j*shard_size : (j+1)*shard_size]

    We compute the reference using torch.matmul + dist.all_reduce + column slicing,
    because dist.reduce_scatter_tensor scatters along dim=0 (rows) by default,
    but the kernel scatters along dim=1 (columns).
"""

import gc

import pytest
import torch
import torch.distributed as dist
import iris


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens, H, K",
    [
        (1, 256, 256),
        (32, 256, 256),
        (32, 512, 512),
        (128, 1024, 1024),
        (512, 4096, 4096),
    ],
)
def test_fused_gemm_rs_correctness(dtype, tokens, H, K):
    """Compare fused GEMM+RS against torch.matmul + dist.reduce_scatter_tensor."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    if H % world_size != 0 or K % world_size != 0:
        ctx.barrier()
        del ctx
        gc.collect()
        pytest.skip(f"H={H} or K={K} not divisible by world_size={world_size}")

    H_shard = H // world_size
    shard_size = K // world_size

    # Generate deterministic data: same full_input and full_weight on all ranks
    torch.manual_seed(42)
    full_input = torch.randn(tokens, H, dtype=dtype, device=f"cuda:{rank}")
    full_weight = torch.randn(H, K, dtype=dtype, device=f"cuda:{rank}")

    # Reference: each rank does partial matmul, then all-reduce + column slice.
    # The kernel scatters along dim=1 (columns), so rank j gets columns
    # j*shard_size : (j+1)*shard_size of the full reduced result.
    input_shard = full_input[:, rank * H_shard : (rank + 1) * H_shard].contiguous()
    weight_shard = full_weight[rank * H_shard : (rank + 1) * H_shard, :].contiguous()

    # Each rank's partial result
    partial = torch.matmul(input_shard, weight_shard)  # [tokens, K]

    # All-reduce to get full result, then slice columns for this rank
    full_result = partial.clone()
    dist.all_reduce(full_result, op=dist.ReduceOp.SUM)
    ref_output = full_result[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    torch.cuda.synchronize()

    # Iris fused path: input_shard and weight_shard must be on the symmetric heap
    iris_input = ctx.zeros((tokens, H_shard), dtype=dtype)
    iris_input.copy_(input_shard)

    # weight_shard is not on the heap (it's a local parameter, not communicated)
    # but it needs to be a regular CUDA tensor
    iris_weight = weight_shard.clone()

    ctx.barrier()

    iris_output = ctx.ccl.gemm_reduce_scatter(iris_input, iris_weight)
    torch.cuda.synchronize()

    # Compare — tolerances account for different accumulation order in fused kernel
    # (tl.dot tiles + atomic adds) vs cuBLAS + NCCL all_reduce.
    # Error grows with sqrt(inner_dim * world_size) due to non-associative FP.
    # The fused kernel accumulates H_shard elements in the GEMM k-loop via tl.dot
    # tiles, then world_size partial results are atomically added. Both stages
    # introduce accumulation-order differences vs cuBLAS + NCCL.
    scale = (H_shard * world_size) ** 0.5
    if dtype == torch.float32:
        atol = max(5e-2, scale * 5e-3)
        rtol = 1e-2
    else:
        atol = max(1e-1, scale * 5e-2)
        rtol = 5e-2

    max_diff = torch.abs(iris_output - ref_output).max().item()
    try:
        assert torch.allclose(iris_output, ref_output, atol=atol, rtol=rtol), (
            f"Rank {rank}: max diff = {max_diff}, atol={atol}\n"
            f"iris_output[:2,:4] = {iris_output[:2, :4]}\n"
            f"ref_output[:2,:4]  = {ref_output[:2, :4]}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_gemm_rs_workspace_reuse(dtype):
    """Verify that calling gemm_reduce_scatter twice with the same workspace produces correct results."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    tokens, H, K = 64, 512, 512
    H_shard = H // world_size
    shard_size = K // world_size

    torch.manual_seed(42)
    full_input = torch.randn(tokens, H, dtype=dtype, device=f"cuda:{rank}")
    full_weight = torch.randn(H, K, dtype=dtype, device=f"cuda:{rank}")

    input_shard = full_input[:, rank * H_shard : (rank + 1) * H_shard].contiguous()
    weight_shard = full_weight[rank * H_shard : (rank + 1) * H_shard, :].contiguous()

    partial = torch.matmul(input_shard, weight_shard)
    full_result = partial.clone()
    dist.all_reduce(full_result, op=dist.ReduceOp.SUM)
    ref_output = full_result[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    torch.cuda.synchronize()

    iris_input = ctx.zeros((tokens, H_shard), dtype=dtype)
    iris_input.copy_(input_shard)
    iris_weight = weight_shard.clone()

    # Prepare workspace once
    workspace = ctx.ccl.gemm_reduce_scatter_preamble(iris_input, iris_weight)
    ctx.barrier()

    # Call 1
    out1 = ctx.ccl.gemm_reduce_scatter(iris_input, iris_weight, workspace=workspace)
    torch.cuda.synchronize()

    # Call 2 (reuse workspace)
    out2 = ctx.ccl.gemm_reduce_scatter(iris_input, iris_weight, workspace=workspace)
    torch.cuda.synchronize()

    scale = (H_shard * world_size) ** 0.5
    atol = max(1e-1, scale * 5e-2)
    rtol = 5e-2
    try:
        assert torch.allclose(out1, ref_output, atol=atol, rtol=rtol), (
            f"Rank {rank}: call 1 failed, max diff = {torch.abs(out1 - ref_output).max().item()}"
        )
        assert torch.allclose(out2, ref_output, atol=atol, rtol=rtol), (
            f"Rank {rank}: call 2 failed, max diff = {torch.abs(out2 - ref_output).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_fused_gemm_rs_single_rank():
    """With a single rank, fused GEMM+RS should equal a plain matmul."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    world_size = ctx.get_num_ranks()

    if world_size != 1:
        ctx.barrier()
        del ctx
        gc.collect()
        pytest.skip("Single-rank test requires world_size=1")

    rank = ctx.get_rank()
    tokens, H, K = 64, 256, 256
    dtype = torch.float32

    torch.manual_seed(42)
    input_tensor = torch.randn(tokens, H, dtype=dtype, device=f"cuda:{rank}")
    weight = torch.randn(H, K, dtype=dtype, device=f"cuda:{rank}")

    ref = torch.matmul(input_tensor, weight)

    iris_input = ctx.zeros((tokens, H), dtype=dtype)
    iris_input.copy_(input_tensor)

    ctx.barrier()
    iris_output = ctx.ccl.gemm_reduce_scatter(iris_input, weight)
    torch.cuda.synchronize()

    atol = 1e-3
    try:
        assert torch.allclose(iris_output, ref, atol=atol), (
            f"Single-rank: max diff = {torch.abs(iris_output - ref).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_fused_gemm_rs_deterministic():
    """Debug test with deterministic fill values."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    tokens, H, K = 4, 128, 128
    dtype = torch.float32
    H_shard = H // world_size
    shard_size = K // world_size

    # All ranks use same input/weight: input filled with 1.0, weight filled with 1/H_shard
    # So each partial = tokens x K matrix with all entries = 1.0
    # After reduce-scatter sum: all entries = world_size * 1.0 = world_size
    input_shard = torch.full((tokens, H_shard), 1.0, dtype=dtype, device=f"cuda:{rank}")
    weight_shard = torch.full((H_shard, K), 1.0 / H_shard, dtype=dtype, device=f"cuda:{rank}")

    # Reference: all-reduce + column slice (kernel scatters along columns)
    partial = torch.matmul(input_shard, weight_shard)
    full_result = partial.clone()
    dist.all_reduce(full_result, op=dist.ReduceOp.SUM)
    ref_output = full_result[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    torch.cuda.synchronize()

    # Iris
    iris_input = ctx.zeros((tokens, H_shard), dtype=dtype)
    iris_input.copy_(input_shard)

    ctx.barrier()
    iris_output = ctx.ccl.gemm_reduce_scatter(iris_input, weight_shard)
    torch.cuda.synchronize()

    if rank == 0:
        print(f"\n  ref_output[0,:4] = {ref_output[0, :4]}")
        print(f"  iris_output[0,:4] = {iris_output[0, :4]}")
        print(f"  expected: all {float(world_size)}")

    atol = 1e-2
    try:
        assert torch.allclose(iris_output, ref_output, atol=atol), (
            f"Rank {rank}: max diff = {torch.abs(iris_output - ref_output).max().item()}\n"
            f"iris_output[0,:4] = {iris_output[0, :4]}\n"
            f"ref_output[0,:4]  = {ref_output[0, :4]}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_fused_gemm_rs_shape_validation():
    """Test that shape mismatches raise ValueError."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    dtype = torch.float32

    # Inner dimension mismatch
    iris_input = ctx.zeros((32, 128), dtype=dtype)
    bad_weight = torch.randn(64, 256, dtype=dtype, device=f"cuda:{rank}")  # 64 != 128

    ctx.barrier()
    with pytest.raises(ValueError, match="inner dimension mismatch"):
        ctx.ccl.gemm_reduce_scatter(iris_input, bad_weight)

    # K not divisible by world_size
    if world_size > 1:
        iris_input2 = ctx.zeros((32, 128), dtype=dtype)
        bad_weight2 = torch.randn(128, world_size * 64 + 1, dtype=dtype, device=f"cuda:{rank}")
        with pytest.raises(ValueError, match="divisible by world_size"):
            ctx.ccl.gemm_reduce_scatter(iris_input2, bad_weight2)

    ctx.barrier()
    del ctx
    gc.collect()


@pytest.mark.parametrize(
    "tokens",
    [3, 7, 17],  # Non-power-of-2
)
def test_fused_gemm_rs_non_pow2_tokens(tokens):
    """Test with non-power-of-2 token counts."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    H, K = 256, 256
    dtype = torch.bfloat16
    H_shard = H // world_size
    shard_size = K // world_size

    torch.manual_seed(42)
    full_input = torch.randn(tokens, H, dtype=dtype, device=f"cuda:{rank}")
    full_weight = torch.randn(H, K, dtype=dtype, device=f"cuda:{rank}")

    input_shard = full_input[:, rank * H_shard : (rank + 1) * H_shard].contiguous()
    weight_shard = full_weight[rank * H_shard : (rank + 1) * H_shard, :].contiguous()

    partial = torch.matmul(input_shard, weight_shard)
    full_result = partial.clone()
    dist.all_reduce(full_result, op=dist.ReduceOp.SUM)
    ref_output = full_result[:, rank * shard_size : (rank + 1) * shard_size].contiguous()
    torch.cuda.synchronize()

    iris_input = ctx.zeros((tokens, H_shard), dtype=dtype)
    iris_input.copy_(input_shard)
    iris_weight = weight_shard.clone()

    ctx.barrier()
    iris_output = ctx.ccl.gemm_reduce_scatter(iris_input, iris_weight)
    torch.cuda.synchronize()

    scale = (H_shard * world_size) ** 0.5
    atol = max(1e-1, scale * 5e-2)
    rtol = 5e-2
    try:
        assert torch.allclose(iris_output, ref_output, atol=atol, rtol=rtol), (
            f"Rank {rank}: tokens={tokens}, max diff = {torch.abs(iris_output - ref_output).max().item()}"
        )
    finally:
        ctx.barrier()
        del ctx
        gc.collect()
