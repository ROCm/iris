# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Scatter operation.

Each rank has a column-sharded weight B_shard (K x N_shard) and a replicated
input A (M x K).  Each rank computes C_shard = A @ B_shard, then scatters its
column stripe to all other ranks so that every rank ends up with the full
output C (M x N) where N = world_size * N_shard.

This is useful for tensor-parallel workloads where weights are column-sharded
and the full output is needed on all ranks.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_matmul_all_scatter_kernel(
    A,  # (M, K)      - replicated across ranks
    B_shard,  # (K, N_shard) - each rank's column shard of weight matrix B
    C,  # (M, N)      - output, N = N_shard * world_size
    bias_ptr,
    M,
    N,
    K,
    N_shard,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    context_tensor: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """
    Fused GEMM + all-scatter kernel.

    Computes local GEMM tile and immediately scatters this rank's column stripe
    to all ranks via ``iris.x.all_scatter``.  No intermediate buffer needed.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # Create tritonblas views, context, and scheduler for GEMM
    # ═══════════════════════════════════════════════════════════════════════
    view_A = make_tensor_view(A, M, K, stride_am, stride_ak)
    view_B = make_tensor_view(B_shard, K, N_shard, stride_bk, stride_bn)
    gemm_ctx = GemmContext(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_sms=NUM_SMS,
        num_xcds=NUM_XCDS,
        group_size_m=GROUP_SIZE_M,
        even_k=EVEN_K,
        allow_tf32=ALLOW_TF32,
    )
    sched = ScheduleContext(M, N_shard, K, gemm_ctx)
    ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
    dst_view = iris.x.make_tensor_view(C, M, N, stride_cm, stride_cn)

    # Persistent loop over local tiles using scheduler
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        # Get tile coordinates with swizzling from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)

        # GEMM using tritonblas stages
        acc = gemm_ctx.reduce_axis(view_A, view_B, out_tile)

        # Add bias if provided
        if BIAS:
            rm, _ = out_tile.indices()
            bias_vec = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_vec[:, None]

        # Convert to output dtype
        c_tile = acc.to(C.type.element_ty)

        # Wrap result in a Tile object and scatter to all ranks
        tile_obj = iris.x.Tile(out_tile.pid_m, out_tile.pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N, c_tile)
        iris.x.all_scatter(tile_obj, dst_view, ctx)


def matmul_all_scatter_preamble(
    shmem,
    A: torch.Tensor,
    B_shard: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_scatter (none needed for scatter pattern)."""
    if config is None:
        config = FusedConfig()

    M, K = A.shape
    K2, N_shard = B_shard.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B_shard has K={K2}"

    N = N_shard * world_size

    return FusedWorkspace(
        operation="matmul_all_scatter",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )


def matmul_all_scatter(
    shmem,
    output: torch.Tensor,
    A: torch.Tensor,
    B_shard: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-scatter.

    Computes: output = all_scatter(A @ B_shard) along N dimension

    Each rank has B_shard of shape (K, N_shard) where N_shard = N / world_size.
    The operation computes C_shard = A @ B_shard on each rank and immediately
    scatters its column stripe to all other ranks via ``iris.x.all_scatter``,
    so that every rank ends up with the full C (M, N).

    This is a single-kernel implementation — no intermediate buffer needed.

    Args:
        shmem: Iris shmem context
        output: Output tensor C of shape (M, N) where N = N_shard * world_size
        A: Input matrix A of shape (M, K) — replicated across ranks
        B_shard: Column-sharded weight matrix of shape (K, N_shard)
        bias: Optional bias vector of shape (M,)
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        FusedWorkspace object

    Example:
        >>> N_shard = N // world_size
        >>> B_shard = shmem.randn((K, N_shard), dtype=torch.float16)
        >>> output = shmem.zeros((M, N), dtype=torch.float16)
        >>> shmem.ops.matmul_all_scatter(output, A, B_shard)
    """
    if config is None:
        config = FusedConfig()

    M, K = A.shape
    K2, N_shard = B_shard.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B_shard has K={K2}"

    N = N_shard * world_size
    assert output.shape == (M, N), f"Output must be ({M}, {N}), got {output.shape}"

    # Validate problem size against block sizes
    assert M >= config.block_size_m, (
        f"M ({M}) must be >= block_size_m ({config.block_size_m}). Use smaller block sizes for small problems."
    )
    assert K >= config.block_size_k, (
        f"K ({K}) must be >= block_size_k ({config.block_size_k}). Use smaller block sizes for small problems."
    )
    assert N_shard >= config.block_size_n, (
        f"N_shard ({N_shard}) must be >= block_size_n ({config.block_size_n}). "
        f"Use smaller block sizes for small problems."
    )

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_scatter_preamble(shmem, A, B_shard, config)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B_shard.stride()
    stride_cm, stride_cn = output.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output
        stride_bias = 1
        use_bias = False

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_matmul_all_scatter_kernel[grid](
        A,
        B_shard,
        output,
        bias_ptr,
        M,
        N,
        K,
        N_shard,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        shmem.get_device_context(),
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.block_size_k,
        config.group_size_m,
        num_sms,
        config.num_xcds,
        use_bias,
        even_k,
        config.allow_tf32,
    )

    if not async_op:
        shmem.barrier()

    return workspace
