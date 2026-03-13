# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM operation using pull pattern.

Each rank has a column-sharded input A_sharded (M x K_local).
This operation computes C = all_gather(A_sharded) @ B by pulling
tiles from remote ranks on-demand during GEMM computation.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x

from tritonblas.kernels.stages import GemmContext, ScheduleContext

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_all_gather_matmul_kernel(
    A_sharded,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    K_local,
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
    NUM_K_BLOCKS_LOCAL: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Fused all-gather + GEMM kernel using pull pattern."""
    # ═══════════════════════════════════════════════════════════════════════
    # Create tritonblas context and scheduler for GEMM configuration
    # ═══════════════════════════════════════════════════════════════════════
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
    sched = ScheduleContext(M, N, K, gemm_ctx)

    # Persistent loop over output tiles using scheduler
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        # Get tile coordinates with swizzling from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)
        pid_m = out_tile.pid_m
        pid_n = out_tile.pid_n

        # Initialize accumulator using GemmContext
        acc = gemm_ctx.init_accumulator()

        # Create DeviceContext and TensorView for gather operations
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        src_view = iris.x.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

        # Precompute B column offsets for this output tile (constant across K iterations)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Loop over all ranks to pull and accumulate
        # Note: K = world_size * K_local, so we iterate over each rank's K_local contribution
        for source_rank_id in range(world_size):
            # Use pre-computed loop bound (constexpr for static unrolling)
            loop_k_local = NUM_K_BLOCKS_LOCAL if EVEN_K else NUM_K_BLOCKS_LOCAL - 1

            # Loop over K dimension for this rank's shard
            for k_block_idx in range(0, loop_k_local):
                k_offset = k_block_idx * BLOCK_SIZE_K

                # Create tile view for this K block
                # Promote tile_k to tensor (TileView expects tl.tensor for pid_n)
                tile_k = pid_m * 0 + k_offset // BLOCK_SIZE_K
                k_tile = iris.x.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                # Load B tile using direct pointer arithmetic
                # Compute global K row index for B matrix
                global_k_offset = source_rank_id * K_local + k_block_idx * BLOCK_SIZE_K
                rk = global_k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk = tl.max_contiguous(tl.multiple_of(rk % K, BLOCK_SIZE_K), BLOCK_SIZE_K)
                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(B_ptrs)

                # Accumulate
                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

            # Handle remaining K elements if not evenly divisible
            if not EVEN_K:
                k_offset = loop_k_local * BLOCK_SIZE_K
                # Promote tile_k to tensor (TileView expects tl.tensor for pid_n)
                tile_k = pid_m * 0 + k_offset // BLOCK_SIZE_K
                k_tile = iris.x.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                # Load B tile with boundary handling
                global_k_offset = source_rank_id * K_local + loop_k_local * BLOCK_SIZE_K
                rk = global_k_offset + tl.arange(0, BLOCK_SIZE_K)
                B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b_mask = (rk[:, None] < K) & (rn[None, :] < N)
                b = tl.load(B_ptrs, mask=b_mask, other=0.0)

                if ALLOW_TF32:
                    acc = tl.dot(a, b, acc, allow_tf32=True)
                else:
                    acc += tl.dot(a, b, allow_tf32=False)

        # Add bias if provided
        if BIAS:
            rm, _ = out_tile.indices()
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_vector[:, None]

        # Convert to output dtype
        c = acc.to(C.type.element_ty)

        # Store result using tritonblas Tile
        rm, rn = out_tile.indices()
        C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptr, c, mask=mask)


def all_gather_matmul_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for all_gather_matmul (none needed for pull pattern)."""
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"

    return FusedWorkspace(
        operation="all_gather_matmul",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        prepared=True,
    )


def all_gather_matmul(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """Fused all-gather and matrix multiplication using pull pattern."""
    if config is None:
        config = FusedConfig()

    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Validate problem size against block sizes
    assert M >= config.block_size_m, (
        f"M ({M}) must be >= block_size_m ({config.block_size_m}). Use smaller block sizes for small problems."
    )
    assert K_local >= config.block_size_k, (
        f"K_local ({K_local}) must be >= block_size_k ({config.block_size_k}). "
        f"Use smaller block sizes for small problems."
    )
    assert N >= config.block_size_n, (
        f"N ({N}) must be >= block_size_n ({config.block_size_n}). Use smaller block sizes for small problems."
    )

    if workspace is None:
        workspace = all_gather_matmul_preamble(shmem, A_sharded, B, config)

    stride_am, stride_ak = A_sharded.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A_sharded.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K_local % config.block_size_k == 0
    num_k_blocks_local = (K_local + config.block_size_k - 1) // config.block_size_k

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_all_gather_matmul_kernel[grid](
        A_sharded,
        B,
        output_tensor,
        bias_ptr,
        M,
        N,
        K,
        K_local,
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
        num_k_blocks_local,
        use_bias,
        even_k,
        config.allow_tf32,
    )

    if not async_op:
        shmem.barrier()

    return workspace
