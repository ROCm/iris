# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather (along M) + GEMM operation using pull pattern.

Each rank has a sequence-sharded input A_local (M_local x K).
This operation computes C = all_gather_m(A_local) @ B by pulling
tiles from remote ranks on-demand during GEMM computation.

SP dataflow context:
  RMSNorm output: (M_local, K) per rank  <- M-sharded
    -> All-Gather along M: (M, K)        <- reconstruct full sequence
    -> Column-parallel GEMM: (M, K) @ (K, N) -> (M, N)

Constraint: M_local must be >= BLOCK_SIZE_M and M_local % BLOCK_SIZE_M == 0
(no tile straddling across rank boundaries).
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris
import iris.x
from iris.tracing.kernel_artifacts import iris_launch

from tritonblas.kernels.stages import GemmContext, ScheduleContext

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def _fused_all_gather_matmul_m_kernel(
    A_local,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    M_local,
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
    NUM_K_BLOCKS: tl.constexpr,
    NUM_M_BLOCKS_LOCAL: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    """Fused all-gather (along M) + GEMM kernel using pull pattern."""
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
        # TensorView describes A_local's layout: (M_local, K) per rank
        ctx = iris.DeviceContext.initialize(context_tensor, cur_rank, world_size)
        src_view = iris.x.make_tensor_view(A_local, M_local, K, stride_am, stride_ak)

        # Determine which source rank owns this M tile
        # global_m_start = pid_m * BLOCK_SIZE_M
        # source_rank = global_m_start // M_local
        # local_pid_m = (global_m_start - source_rank * M_local) // BLOCK_SIZE_M
        #
        # Because M_local % BLOCK_SIZE_M == 0, tiles never straddle rank boundaries.
        # We use an unrolled rank-match loop because iris.x.gather() requires
        # source_rank as tl.constexpr.

        # Precompute B column offsets for this output tile (constant across K iterations)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Unrolled rank-match loop: dispatch to correct constexpr source_rank
        for source_rank_id in range(world_size):
            # Check if this output M tile belongs to source_rank_id
            # M tiles for rank r span [r * NUM_M_BLOCKS_LOCAL, (r+1) * NUM_M_BLOCKS_LOCAL)
            rank_pid_m_start = source_rank_id * NUM_M_BLOCKS_LOCAL
            rank_pid_m_end = rank_pid_m_start + NUM_M_BLOCKS_LOCAL
            if pid_m >= rank_pid_m_start and pid_m < rank_pid_m_end:
                # Compute local M tile index within this rank's shard
                local_pid_m = pid_m - rank_pid_m_start

                # Use pre-computed loop bound (constexpr for static unrolling)
                loop_k = NUM_K_BLOCKS if EVEN_K else NUM_K_BLOCKS - 1

                # Loop over K dimension
                for k_block_idx in range(0, loop_k):
                    k_offset = k_block_idx * BLOCK_SIZE_K

                    # Create tile view for this (local_m, k) block
                    tile_k = local_pid_m * 0 + k_offset // BLOCK_SIZE_K
                    k_tile = iris.x.TileView(local_pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                    # Pull A tile from source_rank_id using gather primitive
                    a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                    # Load B tile using direct pointer arithmetic
                    rk = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
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
                    k_offset = loop_k * BLOCK_SIZE_K
                    tile_k = local_pid_m * 0 + k_offset // BLOCK_SIZE_K
                    k_tile = iris.x.TileView(local_pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                    # Pull A tile from source_rank_id using gather primitive
                    a = iris.x.gather(k_tile, src_view, source_rank_id, ctx)

                    # Load B tile with boundary handling
                    rk = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
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


def all_gather_matmul_m_preamble(
    shmem,
    A_local: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for all_gather_matmul_m (none needed for pull pattern)."""
    if config is None:
        config = FusedConfig()

    M_local, K = A_local.shape
    K_b, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K_b, f"K dimension mismatch: A_local has K={K}, B has K={K_b}"
    M = world_size * M_local

    return FusedWorkspace(
        operation="all_gather_matmul_m",
        shape=(M, N, K),
        dtype=A_local.dtype,
        world_size=world_size,
        prepared=True,
    )


def all_gather_matmul_m(
    shmem,
    output_tensor: torch.Tensor,
    A_local: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
) -> FusedWorkspace:
    """Fused all-gather (along M) and matrix multiplication using pull pattern.

    Gathers A_local from all ranks along the M (sequence/row) dimension,
    then computes C = all_gather_m(A_local) @ B.

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor (M, N) where M = world_size * M_local
        A_local: Each rank's sequence shard (M_local, K), in iris shmem
        B: Weight matrix (K, N), local CUDA tensor (same K on all ranks)
        bias: Optional bias vector
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace

    Returns:
        workspace: Updated workspace object
    """
    if config is None:
        config = FusedConfig()

    M_local, K = A_local.shape
    K_b, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K_b, f"K dimension mismatch: A_local has K={K}, B has K={K_b}"
    M = world_size * M_local
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Validate M_local alignment: no tile straddling across rank boundaries
    assert M_local >= config.block_size_m, (
        f"M_local ({M_local}) must be >= block_size_m ({config.block_size_m}). "
        f"Use smaller block sizes for small problems."
    )
    assert M_local % config.block_size_m == 0, (
        f"M_local ({M_local}) must be divisible by block_size_m ({config.block_size_m}) "
        f"to avoid tile straddling across rank boundaries."
    )
    assert K >= config.block_size_k, (
        f"K ({K}) must be >= block_size_k ({config.block_size_k}). "
        f"Use smaller block sizes for small problems."
    )
    assert N >= config.block_size_n, (
        f"N ({N}) must be >= block_size_n ({config.block_size_n}). Use smaller block sizes for small problems."
    )

    if workspace is None:
        workspace = all_gather_matmul_m_preamble(shmem, A_local, B, config)

    stride_am, stride_ak = A_local.stride()
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

    device = A_local.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0
    num_k_blocks = (K + config.block_size_k - 1) // config.block_size_k
    num_m_blocks_local = M_local // config.block_size_m

    # Launch single fused kernel
    grid = (num_sms,)
    iris_launch(
        _fused_all_gather_matmul_m_kernel,
        grid,
        A_local,
        B,
        output_tensor,
        bias_ptr,
        M,
        N,
        K,
        M_local,
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
        num_k_blocks,
        num_m_blocks_local,
        use_bias,
        even_k,
        config.allow_tf32,
        algorithm="all_gather_matmul_m",
        rank=rank,
        dtype=A_local.dtype,
    )

    if not async_op:
        shmem.barrier()

    return workspace
