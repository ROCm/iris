# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + All-Gather operation using SDMA (copy engine) for scatter.

Each rank has a row-sharded input A_local (M_local x K) and computes C_local = A_local @ B.
Then scatters C_local tiles to form the full C (M x N) where M = world_size * M_local.

This variant uses SDMA hardware for data movement instead of compute shader scatter.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

from .config import FusedConfig
from .workspace import FusedWorkspace


@triton.jit()
def wait_cnt():
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit()
def _fused_matmul_all_gather_copy_engine_kernel(
    A,  # (M_local, K) - each rank's local input
    B,  # (K, N) - replicated across ranks
    C_gathered,  # (M, N) - gathered output (M = M_local * world_size)
    bias_ptr,
    flags,
    M_local,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    heap_bases: tl.tensor,
    copy_engine_ctx: tl.tensor,
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
    USE_COPY_ENGINE: tl.constexpr,
):
    """
    Fused GEMM + all-gather kernel using SDMA (copy engine) for scatter.

    Computes local GEMM tile, stores to memory, then uses SDMA to scatter
    to all ranks. Per-SM flag synchronization ensures completion.
    """
    pid = tl.program_id(0)

    # ═══════════════════════════════════════════════════════════════════════
    # Create tritonblas views, context, and scheduler for GEMM
    # ═══════════════════════════════════════════════════════════════════════
    tensorA = make_tensor_view(A, M_local, K, stride_am, stride_ak)
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
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
    sched = ScheduleContext(M_local, N, K, gemm_ctx)

    # Persistent loop over local tiles using scheduler
    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        # Get tile coordinates with swizzling from scheduler
        out_tile = sched.get_tile_from_idx(tile_id)

        # ═══════════════════════════════════════════════════════════════════
        # GEMM Phase: Compute tile using tritonblas stages
        # ═══════════════════════════════════════════════════════════════════
        acc = gemm_ctx.reduce_axis(tensorA, tensorB, out_tile)

        # Add bias if provided
        if BIAS:
            rm, _ = out_tile.indices()
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M_local, other=0.0)
            acc = acc + bias_vector[:, None]

        # Convert to output dtype
        c = acc.to(C_gathered.type.element_ty)

        # ═══════════════════════════════════════════════════════════════════
        # Scatter Phase: Write to all ranks using SDMA or compute scatter
        # ═══════════════════════════════════════════════════════════════════
        # Get tile indices from out_tile (tritonblas)
        rm, rn = out_tile.indices()

        # Calculate global offset: rank's rows start at cur_rank * M_local
        global_offset = (rm + cur_rank * M_local)[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = ((rm + cur_rank * M_local)[:, None] < M) & (rn[None, :] < N)

        if USE_COPY_ENGINE:
            # Store locally first (SDMA needs data in memory)
            tl.store(C_gathered + global_offset, c, mask=mask, cache_modifier=".wt")
            wait_cnt()
            tl.debug_barrier()

            # SDMA scatter to remote ranks
            for remote_rank in range(world_size):
                if remote_rank != cur_rank:
                    iris.put(
                        C_gathered + global_offset,  # from_ptr
                        C_gathered + global_offset,  # to_ptr (same logical position)
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        copy_engine_ctx,
                        stride_tm=stride_cm,
                        stride_tn=stride_cn,
                        stride_fm=stride_cm,
                        stride_fn=stride_cn,
                        mask=mask,
                        USE_COPY_ENGINE=True,
                        IS_2D_COPY=True,
                        from_base_ptr=C_gathered,
                        to_base_ptr=C_gathered,
                    )
        else:
            # Fallback: baseline scatter (for comparison)
            for remote_rank in range(world_size):
                if remote_rank == cur_rank:
                    tl.store(C_gathered + global_offset, c, mask=mask)
                else:
                    iris.store(
                        C_gathered + global_offset,
                        c,
                        cur_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                    )

    # ═══════════════════════════════════════════════════════════════════════
    # Synchronization: Signal completion to all ranks
    # ═══════════════════════════════════════════════════════════════════════
    # tl.debug_barrier()
    # # Signal other ranks that all our puts/stores are complete
    # for remote_rank in range(world_size):
    #     if remote_rank != cur_rank:
    #         iris.atomic_add(
    #             flags + (pid * world_size) + cur_rank,
    #             1,
    #             cur_rank,
    #             remote_rank,
    #             heap_bases,
    #             sem="release",
    #             scope="sys",
    #             copy_engine_ctx=copy_engine_ctx,
    #             USE_COPY_ENGINE=USE_COPY_ENGINE,
    #         )

    # # Wait for other ranks to signal us
    # for remote_rank in range(world_size):
    #     if remote_rank != cur_rank:
    #         while tl.load(flags + (pid * world_size) + remote_rank, cache_modifier=".cv", volatile=True) != 1:
    #             pass


def matmul_all_gather_copy_engine_preamble(
    shmem,
    A: torch.Tensor,
    B: torch.Tensor,
    config: Optional[FusedConfig] = None,
) -> FusedWorkspace:
    """Allocate workspace for matmul_all_gather_copy_engine including per-SM flags."""
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size

    # Allocate per-SM flags: num_sms * world_size
    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    ws = FusedWorkspace(
        operation="matmul_all_gather_copy_engine",
        shape=(M, N, K),
        dtype=A.dtype,
        world_size=world_size,
        prepared=True,
    )

    # Allocate locks/flags for per-SM synchronization
    ws.locks = shmem.zeros((num_sms * world_size,), dtype=torch.int32)

    return ws


def matmul_all_gather_copy_engine(
    shmem,
    output_tensor: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    config: Optional[FusedConfig] = None,
    workspace: Optional[FusedWorkspace] = None,
    use_copy_engine: bool = True,
) -> FusedWorkspace:
    """
    Fused matrix multiplication and all-gather using SDMA (copy engine) for scatter.

    Computes: output = all_gather(A @ B + bias) along M dimension

    Each rank has A of shape (M_local, K) where M_local = M / world_size.
    The operation computes C_local = A @ B on each rank and uses SDMA hardware
    to scatter the tiles to all ranks (all-gather pattern).

    Args:
        shmem: Iris shmem context
        output_tensor: Output tensor C of shape (M, N) where M = M_local * world_size
        A: Input matrix A of shape (M_local, K)
        B: Input matrix B of shape (K, N)
        bias: Optional bias vector (M_local,)
        async_op: If False, performs barrier at end
        config: Optional FusedConfig for tuning
        workspace: Optional pre-allocated workspace
        use_copy_engine: If True, use SDMA; if False, use compute shader scatter

    Returns:
        FusedWorkspace object
    """
    if config is None:
        config = FusedConfig()

    M_local, K = A.shape
    K2, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    assert K == K2, f"Inner dimensions must match: A has K={K}, B has K={K2}"

    M = M_local * world_size
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    # Allocate workspace if not provided
    if workspace is None:
        workspace = matmul_all_gather_copy_engine_preamble(shmem, A, B, config)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = output_tensor.stride()

    if bias is not None:
        assert bias.shape[0] == M_local
        bias_ptr = bias
        stride_bias = bias.stride()[0] if bias.dim() > 0 else 1
        use_bias = True
    else:
        bias_ptr = output_tensor
        stride_bias = 1
        use_bias = False

    device = A.device
    num_sms = config.num_sms
    if num_sms is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count

    even_k = K % config.block_size_k == 0

    # Reset flags before launch
    workspace.locks.zero_()
    shmem.barrier()

    # Launch single fused kernel
    grid = (num_sms,)
    _fused_matmul_all_gather_copy_engine_kernel[grid](
        A,
        B,
        output_tensor,
        bias_ptr,
        workspace.locks,
        M_local,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        shmem.get_heap_bases(),
        shmem.get_copy_engine_ctx(),
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
        use_copy_engine,
    )

    if not async_op:
        shmem.barrier()

    return workspace
