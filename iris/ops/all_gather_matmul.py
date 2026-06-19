# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM operation using pull pattern.

Each rank has a column-sharded input A_sharded (M x K_local).
This operation computes C = all_gather(A_sharded) @ B by pulling
tiles from remote ranks on-demand during GEMM computation.
"""

import logging
from typing import Optional
import torch
import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch

from tritonblas.matmul import _make_matmul_selector
from tritonblas.kernels.stages import GemmContext, ScheduleContext

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
    CHUNK_SIZE: tl.constexpr,
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
        chunk_size=CHUNK_SIZE,
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
        src_view = iris.make_tensor_view(A_sharded, M, K_local, stride_am, stride_ak)

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
                k_tile = iris.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = ctx.gather(k_tile, src_view, source_rank_id)

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
                k_tile = iris.TileView(pid_m, tile_k, BLOCK_SIZE_M, BLOCK_SIZE_K)

                # Pull A tile from source_rank_id using gather primitive
                a = ctx.gather(k_tile, src_view, source_rank_id)

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
        C_ptr = C + rm.to(tl.int64)[:, None] * stride_cm + rn.to(tl.int64)[None, :] * stride_cn
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptr, c, mask=mask)


def all_gather_matmul_preamble(
    shmem,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    selector=None,
    out_dtype: Optional[torch.dtype] = None,
) -> FusedWorkspace:
    """Prepare selector and launch metadata for all_gather_matmul."""
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"

    if selector is None:
        c_dtype = A_sharded.dtype if out_dtype is None else out_dtype
        selector = _make_matmul_selector(
            M,
            N,
            K,
            A_sharded.dtype,
            B.dtype,
            c_dtype,
            A_sharded.device,
            streamk=False,
        )

    ws = FusedWorkspace(
        operation="all_gather_matmul",
        shape=(M, N, K),
        dtype=A_sharded.dtype,
        world_size=world_size,
        variant="origami",
        prepared=True,
    )
    ws.selector = selector
    ws.launch_params = _all_gather_matmul_launch_params(
        M,
        N,
        K_local,
        selector,
        A_sharded.device,
    )

    return ws


def _default_chunk_size(total_tiles: int, group_size_m: int, num_xcds: int) -> int:
    chunk_size = group_size_m * group_size_m
    if num_xcds > 0:
        chunk_size = min(chunk_size, max(1, total_tiles // num_xcds))
    return max(1, chunk_size)


def _selector_active_cus(selector, device: torch.device) -> int:
    active_cus = getattr(selector, "_ACTIVE_CU", None)
    if active_cus is None or active_cus <= 0:
        props = torch.cuda.get_device_properties(device)
        active_cus = props.multi_processor_count
    return int(active_cus)


def _all_gather_matmul_launch_params(
    M: int,
    N: int,
    K_local: int,
    selector,
    device: torch.device,
) -> dict:
    block_size_m = selector.block_m
    block_size_n = selector.block_n
    block_size_k = selector.block_k
    group_size_m = selector.group_m
    # Origami calls this num_sms, but it is the XCD/chiplet workgroup mapping
    # count used by chiplet_transform_chunked, not the persistent launch grid.
    num_xcds = selector.num_sms
    if num_xcds <= 0:
        num_xcds = 1

    num_tiles_m = (M + block_size_m - 1) // block_size_m
    num_tiles_n = (N + block_size_n - 1) // block_size_n
    total_tiles = num_tiles_m * num_tiles_n

    # This pull-pattern kernel is persistent. Use the active CU count as the
    # launch grid, then compute chunking against that launch grid so XCD
    # remapping does not degenerate to identity.
    num_sms = min(_selector_active_cus(selector, device), total_tiles)
    chunk_size = _default_chunk_size(num_sms, group_size_m, num_xcds)

    return {
        "block_size_m": block_size_m,
        "block_size_n": block_size_n,
        "block_size_k": block_size_k,
        "group_size_m": group_size_m,
        "num_xcds": num_xcds,
        "num_tiles_m": num_tiles_m,
        "num_tiles_n": num_tiles_n,
        "total_tiles": total_tiles,
        "num_sms": num_sms,
        "chunk_size": chunk_size,
        "num_k_blocks_local": (K_local + block_size_k - 1) // block_size_k,
        "even_k": K_local % block_size_k == 0,
        "num_warps": 8,
        "num_stages": getattr(selector, "num_stages", 2),
        "matrix_instr_nonkdim": 16,
    }


def all_gather_matmul(
    shmem,
    output_tensor: torch.Tensor,
    A_sharded: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    async_op: bool = False,
    workspace: Optional[FusedWorkspace] = None,
    selector=None,
) -> FusedWorkspace:
    """Fused all-gather and matrix multiplication using pull pattern."""
    M, K_local = A_sharded.shape
    K, N = B.shape
    world_size = shmem.get_num_ranks()
    rank = shmem.get_rank()

    from iris.host.logging.logging import _log_rank

    _log_rank(
        logging.DEBUG,
        "all_gather_matmul: shape=(%d,%d,%d) dtype=%s rank=%d/%d",
        M,
        N,
        K,
        A_sharded.dtype,
        rank,
        world_size,
        rank=rank,
        num_ranks=world_size,
    )

    expected_K = world_size * K_local
    assert K == expected_K, f"K ({K}) must equal world_size ({world_size}) * K_local ({K_local})"
    assert output_tensor.shape == (M, N), f"Output must be ({M}, {N}), got {output_tensor.shape}"

    if workspace is None:
        workspace = all_gather_matmul_preamble(
            shmem,
            A_sharded,
            B,
            selector=selector,
            out_dtype=output_tensor.dtype,
        )

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

    launch = workspace.launch_params
    block_size_m = launch["block_size_m"]
    block_size_n = launch["block_size_n"]
    block_size_k = launch["block_size_k"]
    group_size_m = launch["group_size_m"]
    num_xcds = launch["num_xcds"]
    num_sms = launch["num_sms"]
    chunk_size = launch["chunk_size"]
    grid = (num_sms,)

    launch_kwargs = {
        "num_warps": launch["num_warps"],
        "num_stages": launch["num_stages"],
        "matrix_instr_nonkdim": launch["matrix_instr_nonkdim"],
    }

    iris_launch(
        _fused_all_gather_matmul_kernel,
        grid,
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
        block_size_m,
        block_size_n,
        block_size_k,
        group_size_m,
        num_sms,
        num_xcds,
        chunk_size,
        launch["num_k_blocks_local"],
        use_bias,
        launch["even_k"],
        torch.backends.cuda.matmul.allow_tf32,
        algorithm="all_gather_matmul",
        rank=rank,
        dtype=A_sharded.dtype,
        **launch_kwargs,
    )

    if not async_op:
        shmem.barrier()

    return workspace
