# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Fused All-Gather + GEMM operation using pull pattern.

Each rank has a column-sharded input A_sharded (M x K_local).
This operation computes C = all_gather(A_sharded) @ B by pulling
tiles from remote ranks on-demand during GEMM computation.

Uses raw Triton + iris.load for XGMI remote reads.
"""

from typing import Optional
import torch
import triton
import triton.language as tl
import iris

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
    heap_bases: tl.tensor,
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
    """Fused all-gather + GEMM kernel using pull pattern with raw Triton."""
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Swizzled tile indexing for better L2 locality
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Initialize fp32 accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Precompute row indices for this output tile
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # Precompute column indices for B / output
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        is_full_m = (pid_m * BLOCK_SIZE_M + BLOCK_SIZE_M) <= M
        is_full_n = (pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N) <= N

        # Loop over all ranks to pull their K_local shard and accumulate
        for source_rank_id in tl.static_range(world_size):
            loop_k_local = NUM_K_BLOCKS_LOCAL if EVEN_K else NUM_K_BLOCKS_LOCAL - 1

            for k_block_idx in range(0, loop_k_local):
                k_offset = k_block_idx * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_local = tl.max_contiguous(tl.multiple_of(rk_local, BLOCK_SIZE_K), BLOCK_SIZE_K)

                # Load A tile: A_sharded[rm, rk_local] from source_rank_id
                a_ptrs = A_sharded + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                if source_rank_id == cur_rank:
                    # Local rank: direct HBM read (fast path)
                    if is_full_m:
                        a = tl.load(a_ptrs)
                    else:
                        a_mask = rm[:, None] < M
                        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                else:
                    # Remote rank: XGMI read via iris.load
                    if is_full_m:
                        a = iris.load(a_ptrs, cur_rank, source_rank_id, heap_bases, hint=(1, BLOCK_SIZE_K))
                    else:
                        a_mask = rm[:, None] < M
                        a = iris.load(a_ptrs, cur_rank, source_rank_id, heap_bases, mask=a_mask, hint=(1, BLOCK_SIZE_K))

                # Load B tile: B[global_k, rn]
                global_k_offset = source_rank_id * K_local + k_block_idx * BLOCK_SIZE_K
                rk_global = global_k_offset + tl.arange(0, BLOCK_SIZE_K)
                rk_global = tl.max_contiguous(tl.multiple_of(rk_global, BLOCK_SIZE_K), BLOCK_SIZE_K)
                b_ptrs = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                if is_full_n:
                    b = tl.load(b_ptrs)
                else:
                    b_mask = rn[None, :] < N
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)

            # Handle remaining K elements if not evenly divisible
            if not EVEN_K:
                k_offset = loop_k_local * BLOCK_SIZE_K
                rk_local = k_offset + tl.arange(0, BLOCK_SIZE_K)

                # A tile with K boundary mask
                a_ptrs = A_sharded + rm[:, None] * stride_am + rk_local[None, :] * stride_ak
                a_mask = (rm[:, None] < M) & (rk_local[None, :] < K_local)
                if source_rank_id == cur_rank:
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                else:
                    a = iris.load(a_ptrs, cur_rank, source_rank_id, heap_bases, mask=a_mask, hint=(1, BLOCK_SIZE_K))

                # B tile with K boundary mask
                global_k_offset = source_rank_id * K_local + loop_k_local * BLOCK_SIZE_K
                rk_global = global_k_offset + tl.arange(0, BLOCK_SIZE_K)
                b_ptrs = B + rk_global[:, None] * stride_bk + rn[None, :] * stride_bn
                b_mask = (rk_global[:, None] < K) & (rn[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)

        # Add bias if provided
        if BIAS:
            bias_vector = tl.load(bias_ptr + rm * stride_bias, mask=rm < M, other=0.0)
            acc = acc + bias_vector[:, None]

        # Store output
        c = acc.to(C.type.element_ty)
        c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        if is_full_m and is_full_n:
            tl.store(c_ptrs, c)
        else:
            c_mask = (rm[:, None] < M) & (rn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)


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

    heap_bases = shmem.get_heap_bases()

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
        heap_bases,
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
