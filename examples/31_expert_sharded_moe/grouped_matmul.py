# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Simplified grouped/ragged GEMM for expert-parallel MoE.

Ported / simplified from triton_kernels matmul:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_matmul.py

Non-persistent, non-TMA tiled GEMM that handles variable-length expert
batches described by ragged metadata (slice_offs, slice_sizes).

  Y[offs[e]:offs[e+1], :] = X[offs[e]:offs[e+1], :] @ W[e, :, :] + bias[e, :]
"""

import torch
import triton
import triton.language as tl
from ragged_metadata import RaggedTensorMetadata


@triton.jit
def _grouped_matmul_kernel(
    X_ptr,
    stride_x_m,
    stride_x_k,
    W_ptr,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    B_ptr,
    stride_b_e,
    stride_b_n,
    Y_ptr,
    stride_y_m,
    stride_y_n,
    SliceOffs_ptr,
    SliceSizes_ptr,
    n_experts,
    K,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled GEMM over ragged expert batches.

    Grid: (n_n_tiles * n_experts,)
    Each program handles one (expert, n_tile) pair and loops over M tiles.
    """
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)

    expert_id = pid // n_n_tiles
    pid_n = pid % n_n_tiles

    if expert_id >= n_experts:
        return

    # int64 to prevent pointer-offset overflow when n_experts * K * N > 2^31
    expert_id = expert_id.to(tl.int64)
    slice_off = tl.load(SliceOffs_ptr + expert_id).to(tl.int64)
    slice_size = tl.load(SliceSizes_ptr + expert_id)
    if slice_size == 0:
        return

    n_m_tiles = tl.cdiv(slice_size, BLOCK_M)

    for pid_m in range(0, n_m_tiles):
        offs_m = slice_off + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < slice_size

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = X_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = W_ptr + expert_id * stride_w_e + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
            w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(x, w)

        if HAS_BIAS:
            b_ptrs = B_ptr + expert_id * stride_b_e + offs_n * stride_b_n
            bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc += bias[None, :]

        y_ptrs = Y_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def grouped_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    ragged_metadata: RaggedTensorMetadata,
) -> torch.Tensor:
    """Ragged grouped GEMM: one matmul per expert slice.

    Args:
        x: (total_tokens, K) activations in expert-sorted order.
        w: (n_experts, K, N) weight matrices.
        bias: (n_experts, N) bias vectors, or None.
        ragged_metadata: which rows of x belong to which expert.

    Returns:
        y: (total_tokens, N) output in the same ragged layout as x.
    """
    total_tokens, K = x.shape
    n_experts, _, N = w.shape
    device = x.device

    y = torch.zeros((total_tokens, N), dtype=x.dtype, device=device)

    BLOCK_M = 128
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    BLOCK_K = min(triton.next_power_of_2(K), 64)

    n_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (n_n_tiles * n_experts,)

    _grouped_matmul_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        bias if bias is not None else x,
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        y,
        y.stride(0),
        y.stride(1),
        ragged_metadata.slice_offs,
        ragged_metadata.slice_sizes,
        n_experts,
        K,
        N,
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return y
