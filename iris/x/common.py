# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common utilities for iris.x tile-level primitives.
"""

import triton
import triton.language as tl


@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups: tl.constexpr, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """
    Transform program ID to distribute work across XCDs in a chunked pattern.

    This utility is used for better load balancing across chiplet architectures.
    """
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid


@triton.jit()
def compute_tile_indices(
    pid_m,
    pid_n,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Compute row and column indices for a tile given pid_m and pid_n.

    Returns:
        rm: Row indices for the tile
        rn: Column indices for the tile
        mask: Mask for valid elements within bounds
    """
    # Calculate base indices without modulo to avoid double-counting when blocks are larger than dimensions
    rm_base = pid_m * BLOCK_SIZE_M
    rn_base = pid_n * BLOCK_SIZE_N
    rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
    rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    # Create mask to prevent out-of-bounds access
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    return rm, rn, mask


@triton.jit()
def compute_tile_offsets(
    rm,
    rn,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
):
    """
    Compute input and output offsets for a tile given row/column indices and strides.

    Returns:
        input_offset: Offset for input tensor
        output_offset: Offset for output tensor
    """
    input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
    output_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
    return input_offset, output_offset
