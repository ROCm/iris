# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common utility functions for Iris CCL operations.
"""

import triton
import triton.language as tl


@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups: tl.constexpr, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """
    Transform program ID for chiplet-aware workgroup distribution.

    This function redistributes workgroups across multiple XCDs (chiplets) in chunks
    to improve load balancing and memory access patterns.

    Args:
        pid: Program ID to transform
        num_workgroups: Total number of workgroups
        num_xcds: Number of XCDs (chiplets)
        chunk_size: Size of chunks for distribution

    Returns:
        Transformed program ID
    """
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        return pid

    local_pid = pid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size

    xcd = pid % num_xcds
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid
