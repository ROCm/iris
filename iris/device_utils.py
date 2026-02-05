# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side utility functions for Iris.

This module provides low-level device intrinsics for accessing hardware
information and timing within Triton kernels.
"""

import triton
import triton.language as tl


@triton.jit
def read_realtime():
    """
    Read GPU wall clock timestamp from s_memrealtime.

    Returns a 64-bit timestamp from a constant 100MHz clock (not affected
    by power modes or core clock frequency changes).

    Returns:
        int64: Current timestamp in cycles (100MHz constant clock)
    """
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    return tmp


@triton.jit
def get_xcc_id():
    """
    Get XCC (GPU chiplet) ID.

    Returns:
        int32: XCC ID for the current execution
    """
    xcc_id = tl.inline_asm_elementwise(
        asm="s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
        constraints=("=s"),
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    return xcc_id


@triton.jit
def get_cu_id():
    """
    Get Compute Unit ID.

    Returns:
        int32: CU ID for the current execution
    """
    cu_id = tl.inline_asm_elementwise(
        asm="s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)",
        constraints=("=s"),
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    return cu_id


@triton.jit
def get_se_id():
    """
    Get Shader Engine ID.

    Returns:
        int32: SE ID for the current execution
    """
    se_id = tl.inline_asm_elementwise(
        asm="s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 13, 3)",
        constraints=("=s"),
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    return se_id
