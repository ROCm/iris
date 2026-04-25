# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side utility functions for Iris.

Provides portable device intrinsics for timestamps and hardware topology
that work across all supported AMD GPU architectures. Uses Triton's
architecture-aware APIs (``tl.extra.hip``) where available.
"""

import triton
import triton.language as tl
from triton.language.target_info import is_hip_cdna3, is_hip_cdna4

try:
    from triton.language.extra.hip import memrealtime as _memrealtime
    from triton.language.extra.hip import smid as _smid

    _HAS_HIP_INTRINSICS = True
except ImportError:
    _HAS_HIP_INTRINSICS = False


if _HAS_HIP_INTRINSICS:

    @triton.jit
    def read_realtime():
        """
        Read GPU wall clock timestamp.

        Returns a 64-bit value from the GPU's constant-frequency real-time
        counter (100 MHz, unaffected by power states or clock gating).

        Delegates to ``tl.extra.hip.memrealtime()`` which emits the correct
        instruction for each architecture family.

        Returns:
            int64: Current timestamp in cycles (100 MHz constant clock)
        """
        return _memrealtime()

    @triton.jit
    def get_cu_id():
        """
        Get compute-unit / workgroup-processor ID for the current wave.

        Delegates to ``tl.extra.hip.smid()`` which reads the appropriate
        hardware register for each architecture family (CU_ID on CDNA,
        WGP_ID on RDNA).

        Returns:
            int32: CU / WGP ID for the current execution
        """
        return _smid()
else:

    @triton.jit
    def read_realtime():
        """Fallback stub when HIP intrinsics are missing."""
        tl.static_assert(False, "memrealtime is unavailable in this Triton build")
        return tl.cast(0, tl.int64)

    @triton.jit
    def get_cu_id():
        """Fallback stub when HIP intrinsics are missing."""
        tl.static_assert(False, "smid is unavailable in this Triton build")
        return tl.cast(0, tl.int32)


@triton.jit
def get_xcc_id():
    """
    Get XCC (GPU chiplet) ID.

    On multi-XCC parts (CDNA3/CDNA4) reads ``HW_REG_XCC_ID``.
    On single-die architectures returns 0.

    Returns:
        int32: XCC ID for the current execution
    """
    if is_hip_cdna3() or is_hip_cdna4():
        return tl.inline_asm_elementwise(
            asm="s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
            constraints=("=s"),
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        return tl.cast(0, tl.int32)
