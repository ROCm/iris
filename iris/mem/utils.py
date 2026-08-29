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

# Probe each intrinsic separately. They were added to
# ``triton.language.extra.hip`` at different times, so a build can export one
# and not the other -- upstream Triton at the pinned commit ships
# ``memrealtime`` but not ``smid``. Guarding both behind a single try/except
# meant a missing ``smid`` also disabled ``read_realtime``, and the fallback
# was ``static_assert(False)``, which fails compilation of every kernel that
# records a trace event rather than degrading the trace.
try:
    from triton.language.extra.hip import memrealtime as _memrealtime

    _HAS_MEMREALTIME = True
except ImportError:
    _HAS_MEMREALTIME = False

try:
    from triton.language.extra.hip import smid as _smid

    _HAS_SMID = True
except ImportError:
    _HAS_SMID = False


if _HAS_MEMREALTIME:

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
else:

    @triton.jit
    def read_realtime():
        """
        Read GPU wall clock timestamp on builds without ``tl.extra.hip.memrealtime``.

        Emits the same instruction the intrinsic would. gfx11/gfx12 use a
        message rather than ``s_memrealtime``; there is no portable fallback for
        those here, so they report 0 and timestamps are simply unavailable.

        Returns:
            int64: Timestamp in cycles, or 0 where unsupported
        """
        if is_hip_cdna3() or is_hip_cdna4():
            return tl.inline_asm_elementwise(
                asm="s_memrealtime $0\n\ts_waitcnt vmcnt(0)",
                constraints=("=s"),
                args=[],
                dtype=tl.int64,
                is_pure=False,
                pack=1,
            )
        else:
            return tl.cast(0, tl.int64)


if _HAS_SMID:

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
    def get_cu_id():
        """
        Get compute-unit ID on builds without ``tl.extra.hip.smid``.

        Reads CU_ID out of ``HW_REG_HW_ID`` directly, the same mechanism
        ``get_xcc_id`` below uses. The field is 4 bits, so this identifies the
        CU within its shader engine rather than globally; pair it with
        ``get_xcc_id`` for a fuller picture. Verified on gfx950, where a
        256-workgroup launch reports CU_ID 0-8 alongside XCC_ID 0-7.

        Other architectures report 0 rather than failing to compile: tracing is
        diagnostic, and losing CU attribution is preferable to breaking every
        traced kernel.

        Returns:
            int32: CU ID within the shader engine, or 0 where unsupported
        """
        if is_hip_cdna3() or is_hip_cdna4():
            return tl.inline_asm_elementwise(
                asm="s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)",
                constraints=("=s"),
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        else:
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
