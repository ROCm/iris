# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 (E2M1) magnitude lookup for the dequantizing FP4 GEMV path."""

import triton
import triton.language as tl


@triton.jit
def _fp4_lut(mag_idx):
    return tl.where(
        mag_idx == 0,
        0.0,
        tl.where(
            mag_idx == 1,
            0.5,
            tl.where(
                mag_idx == 2,
                1.0,
                tl.where(
                    mag_idx == 3,
                    1.5,
                    tl.where(mag_idx == 4, 2.0, tl.where(mag_idx == 5, 3.0, tl.where(mag_idx == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
