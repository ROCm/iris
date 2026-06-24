# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Grid-wide barrier device helper for the persistent megakernel."""

import triton
import triton.language as tl


@triton.jit
def _barrier(bar_ptr, target):
    # Arrive: release so this program's phase writes are flushed to the shared L2
    # before the counter increment becomes visible to peers.
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    # Spin on a relaxed read: polling with acquire emits a full L1 invalidate
    # (buffer_inv sc1) every iteration, which dominates the barrier cost. A relaxed
    # poll just reads the counter cheaply.
    done = 0
    while done == 0:
        cur = tl.atomic_add(bar_ptr, 0, sem="relaxed", scope="gpu")
        if cur >= target:
            done = 1
    # One acquire after the count is reached invalidates L1 a single time, so the
    # next phase reads every peer's writes fresh from L2.
    _ = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
