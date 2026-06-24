# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Grid-wide barrier device helper for the persistent megakernel."""

import triton
import triton.language as tl


@triton.jit
def _l1_invalidate():
    # gfx950: invalidate only the per-CU vector L1 (sc0), NOT the system level (sc1).
    # For a single-GPU grid barrier the device-shared L2 is the point of coherence, so
    # an L1-only invalidate is sufficient to read peers' L2-resident writes fresh, and
    # is cheaper than Triton's acquire (which emits buffer_inv sc1 + a round-trip).
    tl.inline_asm_elementwise(
        "buffer_inv sc0\n\ts_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _barrier(bar_ptr, target):
    # Arrive: release so this program's phase writes are flushed to the shared L2
    # before the counter increment becomes visible to peers.
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    # Spin on a relaxed read (cheap; the release above already ordered the writes).
    done = 0
    while done == 0:
        cur = tl.atomic_add(bar_ptr, 0, sem="relaxed", scope="gpu")
        if cur >= target:
            done = 1
    # L1-only invalidate (asm) instead of a full sc1 acquire, so the next phase reads
    # every peer's writes fresh from the shared L2.
    _l1_invalidate()
