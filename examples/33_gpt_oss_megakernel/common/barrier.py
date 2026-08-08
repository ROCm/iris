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


@triton.jit
def _barrier_noinv(bar_ptr, target):
    """Grid-wide barrier WITHOUT the trailing cache invalidate. Use this on a single-GPU
    grid where programs share a per-XCD L2.

    `buffer_inv` *discards* dirty lines rather than writing them back, and on MI355X the L2
    is per-XCD and shared with ~31 other workgroups. By the time one program leaves the spin,
    peers have already passed the barrier and begun writing the next phase -- a cache-wide
    invalidate throws those writes away. Measured on GPT-OSS-120B decode, 300 fresh-model
    reps per arm, one variable, same node and prompt:

        invalidate            corrupt runs      distinct wrong outputs
        (none)                     0/300                 0
        buffer_inv sc0            15/300                 1
        buffer_inv sc1           293/300                21
        buffer_inv sc0 sc1       300/300                 6

    Monotonic in invalidation strength; no-invalidate vs sc0 is Fisher one-sided p = 2.6e-05,
    and it is perf-neutral. The release side is unchanged, so writes are still published:
    `buffer_wbl2 sc1` + `s_waitcnt vmcnt(0)` still precede the arrival.

    NOT a drop-in replacement for `_barrier` everywhere. Correctness here relies on there
    being no cross-phase reuse of shared addresses -- true for BS=1 decode, where weights
    stream read-once so nothing stale is resident. A pattern that re-reads shared addresses
    it already has cached needs visibility from somewhere; an all-to-all test fails 0/10
    with this barrier and passes with an invalidating one. `multi_gpu/` still uses
    `_barrier` and is untested against this change.
    """
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    done = 0
    while done == 0:
        # DO NOT "optimise" this RMW into a plain or cache-bypassing load. Polling with
        # atomic_add(ptr, 0) looks wasteful -- it is an unbounded read-modify-write on one
        # line -- but it is also forcing a coherence point that the PAYLOAD depends on,
        # which is what makes a barrier with no invalidate sound at all. Swapping it for
        # tl.load(bar_ptr, cache_modifier=".cv") keeps the counting semantics exactly
        # right (the counter is monotonic and the test is >=, so a stale read can only
        # spin longer, never exit early) and still corrupts output: 25/25 runs wrong with
        # 25 distinct trajectories, versus 0/300 with this RMW. It is ~1.2x faster,
        # because the speed came from the coherence it removed.
        cur = tl.atomic_add(bar_ptr, 0, sem="relaxed", scope="gpu")
        if cur >= target:
            done = 1

