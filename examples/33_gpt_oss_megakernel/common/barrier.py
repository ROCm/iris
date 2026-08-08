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
        buffer_inv sc0            15/300                 1    <- SEE BELOW: not reproduced
        buffer_inv sc1           293/300                21
        buffer_inv sc0 sc1       300/300                 6

    THE sc0 ROW SORTS BY MACHINE, NOT BY RATE. Every run of the identical file, by host:

        thor-2          co-tenant then solo    15/300   fails
        thor-2          synthetic load ON        4/300   fails (from rep ~50, load still on)
        thor-3          solo, verified           0/300   clean   <- same node family
        pollara-1       synthetic load ON        0/300   clean
        dlc-pollara-4   solo, verified           0/300   clean

    Load was varied on both sides and sorts nothing: the failing host fails both with a
    co-tenant and under synthetic load, and the passing hosts pass both loaded and solo.
    The host sorts every run. It is ONE MACHINE, not a node family -- thor-3 is the same
    family as thor-2 and is clean at full length.

    How strong that is: 0/300 on thor-3 excludes thor-2's measured rate (P = 0.008 at 1.6%)
    but not the bottom of its interval (P = 0.27 at 0.44%). So thor-3 does not behave like
    thor-2 at any rate actually observed there, and a rate below ~0.4% is not excluded. Note
    also that thor-2's own two runs disagree by 4x (15/300 and 4/300), so the host sorts the
    outcome without producing a stable rate underneath it.

    THE COMPARISON THAT DOES HOLD IS ON ONE MACHINE. Restricting to thor-2, the only host
    where `buffer_inv sc0` has ever failed, and taking every full-length run of either
    barrier there:

        buffer_inv sc0    19/600 = 3.2%   (two runs, two different load conditions)
        _barrier_noinv     0/600 = 0.0%   (two runs, paired in one container)
                                          Fisher one-sided p = 1.7e-06

    Same machine, both barriers, full n, no cross-venue or cross-run division on either
    side. That is a stronger justification for this change than the 5% ever was: it is
    measured on the host that actually exhibits the defect rather than averaged over hosts
    that do not. (The noinv runs predate knowing thor-2 was the interesting machine, so
    they were not designed as this comparison -- they simply are one.)

    Practical consequence: there is no reproducible failure rate for the shipping dose and
    none should be quoted, but "does not reproduce" is host-conditioned and not a clean bill.
    The 15 failures were real -- 15 byte-identical prompt-echo trajectories is not noise --
    and remain unexplained. "Host" here is a label for whatever differs, not a mechanism;
    Slurm reports the nodes as identical in version, GRES, CPU and memory.

    Removing the invalidate is still the right call, and the argument does not need that rate.
    It is weakly dominant: removal has zero measured cost (perf-neutral in both dtypes),
    keeping it has no demonstrated benefit of any kind, the operation is destructive by
    mechanism and catastrophic at stronger scopes, and no run of any length in any venue has
    ever failed after removing it. A decision that survives its headline number being wrong is
    sturdier than one that needed it.

    The sc1 rows are what the mechanism rests on; they reproduce in every venue tested at full
    n and no venue argument reaches numbers that size. Monotonic in invalidation strength,
    and it is perf-neutral. The release side is unchanged, so writes are still published:
    `buffer_wbl2 sc1` + `s_waitcnt vmcnt(0)` still precede the arrival.

    NOT a drop-in replacement for `_barrier`. THE CONDITION IS THE ACCESS PATTERN, NOT THE
    BATCH SIZE: this barrier is sound only where programs do not re-read, across a barrier,
    shared addresses they already have cached. Measured on both sides of that condition:

        reuse absent  (this kernel)              0/300 correct
        reuse present (64-program all-to-all,     0/10  FAIL -- and passes with an
                       same slots re-read each          invalidating barrier
                       phase)

    This kernel satisfies it because decode streams weights read-once, so nothing stale is
    resident to go unnoticed. Do not restate that as "safe at batch size 1" -- this kernel
    has no batch dimension and cannot leave that regime, so batch size is not the axis and
    points a reader at the wrong test. The audience for this warning is anyone reusing the
    barrier in a *different* kernel: check whether your programs re-read shared addresses
    across a barrier, and if they do, use `_barrier`.

    A RESIDUAL DEFECT IS NOT EXCLUDED, and the one observation sits entirely in the venue
    we could not characterise. Split by machine rather than pooled, because that is where
    the structure is:

        near-idle node, informative region     0/300    95% CI [0.00%, 1.22%]
        exclusive node, channel measured absent 0/300   95% CI [0.00%, 1.22%]
        characterised venues combined          0/600    95% CI [0.00%, 0.61%]
        organically busy shared node           1/100    95% CI [0.03%, 5.45%]  <- the event

    Pooled it is 1/700 (0.14%), but the pooled number hides the only interesting fact: this
    barrier has never failed on a machine whose conditions we understand, and the single
    failure came from one we do not. That is not evidence the barrier is safe on a busy
    node -- 100 reps is thin and a synthetic load built to reproduce the condition was
    measured inert (it saturates sibling GPUs, and sibling GPUs do not move this GPU's
    clock). "Organically busy multi-tenant node" is a condition we cannot reproduce on
    demand, so this rate is unknown there rather than merely unmeasured.

    Node contention was the first explanation for that event and it is NOT supported:
    sampling GPU 0 under this workload measured 2388 MHz on an idle node and 2392 MHz with
    seven sibling GPUs saturated (0.2% apart), so the board power cap is not binding and
    neighbours do not move this GPU's clock. They also cannot reach its L2. With no
    established channel, the event is more likely a real low-rate defect than an artifact.

    Do not quote a ratio against the barrier this replaces, and do not quote its 5% either:
    that rate is venue-conditioned. The same `buffer_inv sc0` file, same harness, same n:

        thor-2, co-tenant for reps 1-117 then solo   15/300 = 5.0%
        pollara-1, controlled synthetic load          0/300 = 0.0%   Fisher p = 2.6e-05

    Both are real measurements of the same code. Note also that all 15 failures in the
    first run landed after its co-tenant left, so "5% on a quiet node" was wrong in both
    halves -- the node was not quiet, and 5% is not the barrier's rate.

    What does not depend on any of this: the sc1 variants measured 293/300 and 300/300.
    No venue argument reaches numbers like those, so the decision to remove the invalidate
    stands independently of what the low-rate figures settle at. Treat this barrier as a
    large improvement with an uncharacterised remainder, not as a clean one.

    `multi_gpu/` still uses `_barrier` and is untested against this change.
    """
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    done = 0
    while done == 0:
        # DO NOT "optimise" this RMW into a plain or cache-bypassing load, and note that
        # WE DO NOT KNOW WHY IT IS LOAD-BEARING. Polling with atomic_add(ptr, 0) looks
        # like pure waste -- an unbounded read-modify-write on one line from every
        # workgroup. Replacing it with tl.load(bar_ptr, cache_modifier=".cv") keeps the
        # counting semantics exactly right (the counter is monotonic and the test is >=,
        # so a stale read can only spin longer, never exit early), makes the barrier
        # ~1.9x cheaper and the token ~1.2x faster -- and corrupts the output: 25/25 runs
        # wrong with 25 distinct trajectories, against 0/300 with this RMW.
        #
        # It is NOT that the poll is merely slow. Substituting a delay of 27x the RMW's
        # measured cost (s_sleep 4096, ~109 us against ~4.00 us) on an exclusive node is
        # 25/25 corrupt -- so "correct by being slow" is eliminated, at a dose and in a
        # venue where a failure is interpretable. Nor is it one discrete coherence event:
        # adding a single relaxed atomic back at the exit does not fix it either, and
        # neither does draining with s_waitcnt. What remains is something about the
        # REPEATED RMW's memory-model semantics, and it is not yet explained.
        #
        # Practical rule until someone settles it: any change that makes this loop faster
        # must be gated on a >=300-rep output-correctness run. A timing harness cannot
        # see this failure, and neither can an assert that the counter reached NB*NWG --
        # that only proves every arrival happened, not that anyone waited.
        cur = tl.atomic_add(bar_ptr, 0, sem="relaxed", scope="gpu")
        if cur >= target:
            done = 1

