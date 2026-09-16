# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Test the grid-wide barrier used inside the persistent megakernel.

The barrier is a monotonic global arrival counter: at barrier b every program
adds one and then spins until the counter reaches (b + 1) * num_programs. The
targets only grow, so no reset is needed between barriers. The test runs a
multi-phase pipeline where each phase must observe every program's writes from
the previous phase, which holds only if the barrier both synchronizes the grid
and makes those writes visible.

TWO deliberate choices, both of which cost measurement to establish:

1. TWO counters, not one. One monotonic counter serving both the write->read and
   read->write edges is lappable: a fast program's own later arrival can satisfy a
   target meant to require all P distinct programs, so the barrier releases early.
   With a single counter this test fails roughly 3 runs in 10, and the failure is a
   race rather than a wrong answer, so a single green run proves little -- hence the
   repeat loop below.

2. The barrier is INLINED rather than imported from common.barrier, because neither
   shipped variant is sound for this access pattern. Every program re-reads all N
   slots every phase, which is exactly the cross-phase reuse those barriers document
   as out of scope. Measured, same test, only the barrier swapped:

       no invalidate       (_barrier_noinv)   0/10  FAIL
       buffer_inv sc0      (_barrier)         0/10  FAIL
       buffer_inv sc1      (sem="acquire")   10/10  pass
       buffer_inv sc0 sc1                    10/10  pass

   The spin below uses sem="acquire", which lowers to buffer_inv sc1 on gfx950 -- an
   invalidate that reaches L2. Importing either shipped barrier here would fail, and
   that is a property of this test's all-to-all sharing, not a defect in them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _persistent_pipeline(scratch_ptr, barA_ptr, barB_ptr, out_ptr, P: tl.constexpr, N: tl.constexpr, NPHASE: tl.constexpr):
    pid = tl.program_id(0)
    # Phase loop: each phase, program pid writes scratch[pid]=phase*1000+pid,
    # barrier, then sums ALL scratch (must equal sum over programs for this phase).
    for ph in range(NPHASE):
        tl.store(scratch_ptr + pid, ph * 1000 + pid)
        # ---- grid-wide barrier #ph ----
        tl.debug_barrier()
        tl.atomic_add(barA_ptr, 1, sem="release")
        target = (ph + 1) * P
        done = 0
        while done == 0:
            cur = tl.atomic_add(barA_ptr, 0, sem="acquire")
            if cur >= target:
                done = 1
        # ---- after barrier: read everyone's scratch ----
        acc = tl.zeros((1,), dtype=tl.int32)
        for j in range(0, N):
            acc += tl.load(scratch_ptr + j)
        if pid == 0:
            tl.store(out_ptr + ph, tl.sum(acc))
        # Barrier B: every program has finished READING before any program writes
        # the next phase. Without it a fast program overwrites its slot while a slow
        # one is still summing -- the ~3-in-10 failure this test used to have.
        tl.debug_barrier()
        tl.atomic_add(barB_ptr, 1, sem="release")
        doneB = 0
        while doneB == 0:
            curB = tl.atomic_add(barB_ptr, 0, sem="acquire")
            if curB >= (ph + 1) * P:
                doneB = 1


def main(reps: int = 10):
    dev = "cuda"
    P = 64
    N = P
    NPHASE = 4
    # expected sum at phase ph = sum_j (ph*1000 + j) = P*ph*1000 + P*(P-1)/2
    exp = [P * ph * 1000 + P * (P - 1) // 2 for ph in range(NPHASE)]
    # Repeat: the defect this test exists to catch is a race, so one green run is
    # not evidence. The single-counter version passed ~7 times in 10.
    bad = 0
    for r in range(reps):
        scratch = torch.zeros(N, dtype=torch.int32, device=dev)
        barA = torch.zeros(1, dtype=torch.int32, device=dev)
        barB = torch.zeros(1, dtype=torch.int32, device=dev)
        out = torch.zeros(NPHASE, dtype=torch.int32, device=dev)
        _persistent_pipeline[(P,)](scratch, barA, barB, out, P=P, N=N, NPHASE=NPHASE, num_warps=1)
        torch.cuda.synchronize()
        got = out.tolist()
        if got != exp:
            bad += 1
            print(f"run {r}: FAIL expected {exp} got {got}")
    print("expected", exp)
    print(f"{reps - bad}/{reps} runs matched")
    print("PASS" if bad == 0 else "FAIL")


if __name__ == "__main__":
    main()
