# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""De-risk: a grid-wide barrier inside ONE persistent Triton kernel on ROCm.

Pattern: monotonic global arrival counter. At barrier #b, every program does
atomic_add(counter, 1) then spins until counter >= (b+1)*P. Targets are
monotonic so no reset race. We test a 3-phase pipeline where phase n must see
all of phase n-1's writes (cross-program), which only holds if the barrier
actually synchronizes + makes writes visible.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _persistent_pipeline(scratch_ptr, bar_ptr, out_ptr, P: tl.constexpr, N: tl.constexpr, NPHASE: tl.constexpr):
    pid = tl.program_id(0)
    # Phase loop: each phase, program pid writes scratch[pid]=phase*1000+pid,
    # barrier, then sums ALL scratch (must equal sum over programs for this phase).
    for ph in range(NPHASE):
        tl.store(scratch_ptr + pid, ph * 1000 + pid)
        # ---- grid-wide barrier #ph ----
        tl.debug_barrier()
        tl.atomic_add(bar_ptr, 1, sem="release")
        target = (ph + 1) * P
        done = 0
        while done == 0:
            cur = tl.atomic_add(bar_ptr, 0, sem="acquire")
            if cur >= target:
                done = 1
        # ---- after barrier: read everyone's scratch ----
        acc = tl.zeros((1,), dtype=tl.int32)
        for j in range(0, N):
            acc += tl.load(scratch_ptr + j)
        if pid == 0:
            tl.store(out_ptr + ph, tl.sum(acc))


def main():
    dev = "cuda"
    P = 64
    N = P
    NPHASE = 4
    scratch = torch.zeros(N, dtype=torch.int32, device=dev)
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.zeros(NPHASE, dtype=torch.int32, device=dev)
    _persistent_pipeline[(P,)](scratch, bar, out, P=P, N=N, NPHASE=NPHASE, num_warps=1)
    torch.cuda.synchronize()
    # expected sum at phase ph = sum_j (ph*1000 + j) = P*ph*1000 + P*(P-1)/2
    exp = [P * ph * 1000 + P * (P - 1) // 2 for ph in range(NPHASE)]
    got = out.tolist()
    print("expected", exp)
    print("got     ", got)
    print("PASS" if got == exp else "FAIL")


if __name__ == "__main__":
    main()
