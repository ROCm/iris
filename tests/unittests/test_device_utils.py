# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Device intrinsics in iris.mem.utils, on one GPU.

import pytest
import torch
import triton
import triton.language as tl

from iris.mem.utils import read_realtime


@triton.jit
def _elapsed_kernel(out, ticks):
    """Spin until ``ticks`` have passed, then store the start and end timestamps."""
    t0 = read_realtime()
    while read_realtime() - t0 < ticks:
        pass
    tl.store(out, t0)
    tl.store(out + 1, read_realtime())


def test_read_realtime_waits_for_the_clock():
    """
    s_memrealtime completes on lgkmcnt, which the compiler does not track inside inline asm: each
    read must wait on it before the register is used, or a loop bounded by the clock can exit early.
    """
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.startswith("gfx9"):
        pytest.skip("s_memrealtime is the gfx9 clock read")
    out = torch.zeros(2, dtype=torch.int64, device="cuda")
    ticks = 100_000  # 1 ms of the 100 MHz clock
    compiled = _elapsed_kernel[(1,)](out, ticks, num_warps=1)
    lines = [line.strip() for line in compiled.asm["amdgcn"].splitlines()]
    reads = [i for i, line in enumerate(lines) if line.startswith("s_memrealtime")]
    assert reads
    for i in reads:
        assert lines[i + 1] == "s_waitcnt lgkmcnt(0)", lines[i : i + 2]
    t0, t1 = out.tolist()
    assert t1 - t0 >= ticks
