# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Device intrinsics in iris.mem.utils, on one GPU.

import torch
import triton
import triton.language as tl

from iris.mem.utils import read_realtime


@triton.jit
def _spin_kernel(out, ticks):
    """Spin until ``ticks`` have passed by read_realtime(), then store the time it measured."""
    t0 = read_realtime()
    while read_realtime() - t0 < ticks:
        pass
    tl.store(out + tl.program_id(0), read_realtime() - t0)


def test_read_realtime_waits_for_the_clock():
    """
    A loop bounded by read_realtime() runs for its whole budget. s_memrealtime completes on
    lgkmcnt, which the compiler does not track inside inline asm: a read that waits on anything else
    can be used before the clock value lands, and a small share of such loops exit early.
    """
    ticks = 100_000  # 1 ms of the 100 MHz clock
    out = torch.zeros(1024, dtype=torch.int64, device="cuda")
    for _ in range(20):
        _spin_kernel[(out.numel(),)](out, ticks, num_warps=1)
        early = (out < ticks).sum().item()
        assert early == 0, f"{early} of {out.numel()} loops ended early, after {out.min().item()} ticks"
