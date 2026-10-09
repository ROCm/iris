# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Shared pieces for the CCL benchmarks.

Sweep axes and the RCCL-baseline helper live here rather than being repeated in
every bench_*.py. The rationale for the baseline choice is in
benchmark/ccl/README.md.
"""

import torch

# Shapes follow the triton-shmem harness rather than a square sweep. Collectives
# in these workloads move (tokens, hidden) tensors: the token count varies over
# orders of magnitude while the hidden dimension stays in a narrow band, so M and
# N are swept independently and N is bounded. A square sweep spends most of its
# time on shapes nobody runs.
NUM_RANKS = [2, 4, 8]
M_VALUES = [1024, 4096, 16384, 65536]
N_VALUES = [1024, 2048, 4096, 8192]


def _fp8_dtype():
    """The fp8 type to sweep, or None if this torch build has none.

    gfx950 (MI350X/MI355X) carries the OCP types; gfx942 (MI300X) carries the
    ``fnuz`` variants. Preferred in that order rather than hardcoded, so the
    same sweep runs on either. Which encoding is picked does not affect the
    measurement: these collectives move bytes and, for all_reduce, sum them --
    neither depends on how the exponent and mantissa are laid out.
    """
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            return dtype
    return None


FP8_DTYPE = _fp8_dtype()

# fp8 is swept for Iris only. RCCL has no fp8 support to baseline against, so
# the rccl arm of each benchmark skips it rather than reporting a bogus
# comparison. See "What is not covered" in README.md.
FP8_DTYPES = [FP8_DTYPE] if FP8_DTYPE is not None else []
DTYPES = [torch.float16, torch.bfloat16] + FP8_DTYPES


def is_fp8(dtype):
    """True for the fp8 types this sweep knows about."""
    return dtype in FP8_DTYPES


def torch_tensor(ctx, shape, dtype):
    """A plain device tensor for the RCCL baseline.

    Deliberately not Iris heap memory: the symmetric heap is allocated with the
    flags RMA requires, so timing RCCL against it would measure Iris's
    allocation choice rather than RCCL. Someone choosing between the two would
    call RCCL on ordinary tensors.
    """
    return torch.zeros(shape, dtype=dtype, device=f"cuda:{ctx.get_rank()}")
