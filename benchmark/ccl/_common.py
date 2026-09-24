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
DTYPES = [torch.float16, torch.bfloat16]


def torch_tensor(ctx, shape, dtype):
    """A plain device tensor for the RCCL baseline.

    Deliberately not Iris heap memory: the symmetric heap is allocated with the
    flags RMA requires, so timing RCCL against it would measure Iris's
    allocation choice rather than RCCL. Someone choosing between the two would
    call RCCL on ordinary tensors.
    """
    return torch.zeros(shape, dtype=dtype, device=f"cuda:{ctx.get_rank()}")
