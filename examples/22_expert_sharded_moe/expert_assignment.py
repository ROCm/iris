# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Expert-to-rank assignment for expert-parallel MoE.

Ported from triton_kernels/distributed.py:
  https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/distributed.py
"""

import torch
from dataclasses import dataclass


@dataclass
class ExptAssignment:
    # (n_shards, ceil(n_expts_tot / 32))  -- packed int32 bitmask
    # (expt_bitmask[i, j//32] >> j%32) & 1 == 1 iff expert j is owned by shard i
    expt_bitmask: torch.Tensor
    # (n_shards, n_expts_tot)  -- boolean mask
    expt_boolmask: torch.Tensor
    # (n_shards, n_expts_tot)  -- local expert id or -1
    expt_map: torch.Tensor
    n_expts_per_shard: list[int]


def make_expt_dict_uniform(n_shards: int, n_expts_tot: int) -> dict[int, list[int]]:
    """Contiguous assignment: shard i owns experts [i*E_per_shard, (i+1)*E_per_shard)."""
    assert n_expts_tot % n_shards == 0, "n_expts_tot must be divisible by n_shards"
    e_per_shard = n_expts_tot // n_shards
    return {i: list(range(i * e_per_shard, (i + 1) * e_per_shard)) for i in range(n_shards)}


def make_expt_assignment(
    n_shards: int,
    n_expts_tot: int,
    expt_dict: dict[int, list[int]],
    device,
) -> ExptAssignment:
    """Build bitmask, boolmask, and local-id map from an expert ownership dict."""
    words = (n_expts_tot + 31) // 32
    expt_bitmask = torch.zeros((n_shards, words), dtype=torch.int32)
    expt_boolmask = torch.zeros((n_shards, n_expts_tot), dtype=torch.bool)
    counts = {e: 0 for e in range(n_expts_tot)}

    for shard, experts in expt_dict.items():
        if not (0 <= shard < n_shards):
            raise ValueError(f"shard {shard} out of range [0, {n_shards})")
        if len(experts) == 0:
            raise ValueError(f"shard {shard} has no experts")
        for e in experts:
            counts[e] += 1
            if not (0 <= e < n_expts_tot):
                raise ValueError(f"expert id {e} out of range [0, {n_expts_tot})")
            word = e >> 5
            bit = e & 31
            expt_bitmask[shard, word] |= 1 << bit
            expt_boolmask[shard, e] = True

    if not all(c == 1 for c in counts.values()):
        raise ValueError("each expert must be owned by exactly one shard")

    expt_bitmask = expt_bitmask.to(device)
    expt_boolmask = expt_boolmask.to(device)

    expt_map = torch.full((n_shards, n_expts_tot), -1, dtype=torch.int32)
    for shard, experts in expt_dict.items():
        for local_id, global_id in enumerate(sorted(experts)):
            expt_map[shard, global_id] = local_id
    expt_map = expt_map.to(device)

    n_expts_per_shard = [len(expt_dict[s]) for s in range(n_shards)]
    return ExptAssignment(expt_bitmask, expt_boolmask, expt_map, n_expts_per_shard)
