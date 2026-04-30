#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import sys
from typing import Optional

import torch


def dist_print(message: str, rank: int, is_error: bool = False):
    """Prints a message only from rank 0."""
    if rank == 0:
        if is_error:
            print(f"❌ ERROR: {message}", file=sys.stderr)
        else:
            print(message)


def print_correctness_report(
    rank: int, computed: torch.Tensor, reference: torch.Tensor, error: Optional[Exception] = None
):
    """
    Prints a detailed report from rank 0 and the final status from all ranks.
    """
    if rank == 0:
        print("\n<<<<<<<<<< Correctness Test Report (Impl: FUSED_FULL) >>>>>>>>>>")
        print(f"--- Detailed Validation on Rank {rank} ---")
        header = f"{'Index':<8} | {'Computed':<15} | {'Reference':<15} | {'Abs. Diff':<15}"
        print("--- Comparison of First 16 Values (Head 0) ---")
        print(header)
        print("-" * len(header))

        comp_slice = computed[0, 0, :16].cpu().float()
        ref_slice = reference[0, 0, :16].cpu().float()
        diff_slice = torch.abs(comp_slice - ref_slice)

        for i in range(len(comp_slice)):
            print(f"{i:<8} | {comp_slice[i]:<15.6f} | {ref_slice[i]:<15.6f} | {diff_slice[i]:<15.6f}")
        print("-" * len(header))

    # This final status prints from ALL ranks
    if error:
        print(f"❌ TEST FAILED for Rank {rank}:\n{error}")
    else:
        max_diff = torch.max(torch.abs(computed - reference))
        print(f"✅ TEST PASSED for Rank {rank}. Max absolute difference: {max_diff:.6f}")
