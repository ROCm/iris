#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example: Cross-GPU Atomic Operations
Demonstrates: iris.atomic_add(), iris.atomic_cas(), iris.atomic_xchg()

Every rank atomically adds 1 to a counter that lives on rank 0's symmetric
heap.  After a barrier the counter should equal world_size.

The kernel uses sem="acq_rel" and scope="sys" to ensure global visibility
across GPUs connected via XGMI / PCIe.  Other useful combinations:

    sem="relaxed"  / scope="gpu"  -- fastest, single-GPU only
    sem="acquire"  / scope="sys"  -- read-acquire for flag polling
    sem="release"  / scope="sys"  -- write-release for flag signalling

Run with:
    torchrun --nproc_per_node=N --standalone examples/05_mem_atomics/example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def atomic_add_kernel(
    counter_ptr,
    cur_rank: tl.constexpr,
    target_rank: tl.constexpr,
    heap_bases: tl.tensor,
):
    """Each program instance atomically adds 1 to counter on target_rank."""
    val = tl.full([], 1, dtype=tl.int32)
    iris.atomic_add(
        counter_ptr,
        val,
        cur_rank,
        target_rank,
        heap_bases,
        sem="acq_rel",
        scope="sys",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="iris.atomic_add example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--validate", action="store_true", help="Run correctness check")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Symmetric heap size in bytes")
    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    heap_bases = ctx.get_heap_bases()

    # Allocate a single int32 counter on every rank's heap (only rank 0's is used).
    counter = ctx.zeros(1, dtype=torch.int32)

    ctx.barrier()

    # Every rank atomically adds 1 to rank 0's counter.
    grid = (1,)
    atomic_add_kernel[grid](counter, rank, 0, heap_bases)

    ctx.barrier()

    # --- Validation ---------------------------------------------------------
    if args["validate"]:
        if rank == 0:
            expected = torch.tensor([world_size], dtype=torch.int32, device="cuda")
            torch.testing.assert_close(counter, expected, rtol=0, atol=0)
            print(f"[rank {rank}] atomic_add validation PASSED  (counter={counter.item()})")
    else:
        if rank == 0:
            print(f"[rank {rank}] counter = {counter.item()} (run with --validate to check)")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
