#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: fused independent compute + communication in ONE kernel.

A single persistent kernel runs an independent GEMM (C = A @ B) alongside an
all-gather on unrelated data. The two workloads are NOT producer/consumer --
they share nothing but GPU resources.

The kernel launches ``num_wgs`` persistent workgroups split across two
work-stealing queues:

    WG id <  gemm_wgs  ->  starts in the GEMM queue
    WG id >= gemm_wgs  ->  starts in the comm queue

Each WG drains its home queue by atomically claiming tile indices, then steals
from the other queue once its own is empty. This self-balances: whichever
workload finishes first donates its workgroups to the other, so no CU sits idle
while the other queue still has work.

Contrast with example 33 (``mode="concurrent"``), which runs the same two
workloads as two separate kernels on two streams -- there, the CU split is
fixed at launch and cannot rebalance.

Run with:
    torchrun --nproc_per_node=8 --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
import iris.concurrent.gemm as cg


def parse_args():
    parser = argparse.ArgumentParser(description="Fused independent compute + communication")
    parser.add_argument("-m", type=int, default=2048, help="GEMM M dimension")
    parser.add_argument("-n", type=int, default=1024, help="GEMM N dimension")
    parser.add_argument("-k", type=int, default=2048, help="GEMM K dimension")
    parser.add_argument("--comm_m", type=int, default=256, help="Per-rank all-gather rows")
    parser.add_argument("--comm_n", type=int, default=512, help="All-gather columns")
    parser.add_argument("--gemm_wgs", type=int, default=None, help="WGs starting in the GEMM queue")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size in bytes")
    parser.add_argument("--validate", action="store_true", help="Validate both halves")
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    shmem = iris.iris(args.heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    cu_count = shmem.get_cu_count()
    dtype = torch.float16

    # Split the CUs: 3/4 start on GEMM, 1/4 on comm. Work stealing rebalances
    # this at runtime, so the split only decides who starts where.
    gemm_wgs = args.gemm_wgs if args.gemm_wgs is not None else (cu_count * 3) // 4

    if rank == 0:
        print(f"Fused compute + communication ({world_size} ranks, {cu_count} CUs)")
        print(f"  GEMM: ({args.m}, {args.k}) @ ({args.k}, {args.n})")
        print(f"  All-gather: ({args.comm_m}, {args.comm_n}) -> ({world_size * args.comm_m}, {args.comm_n})")
        print(f"  Queue split: {gemm_wgs} WGs start on GEMM, {cu_count - gemm_wgs} start on comm")

    # GEMM operands -- independent of the collective.
    A = shmem.randn(args.m, args.k, device="cuda", dtype=dtype)
    B = shmem.randn(args.n, args.k, device="cuda", dtype=dtype).T
    C = shmem.zeros((args.m, args.n), device="cuda", dtype=dtype)

    # All-gather operands -- unrelated to the GEMM. Each rank contributes a
    # block filled with its own rank id so the gathered result is checkable.
    comm_src = shmem.full((args.comm_m, args.comm_n), float(rank + 1), device="cuda", dtype=dtype)
    comm_dst = shmem.zeros((world_size * args.comm_m, args.comm_n), device="cuda", dtype=dtype)

    shmem.barrier()

    # One kernel. Two queues. Both workloads run to completion.
    C, comm_dst = cg.all_gather(
        shmem,
        A,
        B,
        comm_src,
        C=C,
        comm_dst=comm_dst,
        mode="fused",
        num_wgs=cu_count,
        gemm_wgs=gemm_wgs,
    )

    shmem.barrier()

    if args.validate:
        # GEMM half: C == A @ B
        gemm_ref = torch.matmul(A.float(), B.float())
        gemm_err = (C.float() - gemm_ref).abs().max().item()
        gemm_scale = max(gemm_ref.abs().max().item(), 1e-6)
        gemm_rel = gemm_err / gemm_scale
        gemm_ok = gemm_rel < 1e-2

        # Comm half: comm_dst[r] == r + 1 for every rank block
        comm_ok = True
        for r in range(world_size):
            block = comm_dst[r * args.comm_m : (r + 1) * args.comm_m]
            expected = float(r + 1)
            if not torch.allclose(block.float(), torch.full_like(block.float(), expected)):
                comm_ok = False
                if rank == 0:
                    got = block.float().flatten()[0].item()
                    print(f"  comm FAIL at rank block {r}: expected {expected}, got {got}")
                break

        if rank == 0:
            print(f"  GEMM: {'PASS' if gemm_ok else 'FAIL'} (rel err {gemm_rel:.2e})")
            print(f"  All-gather: {'PASS' if comm_ok else 'FAIL'}")
            print(f"  {'PASS' if (gemm_ok and comm_ok) else 'FAIL'}")

        shmem.barrier()
        if not (gemm_ok and comm_ok):
            raise SystemExit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
