#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: two concurrent independent kernels (iris.concurrent, mode="concurrent").

Runs an independent GEMM (C = A @ B) alongside an all-gather collective as TWO
separate persistent work-stealing kernels on separate streams. The operands are
unrelated -- the GEMM does not produce the collective's input -- so the two
kernels contend only for GPU resources (CUs, memory bandwidth, fabric).

This is the "concurrent" overlap model. Compare with example 34, which fuses
both into a single persistent kernel with two work-stealing queues.

--gemm_cus and --comm_cus set each kernel's grid. Because these are two
independent kernels on two streams, the counts are a real partition of the
device -- and unlike the fused variant they may sum to more than cu_count,
deliberately oversubscribing.

Run with:
    torchrun --nproc_per_node=<num_gpus> --standalone example.py [--validate]
"""

import argparse
import os

import torch
import torch.distributed as dist

import iris
import iris.concurrent.gemm as cg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two concurrent independent kernels: GEMM + all-gather",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gemm_m", type=int, default=2048, help="GEMM M dimension")
    parser.add_argument("--gemm_n", type=int, default=1024, help="GEMM N dimension")
    parser.add_argument("--gemm_k", type=int, default=2048, help="GEMM K dimension")
    parser.add_argument("--comm_m", type=int, default=256, help="All-gather rows, per rank")
    parser.add_argument("--comm_n", type=int, default=512, help="All-gather columns")
    parser.add_argument(
        "--gemm_cus",
        type=int,
        default=None,
        help="CUs for the GEMM kernel (default: cu_count - comm_cus)",
    )
    parser.add_argument(
        "--comm_cus",
        type=int,
        default=None,
        help="CUs for the all-gather kernel (default: cu_count // 8). "
        "These are two separate kernels on two streams, so the counts are a real "
        "partition -- and may oversubscribe (sum > cu_count) on purpose.",
    )
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate both halves against references")
    return vars(parser.parse_args())


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    cu_count = ctx.get_cu_count()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args["datatype"]]
    M, N, K = args["gemm_m"], args["gemm_n"], args["gemm_k"]

    # All-gather source is this rank's block: comm_m is PER RANK, exactly as in
    # example 34. The same flag value means the same tensor in both examples at
    # every world size.
    comm_rows = args["comm_m"]
    comm_cols = args["comm_n"]

    # CU counts. These are two independent persistent kernels on separate streams,
    # so each count is that kernel's grid -- a real allocation, not a starting
    # position. Oversubscribing (sum > cu_count) is permitted and occasionally
    # useful; we only require each side to be non-empty.
    comm_cus = args["comm_cus"] if args["comm_cus"] is not None else max(1, cu_count // 8)
    gemm_cus = args["gemm_cus"] if args["gemm_cus"] is not None else cu_count - comm_cus
    if gemm_cus < 1:
        raise ValueError(f"--gemm_cus must be >= 1, got {gemm_cus}")
    if comm_cus < 1:
        raise ValueError(f"--comm_cus must be >= 1, got {comm_cus}")

    # GEMM operands (independent of the collective)
    torch.manual_seed(42 + rank)
    A = ctx.randn(M, K, device="cuda", dtype=dtype)
    B = ctx.randn(N, K, device="cuda", dtype=dtype).T
    C = ctx.zeros((M, N), device="cuda", dtype=dtype)

    # Collective operand: this rank's block, filled with its rank id so the
    # gathered result is trivially checkable.
    comm_src = ctx.full((comm_rows, comm_cols), float(rank + 1), device="cuda", dtype=dtype)
    comm_dst = ctx.zeros((world_size * comm_rows, comm_cols), device="cuda", dtype=dtype)

    if rank == 0:
        total_cus = gemm_cus + comm_cus
        oversub = " OVERSUBSCRIBED" if total_cus > cu_count else ""
        ctx.info(
            f"concurrent mode: GEMM ({M}x{K} @ {K}x{N}) || all_gather "
            f"({comm_rows}x{comm_cols} -> {world_size * comm_rows}x{comm_cols}), "
            f"gemm_cus={gemm_cus} comm_cus={comm_cus} "
            f"sum={total_cus} cu_count={cu_count}{oversub} "
            f"[hard partition: 2 kernels, 2 streams -- cf. ex34 where the same "
            f"numbers are initial placement and stealing rebalances]"
        )

    ctx.barrier()
    cg.all_gather(
        ctx,
        A,
        B,
        comm_src,
        C=C,
        comm_dst=comm_dst,
        mode="concurrent",
        gemm_wgs=gemm_cus,
        comm_wgs=comm_cus,
    )
    torch.cuda.synchronize()
    ctx.barrier()

    if args["validate"]:
        # GEMM half: C ~= A @ B
        gemm_ref = torch.matmul(A.float(), B.float())
        gemm_err = (C.float() - gemm_ref).abs().max().item() / max(gemm_ref.abs().max().item(), 1e-6)
        assert gemm_err < 0.05, f"Rank {rank}: GEMM rel err {gemm_err:.4f} too high"

        # Comm half: comm_dst ~= torch.distributed.all_gather(comm_src)
        ref_src = torch.full((comm_rows, comm_cols), float(rank + 1), device=comm_src.device, dtype=dtype)
        comm_ref = torch.empty((world_size * comm_rows, comm_cols), device=comm_src.device, dtype=dtype)
        dist.all_gather_into_tensor(comm_ref, ref_src)
        max_diff = (comm_dst.float() - comm_ref.float()).abs().max().item()
        assert torch.allclose(comm_dst, comm_ref, atol=1e-3), f"Rank {rank}: all-gather mismatch, max diff {max_diff}"

        if rank == 0:
            ctx.info(f"Validation passed: GEMM rel err {gemm_err:.2e}, all-gather max diff {max_diff:.2e}")

    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
