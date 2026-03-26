#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused all-gather GEMM (pull variant)."""

import importlib.util
from pathlib import Path

import torch
import iris.bench as bench

# Load the AG-GEMM pull kernel from the examples directory.
_file = (Path(__file__).parent / "../../examples/14_all_gather_gemm/all_gather_gemm_pull.py").resolve()
_spec = importlib.util.spec_from_file_location("all_gather_gemm_pull", _file)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
persistent_ag_gemm = _mod.persistent_ag_gemm


@bench.register
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_gemm_pull(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if K % world_size != 0:
        state.skip(f"K={K} not divisible by world_size={world_size}")
        return
    K_local = K // world_size
    device = torch.device(f"cuda:{rank}")

    # Allocate tensors
    A_local = ctx.empty((M, K_local), dtype=dtype)
    A_local.fill_(1.0)
    B = torch.randn((K, N), device=device, dtype=dtype)
    C = torch.empty((M, N), device=device, dtype=dtype)

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    BLK_M, BLK_N, BLK_K, gsize_m = 256, 64, 64, 6
    heap_bases = ctx.get_heap_bases()

    state.set_flops(2 * M * N * K)

    state.exec(
        lambda: persistent_ag_gemm[(num_sms,)](
            A_local,
            B,
            C,
            M,
            N,
            K,
            A_local.stride(0),
            A_local.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            BLK_M,
            BLK_N,
            BLK_K,
            gsize_m,
            num_sms,
            1,  # NUM_XCDs
            (K % BLK_K == 0),
            heap_bases,
            rank,
            world_size,
        )
    )


if __name__ == "__main__":
    bench.main()
