#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for fused all-gather GEMM (push variant)."""

import importlib.util
from pathlib import Path

import torch
import iris.bench as bench

# Load the AG-GEMM push kernels from the examples directory.
_file = (Path(__file__).parent / "../../examples/14_all_gather_gemm/all_gather_gemm_push.py").resolve()
_spec = importlib.util.spec_from_file_location("all_gather_gemm_push", _file)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
gemm_push_kernel = _mod.gemm_push_kernel
push_shards_kernel = _mod.push_shards_kernel


@bench.register
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def all_gather_gemm_push(state, ctx):
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    if K % world_size != 0:
        state.skip(f"K={K} not divisible by world_size={world_size}")
        return
    K_local = K // world_size
    device = torch.device(f"cuda:{rank}")

    BLK_M, BLK_N, BLK_K, gsize_m = 256, 64, 64, 6
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    # Allocate tensors
    A_local = ctx.empty((M, K_local), dtype=dtype)
    A_local.fill_(1.0)
    A_inbox = ctx.empty((world_size, M, K_local), dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)
    C = torch.empty((M, N), device=device, dtype=dtype)

    num_m_tiles = (M + BLK_M - 1) // BLK_M
    num_k_tiles = (K_local + BLK_K - 1) // BLK_K
    signal_flags = ctx.zeros((world_size, world_size, num_m_tiles, num_k_tiles), dtype=torch.int32)
    heap_bases = ctx.get_heap_bases()

    state.set_flops(2 * M * N * K)

    def _exec():
        push_grid = (num_m_tiles, num_k_tiles)
        push_shards_kernel[push_grid](
            A_local,
            A_inbox,
            signal_flags,
            M,
            K_local,
            A_local.stride(0),
            A_local.stride(1),
            A_inbox.stride(0),
            A_inbox.stride(1),
            A_inbox.stride(2),
            signal_flags.stride(0),
            signal_flags.stride(1),
            signal_flags.stride(2),
            signal_flags.stride(3),
            BLK_M,
            BLK_K,
            rank,
            world_size,
            heap_bases,
        )
        gemm_push_kernel[(num_sms,)](
            A_inbox,
            B,
            C,
            M,
            N,
            K,
            signal_flags,
            A_inbox.stride(0),
            A_inbox.stride(1),
            A_inbox.stride(2),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            signal_flags.stride(0),
            signal_flags.stride(1),
            signal_flags.stride(2),
            signal_flags.stride(3),
            BLK_M,
            BLK_N,
            BLK_K,
            gsize_m,
            num_sms,
            1,  # NUM_XCDs
            (K_local % BLK_K == 0),
            rank,
            world_size,
        )

    state.exec(_exec, preamble_fn=lambda: signal_flags.zero_())


if __name__ == "__main__":
    bench.main()
