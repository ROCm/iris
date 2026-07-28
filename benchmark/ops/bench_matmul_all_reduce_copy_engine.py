#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for matmul + all-reduce copy-engine variants."""

import torch
import iris.bench as bench

from iris.ops.matmul_all_reduce_copy_engine import (
    FusedConfig,
    matmul_all_reduce_copy_engine as _copy_engine,
    matmul_all_reduce_copy_engine_preamble,
    matmul_all_reduce_copy_engine_prepost_transfers,
)
from tritonblas.matmul import _make_matmul_selector


_MAX_ONE_SHOT_INBOX_ELEMENTS = 2**31


def _make_selector(M: int, N: int, K: int, dtype: torch.dtype, device: torch.device):
    return _make_matmul_selector(
        M,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )


def _register_copy_engine(state, ctx, *, variant: str, prepost: bool = False) -> None:
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    if variant == "two_shot_gpu_init" and prepost:
        state.skip("two_shot_gpu_init posts SDMA transfers from the GEMM kernel; host prepost is not used")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector = _make_selector(M, N, K, dtype, device)

    if variant == "one_shot" and world_size * M * N > _MAX_ONE_SHOT_INBOX_ELEMENTS:
        state.skip(
            "one_shot requires world_size * M * N <= 2**31 elements "
            f"(got {world_size * M * N})"
        )

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)


    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_copy_engine_preamble(
        ctx,
        C,
        A,
        B,
        config=config,
        selector=selector,
    )

    prepost_transfers = prepost

    def _run_copy_engine():
        _copy_engine(
            ctx,
            C,
            A,
            B,
            async_op=True,
            config=config,
            workspace=workspace,
            copy_engine_transfers_preposted=prepost_transfers,
        )

    def _preamble():
        if prepost_transfers:
            matmul_all_reduce_copy_engine_prepost_transfers(
                ctx,
                A,
                B,
                workspace,
            )

    def _run():
        _run_copy_engine()

    state.set_flops(2 * M * N * K)
    state.exec(_run, preamble_fn=_preamble)

    # HBM bytes: matmul A + B + C
    hbm_bytes = (M * K + K * N + M * N) * C.element_size()
    state.set_bytes(hbm_bytes)

    # XGMI bytes: depends on algorithm variant
    if variant == "one_shot":
        # One-shot: each rank broadcasts its full M×N output to (world_size-1) peers
        xgmi_bytes = (world_size - 1) * M * N * C.element_size()
    else:  # two_shot
        # Two-shot ring all-reduce: 2 × (world_size-1)/world_size × full_output
        xgmi_bytes = 2 * (world_size - 1) * M * N * C.element_size() // world_size
    state.add_counter("xgmi_bytes", xgmi_bytes)

    launch = workspace.launch_params
    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("total_tiles", launch["total_tiles"])
    state.add_counter("num_stages", launch["num_stages"] or 0)
    state.add_counter("aux_rows", workspace.aux_buffer.shape[0] if workspace.aux_buffer is not None else 0)
    state.add_counter("reduce_rows", workspace.a_inbox.shape[0] if workspace.a_inbox is not None else 0)
    state.add_counter("transfer_waves", getattr(workspace, "num_transfer_waves", 0))
    transfer_rects = getattr(workspace, "num_transfers", 0)
    if variant == "one_shot":
        transfer_rects *= world_size - 1
    state.add_counter("transfer_rects", transfer_rects)
    state.add_counter("reduce_block_m", launch["reduce_block_size_m"])
    state.add_counter("reduce_block_n", launch["reduce_block_size_n"])
    state.add_counter("reduce_num_sms", launch.get("reduce_num_sms", 0))
    state.add_counter("variant_one_shot", 1.0 if variant == "one_shot" else 0.0)
    state.add_counter("variant_two_shot", 1.0 if variant == "two_shot" else 0.0)
    state.add_counter("variant_two_shot_gpu_init", 1.0 if variant == "two_shot_gpu_init" else 0.0)
    state.add_counter("gpu_init_row_major", 1.0 if variant == "two_shot_gpu_init" else 0.0)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["one_shot", "two_shot", "two_shot_gpu_init"])
@bench.axis("prepost", [False, True])
def matmul_all_reduce_copy_engine(state, ctx):
    """Kernel-based matmul_all_reduce with one_shot/two_shot variants and prepost option."""
    variant = state["variant"]
    prepost = state["prepost"]
    _register_copy_engine(state, ctx, variant=variant, prepost=prepost)


if __name__ == "__main__":
    bench.main()
