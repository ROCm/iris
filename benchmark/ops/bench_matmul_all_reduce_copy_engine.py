#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for matmul + all-reduce copy-engine variants."""

import os
import statistics
import torch
import iris.bench as bench

from iris.ops.matmul_all_reduce_copy_engine import (
    matmul_all_reduce_copy_engine as _copy_engine,
    matmul_all_reduce_copy_engine_preamble,
    matmul_all_reduce_copy_engine_prepost_transfers,
)
from tritonblas.matmul import _make_matmul_selector


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


def _register_copy_engine(state, ctx, *, variant: str) -> None:
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    selector = _make_selector(M, N, K, dtype, device)

    if M % selector.block_m != 0:
        state.skip(f"M={M} must be divisible by block_size_m={selector.block_m}")
    if N % selector.block_n != 0:
        state.skip(f"N={N} must be divisible by block_size_n={selector.block_n}")
    if K % selector.block_k != 0:
        state.skip(f"K={K} must be divisible by block_size_k={selector.block_k}")

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    from iris.ops import FusedConfig

    config = FusedConfig(all_reduce_variant=variant)
    workspace = matmul_all_reduce_copy_engine_preamble(
        ctx,
        C,
        A,
        B,
        config=config,
        selector=selector,
    )

    flag_iteration = [0]
    # TEMPORARY: Disable SDMA transfers to isolate all_gather performance
    prepost_transfers = False  # variant == "two_shot"
    last_host_post_ms = [0.0]

    def _run_copy_engine():
        # if ctx.get_rank() == 0 and flag_iteration[0] < 3:
        #     print(f"DEBUG _run_copy_engine: flag_iteration={flag_iteration[0]}")
        _copy_engine(
            ctx,
            C,
            A,
            B,
            async_op=True,
            config=config,
            workspace=workspace,
            flag_iteration=flag_iteration[0],
            copy_engine_transfers_preposted=prepost_transfers,
        )
        flag_iteration[0] += 1
        # if ctx.get_rank() == 0 and flag_iteration[0] <= 3:
        #     print(f"DEBUG _run_copy_engine done: flag_iteration={flag_iteration[0]}")

    def _preamble():
        pass

    # Benchmark runs: 1 initial + n_warmup + n_repeat iterations
    # Allocate +1 event for iteration 0 so we can use flag_iteration directly as index
    n_warmup = state._n_warmup
    n_repeat = state._n_repeat
    n_total = 1 + n_warmup + n_repeat  # +1 for iteration 0

    # Create event arrays
    # gemm_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_total)]
    # gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_total)]
    # allgather_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_total)]
    # allgather_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_total)]

    # # Store events and iteration info in workspace
    # workspace.bench_gemm_start_events = gemm_start_events
    # workspace.bench_gemm_end_events = gemm_end_events
    # workspace.bench_start_events = allgather_start_events
    # workspace.bench_end_events = allgather_end_events
    # workspace.bench_n_warmup = n_warmup
    # workspace.bench_n_repeat = n_repeat

    def _run():
        _run_copy_engine()

        # After the last iteration, calculate timing from recorded events
        # DISABLED: Event recording adds overhead to e2e time
        # if flag_iteration[0] == n_total:
        #     torch.cuda.synchronize()

        #     # Only use events from the timed iterations (skip iteration 0 and warmup)
        #     timed_start = 1 + n_warmup
        #     timed_end = n_total  # All iterations completed
        #     actual_timed = timed_end - timed_start

        #     if ctx.get_rank() == 0:
        #         print(f"DEBUG: Calculating timings: timed_start={timed_start}, timed_end={timed_end}, actual_timed={actual_timed}")

        #     if actual_timed > 0:
        #         gemm_times = [gemm_start_events[i].elapsed_time(gemm_end_events[i])
        #                       for i in range(timed_start, timed_end)]
        #         allgather_times = [allgather_start_events[i].elapsed_time(allgather_end_events[i])
        #                           for i in range(timed_start, timed_end)]

        #         avg_gemm_ms = statistics.mean(gemm_times)
        #         avg_allgather_ms = statistics.mean(allgather_times)

        #         state.add_counter("gemm_ms_profile", avg_gemm_ms)
        #         state.add_counter("gemm_ms_profile_min", min(gemm_times))
        #         state.add_counter("gemm_ms_profile_max", max(gemm_times))
        #         state.add_counter("reduce_allgather_no_wait_ms_profile", avg_allgather_ms)
        #         state.add_counter("reduce_allgather_no_wait_ms_profile_min", min(allgather_times))
        #         state.add_counter("reduce_allgather_no_wait_ms_profile_max", max(allgather_times))
        #         state.add_counter("reduce_allgather_ms_profile", avg_allgather_ms)
        #         state.add_counter("copy_completion_wait_ms_profile", 0.0)
        #         state.add_counter("profile_samples", actual_timed)
        #     else:
        #         state.add_counter("reduce_allgather_no_wait_ms_profile", 0.0)
        #         state.add_counter("reduce_allgather_ms_profile", 0.0)
        #         state.add_counter("copy_completion_wait_ms_profile", 0.0)
        #         state.add_counter("gemm_ms_profile", 0.0)
        #         state.add_counter("profile_samples", 0)

    state.set_flops(2 * M * N * K)
    state.exec(_run, preamble_fn=_preamble)

    # Don't clear events - they'll be garbage collected and clearing them causes issues
    # if the workspace is reused across benchmark invocations
    # workspace.bench_gemm_start_events = None
    # workspace.bench_gemm_end_events = None
    # workspace.bench_start_events = None
    # workspace.bench_end_events = None
    # workspace.bench_event_count = None
    state.set_bytes((world_size - 1) * M * N * C.element_size())
    launch = workspace.launch_params
    state.add_counter("uses_selector", int(workspace.selector is not None))
    state.add_counter("selector_fallback", int(workspace.selector_fallback))
    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("total_tiles", launch["total_tiles"])
    state.add_counter("num_stages", launch["num_stages"] or 0)
    state.add_counter("aux_rows", workspace.aux_buffer.shape[0])
    state.add_counter("reduce_rows", workspace.a_inbox.shape[0] if workspace.a_inbox is not None else 0)
    state.add_counter("transfer_waves", getattr(workspace, "num_transfer_waves", 0))
    state.add_counter("transfer_rects", getattr(workspace, "num_transfers", 0))
    state.add_counter("reduce_block_m", launch["reduce_block_size_m"])
    state.add_counter("reduce_block_n", launch["reduce_block_size_n"])
    state.add_counter("reduce_num_sms", launch.get("reduce_num_sms", 0))
    state.add_counter("variant_one_shot", 1.0 if variant == "one_shot" else 0.0)
    state.add_counter("variant_two_shot", 1.0 if variant == "two_shot" else 0.0)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
@bench.axis("variant", ["one_shot", "two_shot"])
def matmul_all_reduce_copy_engine(state, ctx):
    """Kernel-based matmul_all_reduce with one_shot/two_shot variants."""
    variant = state["variant"]
    _register_copy_engine(state, ctx, variant=variant)


if __name__ == "__main__":
    bench.main()
