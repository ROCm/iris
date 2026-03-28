#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark: iris MoEDispatcher vs naive torch.distributed.all_to_all dispatch.

Run with:
    torchrun --nproc_per_node=8 benchmark/ccl/bench_moe_dispatch.py

Compares:
  1. iris MoEDispatcher (direct iris.store scatter)
  2. Naive all_to_all based dispatch (pack → all_to_all → unpack)
"""

import gc
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import iris
from iris.ccl import MoEDispatcher
from iris.ccl.moe_utils import (
    make_expt_dict_uniform,
    make_expt_assignment,
    topk,
    _make_bitmatrix_metadata,
    make_ragged_tensor_metadata,
    remap_ragged_tensor_metadata,
    reduce,
)

# Load grouped_matmul from examples
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "tests").is_dir() or not (PROJECT_ROOT / "examples").is_dir():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise FileNotFoundError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

EXAMPLE_DIR = PROJECT_ROOT / "examples" / "31_expert_sharded_moe"
sys.path.insert(0, str(EXAMPLE_DIR))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GROUPED_MATMUL_MOD = _load_module("gm", EXAMPLE_DIR / "grouped_matmul.py")
grouped_matmul = GROUPED_MATMUL_MOD.grouped_matmul


def naive_a2a_dispatch(
    tokens, topk_idx, topk_weight, w_local, b_local, expt_assignment, n_expts_tot, n_expts_act, rank, world_size
):
    """Naive dispatch using torch.distributed.all_to_all.

    This is the baseline: gather all tokens, sort by expert, send via all_to_all,
    run expert matmul, send back via all_to_all, reduce.
    """
    n_tokens_local, d_model = tokens.shape
    n_tokens_global = n_tokens_local * world_size
    device = tokens.device

    # All-gather tokens
    x_gathered = torch.empty(n_tokens_global, d_model, dtype=tokens.dtype, device=device)
    dist.all_gather_into_tensor(x_gathered, tokens.contiguous())

    # All-gather routing info
    topk_idx_i32 = topk_idx.contiguous().to(torch.int32)
    topk_weight_f32 = topk_weight.contiguous().to(torch.float32)
    idx_gathered = torch.empty(n_tokens_global, n_expts_act, dtype=torch.int32, device=device)
    val_gathered = torch.empty(n_tokens_global, n_expts_act, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(idx_gathered, topk_idx_i32)
    dist.all_gather_into_tensor(val_gathered, topk_weight_f32)

    # Build routing metadata
    mask_metadata = _make_bitmatrix_metadata(idx_gathered, n_expts_tot)
    dispatch_indx = mask_metadata.row_sorted_indx
    combine_indx = mask_metadata.col_sorted_indx
    expt_sizes = mask_metadata.col_sum
    n_active = int(expt_sizes.sum().item())

    # Gather tokens into expert-sorted order
    gather_idx = torch.div(combine_indx[:n_active], n_expts_act, rounding_mode="trunc")
    x_sorted = torch.zeros(n_active, d_model, dtype=tokens.dtype, device=device)
    valid_gather = gather_idx >= 0
    x_sorted[valid_gather] = x_gathered[gather_idx[valid_gather].long()]

    # Build ragged metadata and remap to local view
    ragged_meta = make_ragged_tensor_metadata(expt_sizes, n_active)
    expt_map = expt_assignment.expt_map[rank, :].contiguous()
    local_meta = remap_ragged_tensor_metadata(ragged_meta, expt_map)

    # Expert computation
    expert_out = grouped_matmul(x_sorted, w_local, b_local, local_meta)

    # Scatter back using combine indices
    y_flat = torch.zeros(n_tokens_global * n_expts_act, d_model, dtype=tokens.dtype, device=device)
    for i in range(n_active):
        dst = combine_indx[i].item()
        if dst >= 0:
            y_flat[dst] = expert_out[i]

    # Reduce
    y_mask = (dispatch_indx != -1).view(n_tokens_global, n_expts_act, 1)
    local_mask = y_mask[rank * n_tokens_local : (rank + 1) * n_tokens_local]
    y_3d = y_flat[rank * n_tokens_local * n_expts_act : (rank + 1) * n_tokens_local * n_expts_act]
    y_3d = y_3d.view(n_tokens_local, n_expts_act, d_model)
    local_mask = local_mask.expand_as(y_3d).contiguous()
    z_local, _ = reduce(y_3d, dim=1, mask=local_mask)
    return z_local


def iris_dispatch_combine(tokens, topk_idx, topk_weight, w_local, b_local, dispatcher, local_meta_out):
    """Iris MoEDispatcher dispatch + expert matmul + combine."""
    dispatch_buf, local_meta, handle = dispatcher.dispatch(tokens, topk_idx, topk_weight)
    expert_out = grouped_matmul(dispatch_buf, w_local, b_local, local_meta)
    z_local = dispatcher.combine(expert_out, handle)
    return z_local


def benchmark(fn, warmup=20, measured=100):
    """Benchmark a function, return median latency in us."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measured
    times = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[9 * len(times) // 10]
    return median, p10, p90


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    ctx = iris.iris()
    dtype = torch.bfloat16

    # Test configurations
    configs = [
        # (n_tokens_local, d_model, n_expts_tot, n_expts_act)
        (32, 4096, world_size * 8, 2),  # Small decode batch, LLM hidden dim
        (64, 4096, world_size * 8, 2),  # Medium decode
        (128, 4096, world_size * 8, 2),  # Large decode
        (256, 4096, world_size * 8, 2),  # Prefill-like
        (512, 4096, world_size * 8, 2),  # Large prefill
        (128, 2048, world_size * 4, 2),  # Smaller model
        (128, 4096, world_size * 8, 1),  # topk=1
        (128, 4096, world_size * 8, 4),  # topk=4
    ]

    if rank == 0:
        print("# MoE Dispatch/Combine Benchmark — iris vs naive all_to_all")
        print("## Hardware")
        print(f"- GPUs: {world_size}x MI300X (or similar)")
        print(f"- dtype: {dtype}")
        print("- Warmup: 20 iterations, Measured: 100 iterations")
        print()
        print("| T_local | H    | E_tot | k | iris (us) | naive (us) | Speedup |")
        print("|---------|------|-------|---|-----------|------------|---------|")

    for n_tokens_local, d_model, n_expts_tot, n_expts_act in configs:
        n_tokens = n_tokens_local * world_size

        torch.manual_seed(0)
        x_global = torch.randn(n_tokens, d_model, device=device, dtype=dtype)
        l_global = torch.rand(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
        w_global = torch.randn(n_expts_tot, d_model, d_model, device=device, dtype=dtype)
        b_global = torch.randn(n_expts_tot, d_model, device=device, dtype=torch.float32)
        dist.broadcast(x_global, src=0)
        dist.broadcast(l_global, src=0)
        dist.broadcast(w_global, src=0)
        dist.broadcast(b_global, src=0)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_local = x_global[first:last].contiguous()
        l_local = l_global[first:last].contiguous()
        w_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
        b_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

        topk_result = topk(l_local, n_expts_act, apply_softmax=True)

        # Create dispatcher
        dispatcher = MoEDispatcher(
            ctx,
            d_model,
            n_expts_tot,
            n_expts_act,
            n_tokens_local,
            dtype=dtype,
            expt_assignment=expt_assignment,
        )
        ctx.barrier()

        # Benchmark iris
        def _iris_fn(x=x_local, idx=topk_result.indx, vals=topk_result.vals, w=w_local, b=b_local, d=dispatcher):
            return iris_dispatch_combine(x, idx, vals, w, b, d, None)

        iris_time, _, _ = benchmark(_iris_fn, warmup=20, measured=100)

        # Benchmark naive
        naive_time, _, _ = benchmark(
            lambda: naive_a2a_dispatch(
                x_local,
                topk_result.indx,
                topk_result.vals,
                w_local,
                b_local,
                expt_assignment,
                n_expts_tot,
                n_expts_act,
                rank,
                world_size,
            ),
            warmup=20,
            measured=100,
        )

        speedup = naive_time / iris_time if iris_time > 0 else float("inf")

        if rank == 0:
            print(
                f"| {n_tokens_local:>7} | {d_model:>4} | {n_expts_tot:>5} | {n_expts_act} | {iris_time:>9.0f} | {naive_time:>10.0f} | {speedup:>6.2f}x |"
            )

        # Cleanup
        del dispatcher
        gc.collect()
        ctx.barrier()

    if rank == 0:
        print()
        print("## Analysis")
        print("iris MoEDispatcher uses direct symmetric heap scatter (iris.store)")
        print("which avoids the all_to_all pack/unpack overhead. The naive approach")
        print("uses all_gather + host-side sorting, which involves extra copies.")

    del ctx
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
