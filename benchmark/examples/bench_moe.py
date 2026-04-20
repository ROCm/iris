#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for expert-sharded MoE."""

import functools
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import iris.bench as bench

# Load MoE example modules.
_project_root = Path(__file__).resolve()
while not (_project_root / "tests").is_dir() or not (_project_root / "examples").is_dir():
    if _project_root == _project_root.parent:
        raise FileNotFoundError("Could not find project root")
    _project_root = _project_root.parent

sys.path.insert(0, str(_project_root / "examples" / "31_expert_sharded_moe"))

from expert_assignment import make_expt_assignment, make_expt_dict_uniform  # noqa: E402
from moe import MoeFusionConfig, mixture_of_expt_epsharded  # noqa: E402


@bench.register
@bench.axis("batch_per_expt", [4, 8, 16, 32, 64, 128, 256])
@bench.axis("d_model", [5760])
@bench.axis("n_expts_tot", [128])
@bench.axis("n_expts_act", [4])
@bench.axis("dtype", [torch.bfloat16])
def moe(state, ctx):
    bpe = state["batch_per_expt"]
    d_model = state["d_model"]
    n_expts_tot = state["n_expts_tot"]
    n_expts_act = state["n_expts_act"]
    dtype = state["dtype"]

    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    n_tokens = bpe * n_expts_tot // n_expts_act
    if n_tokens % world_size != 0:
        state.skip(f"n_tokens={n_tokens} not divisible by world_size={world_size}")
        return

    n_tokens_local = n_tokens // world_size
    device = torch.device(f"cuda:{rank}")

    # Generate tensors and broadcast from rank 0.
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
    x_dp = x_global[first : first + n_tokens_local].contiguous()
    l_dp = l_global[first : first + n_tokens_local].contiguous()
    w_ep = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
    b_ep = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

    fusion_config = MoeFusionConfig.from_mode_name("unfused")

    # Record heap offset for reset between iterations.
    heap_offset = ctx.heap.allocator.heap_offset

    run_dist = functools.partial(
        mixture_of_expt_epsharded,
        x_dp,
        l_dp,
        w_ep,
        b_ep,
        expt_assignment,
        n_expts_act,
        ctx,
        fusion_config=fusion_config,
    )

    # Warmup once to compile kernels.
    run_dist()

    def _preamble():
        ctx.heap.allocator.heap_offset = heap_offset

    state.exec(run_dist, preamble_fn=_preamble)


if __name__ == "__main__":
    bench.main()
