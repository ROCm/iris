#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import gc
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist

import iris


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "tests").is_dir() or not (PROJECT_ROOT / "examples").is_dir():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise FileNotFoundError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

EXAMPLE_DIR = PROJECT_ROOT / "examples" / "31_expert_sharded_moe"
# The example modules use local absolute imports (e.g. `from expert_assignment import ...`),
# so ensure the example directory is on sys.path before loading them.
sys.path.insert(0, str(EXAMPLE_DIR))
EXPERT_ASSIGNMENT = _load_module("expert_assignment_31_moe", EXAMPLE_DIR / "expert_assignment.py")
MOE = _load_module("moe_31_moe", EXAMPLE_DIR / "moe.py")


@pytest.mark.parametrize("n_tokens,d_model,n_expts_act", [(128, 64, 2)])
@pytest.mark.parametrize("fusion_mode", ["unfused", "fused_grouped_matmul_convert_ep_to_dp"])
def test_expert_sharded_moe_matches_reference(n_tokens, d_model, n_expts_act, fusion_mode):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = None
    try:
        shmem = iris.iris()
        rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()
        device = torch.device(f"cuda:{rank}")

        # Keep expert assignment compatible with current distributed world size.
        n_expts_tot = world_size * 2
        if n_tokens % world_size != 0:
            pytest.skip("n_tokens must be divisible by world size for this test")

        n_tokens_local = n_tokens // world_size

        torch.manual_seed(0)
        x_global = torch.randn(n_tokens, d_model, device=device, dtype=torch.bfloat16)
        l_global = torch.rand(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
        w_global = torch.randn(n_expts_tot, d_model, d_model, device=device, dtype=torch.bfloat16)
        b_global = torch.randn(n_expts_tot, d_model, device=device, dtype=torch.float32)

        dist.broadcast(x_global, src=0)
        dist.broadcast(l_global, src=0)
        dist.broadcast(w_global, src=0)
        dist.broadcast(b_global, src=0)

        expt_dict = EXPERT_ASSIGNMENT.make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = EXPERT_ASSIGNMENT.make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        y_global_ref = MOE.mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)

        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_dp_local = x_global[first:last].contiguous()
        l_dp_local = l_global[first:last].contiguous()
        w_ep_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
        b_ep_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

        shmem.barrier()
        fusion_config = MOE.MoeFusionConfig.from_mode_name(fusion_mode)
        z_dp_local = MOE.mixture_of_expt_epsharded(
            x_dp_local,
            l_dp_local,
            w_ep_local,
            b_ep_local,
            expt_assignment,
            n_expts_act,
            shmem,
            fusion_config=fusion_config,
        )

        y_global_tri = torch.empty_like(y_global_ref)
        dist.all_gather_into_tensor(y_global_tri, z_dp_local.contiguous())

        torch.testing.assert_close(y_global_ref, y_global_tri, atol=1e-2, rtol=1e-2)
    finally:
        if shmem is not None:
            try:
                shmem.barrier()
            except Exception:
                pass
            del shmem
            gc.collect()
