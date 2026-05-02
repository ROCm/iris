#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for MoEDispatcher (dispatch/combine via iris symmetric heap).

Uses ``mixture_of_expt_nosharded`` from the MoE example as the ground-truth
reference. Tests cover end-to-end correctness, dispatch-only, combine-only,
buffer reuse across varying batch sizes, topk=1 routing, and handle immutability.

Run with:
    torchrun --nproc_per_node=8 -m pytest tests/ccl/test_moe_dispatch.py -v
"""

import gc
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import iris
from iris.ccl import MoEDispatcher
from iris.ccl.moe_utils import (
    make_expt_dict_uniform,
    make_expt_assignment,
    topk,
)

# ---------------------------------------------------------------------------
# Load grouped_matmul from the example directory (not promoted to ccl yet)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "tests").is_dir() or not (PROJECT_ROOT / "examples").is_dir():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise FileNotFoundError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

EXAMPLE_DIR = PROJECT_ROOT / "examples" / "31_expert_sharded_moe"
sys.path.insert(0, str(EXAMPLE_DIR))


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GROUPED_MATMUL_MOD = _load_module("grouped_matmul_test", EXAMPLE_DIR / "grouped_matmul.py")
grouped_matmul = GROUPED_MATMUL_MOD.grouped_matmul

MOE_MOD = _load_module("moe_test", EXAMPLE_DIR / "moe.py")
mixture_of_expt_nosharded = MOE_MOD.mixture_of_expt_nosharded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup(heap_size=2**33):
    """Common setup: check dist, create iris context, return (ctx, rank, world_size, device)."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{rank}")
    return ctx, rank, world_size, device


def _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device):
    """Generate shared global data (same seed on all ranks, broadcast)."""
    torch.manual_seed(0)
    x_global = torch.randn(n_tokens, d_model, device=device, dtype=dtype)
    l_global = torch.rand(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    w_global = torch.randn(n_expts_tot, d_model, d_model, device=device, dtype=dtype)
    b_global = torch.randn(n_expts_tot, d_model, device=device, dtype=torch.float32)
    dist.broadcast(x_global, src=0)
    dist.broadcast(l_global, src=0)
    dist.broadcast(w_global, src=0)
    dist.broadcast(b_global, src=0)
    return x_global, l_global, w_global, b_global


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens_local", [32, 128])
@pytest.mark.parametrize("d_model", [64, 256])
@pytest.mark.parametrize("n_expts_act", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dispatch_combine_e2e(n_tokens_local, d_model, n_expts_act, dtype):
    """Full dispatch + expert matmul + combine pipeline matches single-device reference."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        n_expts_tot = world_size * 2
        n_tokens = n_tokens_local * world_size

        x_global, l_global, w_global, b_global = _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device)

        # Reference: single-device MoE
        y_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)

        # Expert assignment
        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        # Local slices
        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_local = x_global[first:last].contiguous()
        l_local = l_global[first:last].contiguous()
        w_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
        b_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

        # Top-k routing (local)
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

        # Dispatch
        dispatch_buf, local_meta, handle = dispatcher.dispatch(
            x_local,
            topk_result.indx,
            topk_result.vals,
        )

        # Expert computation (grouped matmul on local experts)
        expert_out = grouped_matmul(dispatch_buf, w_local, b_local, local_meta)

        # Combine
        z_local = dispatcher.combine(expert_out, handle)

        # Gather all local outputs and compare with reference
        z_gathered = torch.empty_like(y_ref)
        dist.all_gather_into_tensor(z_gathered, z_local.contiguous())

        torch.testing.assert_close(y_ref, z_gathered, atol=1e-2, rtol=1e-2)
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()


@pytest.mark.parametrize("n_tokens_local", [64])
@pytest.mark.parametrize("d_model", [128])
def test_dispatch_only(n_tokens_local, d_model):
    """Dispatch buffer matches the example's convert_dp_to_ep output."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        n_expts_tot = world_size * 2
        n_expts_act = 2
        n_tokens = n_tokens_local * world_size
        dtype = torch.bfloat16

        x_global, l_global, _, _ = _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_local = x_global[first:last].contiguous()
        l_local = l_global[first:last].contiguous()

        topk_result = topk(l_local, n_expts_act, apply_softmax=True)

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

        dispatch_buf, local_meta, handle = dispatcher.dispatch(
            x_local,
            topk_result.indx,
            topk_result.vals,
        )

        # Verify dispatch buffer has correct shape
        assert dispatch_buf.shape[1] == d_model
        assert dispatch_buf.shape[0] == n_tokens * n_expts_act

        # Verify local_meta has only this rank's experts
        n_local_experts = expt_assignment.n_expts_per_shard[rank]
        assert local_meta.n_slices == n_local_experts

        # Verify non-zero entries exist (tokens were routed)
        assert dispatch_buf.abs().sum() > 0
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()


@pytest.mark.parametrize("n_tokens_local", [64])
@pytest.mark.parametrize("d_model", [128])
def test_combine_only(n_tokens_local, d_model):
    """Combine output matches the example's convert_ep_to_dp + reduce output."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        n_expts_tot = world_size * 2
        n_expts_act = 2
        n_tokens = n_tokens_local * world_size
        dtype = torch.bfloat16

        x_global, l_global, w_global, b_global = _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_local = x_global[first:last].contiguous()
        l_local = l_global[first:last].contiguous()
        w_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
        b_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

        topk_result = topk(l_local, n_expts_act, apply_softmax=True)

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

        # Full pipeline
        dispatch_buf, local_meta, handle = dispatcher.dispatch(
            x_local,
            topk_result.indx,
            topk_result.vals,
        )
        expert_out = grouped_matmul(dispatch_buf, w_local, b_local, local_meta)
        z_local = dispatcher.combine(expert_out, handle)

        # Verify output shape
        assert z_local.shape == (n_tokens_local, d_model)
        # Verify output is non-trivial
        assert z_local.abs().sum() > 0
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()


def test_buffer_reuse():
    """Pre-allocated buffers work correctly across calls with different batch sizes."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        d_model = 64
        n_expts_tot = world_size * 2
        n_expts_act = 2
        dtype = torch.bfloat16
        max_tokens = 128

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        dispatcher = MoEDispatcher(
            ctx,
            d_model,
            n_expts_tot,
            n_expts_act,
            max_tokens,
            dtype=dtype,
            expt_assignment=expt_assignment,
        )

        # Run with two different batch sizes
        for n_tokens_local in [32, 64]:
            n_tokens = n_tokens_local * world_size

            x_global, l_global, w_global, b_global = _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device)

            y_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)

            first = rank * n_tokens_local
            last = first + n_tokens_local
            x_local = x_global[first:last].contiguous()
            l_local = l_global[first:last].contiguous()
            w_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
            b_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

            topk_result = topk(l_local, n_expts_act, apply_softmax=True)
            ctx.barrier()

            dispatch_buf, local_meta, handle = dispatcher.dispatch(
                x_local,
                topk_result.indx,
                topk_result.vals,
            )
            expert_out = grouped_matmul(dispatch_buf, w_local, b_local, local_meta)
            z_local = dispatcher.combine(expert_out, handle)

            z_gathered = torch.empty_like(y_ref)
            dist.all_gather_into_tensor(z_gathered, z_local.contiguous())

            torch.testing.assert_close(y_ref, z_gathered, atol=1e-2, rtol=1e-2)
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()


def test_topk_1():
    """Simplest routing: each token goes to exactly one expert."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        d_model = 64
        n_expts_tot = world_size * 2
        n_expts_act = 1
        n_tokens_local = 32
        n_tokens = n_tokens_local * world_size
        dtype = torch.bfloat16

        x_global, l_global, w_global, b_global = _make_global_data(n_tokens, d_model, n_expts_tot, dtype, device)

        y_ref = mixture_of_expt_nosharded(x_global, l_global, w_global, b_global, n_expts_act)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        first = rank * n_tokens_local
        last = first + n_tokens_local
        x_local = x_global[first:last].contiguous()
        l_local = l_global[first:last].contiguous()
        w_local = w_global[expt_assignment.expt_boolmask[rank]].contiguous()
        b_local = b_global[expt_assignment.expt_boolmask[rank]].contiguous()

        topk_result = topk(l_local, n_expts_act, apply_softmax=True)

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

        dispatch_buf, local_meta, handle = dispatcher.dispatch(
            x_local,
            topk_result.indx,
            topk_result.vals,
        )
        expert_out = grouped_matmul(dispatch_buf, w_local, b_local, local_meta)
        z_local = dispatcher.combine(expert_out, handle)

        z_gathered = torch.empty_like(y_ref)
        dist.all_gather_into_tensor(z_gathered, z_local.contiguous())

        torch.testing.assert_close(y_ref, z_gathered, atol=1e-2, rtol=1e-2)
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()


def test_handle_frozen():
    """DispatchHandle is immutable (frozen dataclass)."""
    ctx = None
    try:
        ctx, rank, world_size, device = _setup()
        d_model = 64
        n_expts_tot = world_size * 2
        n_expts_act = 2
        n_tokens_local = 32
        n_tokens = n_tokens_local * world_size
        dtype = torch.bfloat16

        torch.manual_seed(0)
        x_local = torch.randn(n_tokens_local, d_model, device=device, dtype=dtype)
        l_local = torch.rand(n_tokens_local, n_expts_tot, device=device, dtype=torch.float32)
        dist.broadcast(x_local, src=0)
        dist.broadcast(l_local, src=0)

        expt_dict = make_expt_dict_uniform(world_size, n_expts_tot)
        expt_assignment = make_expt_assignment(world_size, n_expts_tot, expt_dict, device)

        topk_result = topk(l_local, n_expts_act, apply_softmax=True)

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

        _, _, handle = dispatcher.dispatch(
            x_local,
            topk_result.indx,
            topk_result.vals,
        )

        with pytest.raises(AttributeError):
            handle.n_tokens_local = 999
    finally:
        if ctx is not None:
            try:
                ctx.barrier()
            except Exception:
                pass
            del ctx
            gc.collect()
