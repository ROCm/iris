# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for torch.compile functional collectives (iris.compile).

These tests verify that torch.compile'd functions using iris custom ops
produce results identical to eager-mode execution.  Each test:

  1. Runs the collective in eager mode via torch.ops.iris.*.
  2. Runs the same function through torch.compile(backend="inductor").
  3. Compares the two outputs element-wise.
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
import iris.compile  # registers torch.ops.iris.*


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_unless_distributed():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _make_ctx(heap_size: int = 2**33):
    """Create an Iris context and register it with iris.compile."""
    ctx = iris.iris(heap_size)
    iris.compile.set_context(ctx)
    return ctx


def _cleanup(ctx):
    """Tear-down helper matching existing test conventions."""
    ctx.barrier()
    del ctx
    gc.collect()


# ---------------------------------------------------------------------------
# all_reduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 64), (1024, 256)])
def test_all_reduce_compile(dtype, M, N):
    """Compiled all_reduce matches eager all_reduce."""
    _skip_unless_distributed()
    ctx = _make_ctx()
    rank = ctx.get_rank()

    # Build input on symmetric heap
    inp = ctx.zeros(M, N, dtype=dtype)
    inp.fill_(float(rank + 1))

    # Eager
    eager_out = torch.ops.iris.all_reduce(inp)

    # Compiled
    @torch.compile(backend="inductor")
    def compiled_ar(x):
        return torch.ops.iris.all_reduce(x)

    compiled_out = compiled_ar(inp)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    try:
        assert torch.allclose(compiled_out, eager_out, atol=atol), (
            f"all_reduce mismatch: max diff = {(compiled_out - eager_out).abs().max().item()}"
        )
    finally:
        _cleanup(ctx)


# ---------------------------------------------------------------------------
# all_gather
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 64), (1024, 256)])
def test_all_gather_compile(dtype, M, N):
    """Compiled all_gather matches eager all_gather."""
    _skip_unless_distributed()
    ctx = _make_ctx()
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    inp = ctx.zeros(M, N, dtype=dtype)
    inp.fill_(float(rank + 1))

    # Eager
    eager_out = torch.ops.iris.all_gather(inp)
    assert eager_out.shape[0] == M * world_size

    # Compiled
    @torch.compile(backend="inductor")
    def compiled_ag(x):
        return torch.ops.iris.all_gather(x)

    compiled_out = compiled_ag(inp)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    try:
        assert torch.allclose(compiled_out, eager_out, atol=atol), (
            f"all_gather mismatch: max diff = {(compiled_out - eager_out).abs().max().item()}"
        )
    finally:
        _cleanup(ctx)


# ---------------------------------------------------------------------------
# reduce_scatter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M, N", [(128, 64), (1024, 256)])
def test_reduce_scatter_compile(dtype, M, N):
    """Compiled reduce_scatter matches eager reduce_scatter."""
    _skip_unless_distributed()
    ctx = _make_ctx()
    rank = ctx.get_rank()

    inp = ctx.zeros(M, N, dtype=dtype)
    inp.fill_(float(rank + 1))

    # Eager
    eager_out = torch.ops.iris.reduce_scatter(inp)

    # Compiled
    @torch.compile(backend="inductor")
    def compiled_rs(x):
        return torch.ops.iris.reduce_scatter(x)

    compiled_out = compiled_rs(inp)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    try:
        assert torch.allclose(compiled_out, eager_out, atol=atol), (
            f"reduce_scatter mismatch: max diff = {(compiled_out - eager_out).abs().max().item()}"
        )
    finally:
        _cleanup(ctx)


# ---------------------------------------------------------------------------
# all_to_all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M, N_per_rank", [(128, 64), (1024, 128)])
def test_all_to_all_compile(dtype, M, N_per_rank):
    """Compiled all_to_all matches eager all_to_all."""
    _skip_unless_distributed()
    ctx = _make_ctx()
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    N = N_per_rank * world_size
    inp = ctx.zeros(M, N, dtype=dtype)
    inp.fill_(float(rank + 1))

    # Eager
    eager_out = torch.ops.iris.all_to_all(inp)

    # Compiled
    @torch.compile(backend="inductor")
    def compiled_a2a(x):
        return torch.ops.iris.all_to_all(x)

    compiled_out = compiled_a2a(inp)

    atol = 1e-3 if dtype == torch.float16 else 1e-5
    try:
        assert torch.allclose(compiled_out, eager_out, atol=atol), (
            f"all_to_all mismatch: max diff = {(compiled_out - eager_out).abs().max().item()}"
        )
    finally:
        _cleanup(ctx)


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def test_set_context_rejects_non_iris():
    """set_context raises TypeError for non-Iris objects."""
    with pytest.raises(TypeError, match="Expected an Iris instance"):
        iris.compile.set_context("not an iris context")


def test_get_context_without_set():
    """get_context raises RuntimeError before set_context is called."""
    # Temporarily clear the global context
    import iris.compile.functional as _mod

    saved = _mod._iris_ctx
    _mod._iris_ctx = None
    try:
        with pytest.raises(RuntimeError, match="Iris context not set"):
            _mod.get_context()
    finally:
        _mod._iris_ctx = saved
