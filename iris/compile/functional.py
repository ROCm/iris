# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Functional collective wrappers registered with torch.library.

Each collective is registered as a custom op under the ``iris::`` namespace so
that ``torch.compile`` can trace through calls without graph breaks.  Two
dispatch keys are provided per op:

* **CUDA** -- the real implementation that calls into iris CCL.
* **Meta** -- a shape-only "fake" implementation used during tracing.

The active ``Iris`` context is stored as a module-level singleton.  Call
:func:`set_context` before invoking any compiled function that uses these ops.
"""

from __future__ import annotations

import torch
from torch.library import Library, impl

from iris.host.iris import Iris

# ---------------------------------------------------------------------------
# Iris context singleton
# ---------------------------------------------------------------------------

_iris_ctx: Iris | None = None


def set_context(ctx: Iris) -> None:
    """Set the global iris context used by compiled collectives.

    Must be called once (before the first ``torch.compile``'d invocation)
    with a fully-initialised :class:`~iris.host.iris.Iris` instance.

    Args:
        ctx: An initialised Iris instance.
    """
    global _iris_ctx
    if not isinstance(ctx, Iris):
        raise TypeError(f"Expected an Iris instance, got {type(ctx).__name__}")
    _iris_ctx = ctx


def get_context() -> Iris:
    """Return the current global iris context.

    Raises:
        RuntimeError: If :func:`set_context` has not been called yet.
    """
    if _iris_ctx is None:
        raise RuntimeError(
            "Iris context not set. Call iris.compile.set_context(ctx) "
            "before using compiled collectives."
        )
    return _iris_ctx


# ---------------------------------------------------------------------------
# Library definition
# ---------------------------------------------------------------------------

iris_lib = Library("iris", "DEF")

# ---------------------------------------------------------------------------
# all_reduce
# ---------------------------------------------------------------------------

iris_lib.define("all_reduce(Tensor input, str op='sum') -> Tensor")


@impl(iris_lib, "all_reduce", "CUDA")
def _all_reduce_cuda(input: torch.Tensor, op: str = "sum") -> torch.Tensor:
    ctx = get_context()
    output = ctx.zeros_like(input)
    ctx.ccl.all_reduce(output, input)
    return output


@impl(iris_lib, "all_reduce", "Meta")
def _all_reduce_meta(input: torch.Tensor, op: str = "sum") -> torch.Tensor:
    return torch.empty_like(input)


# ---------------------------------------------------------------------------
# all_gather
# ---------------------------------------------------------------------------

iris_lib.define("all_gather(Tensor input) -> Tensor")


@impl(iris_lib, "all_gather", "CUDA")
def _all_gather_cuda(input: torch.Tensor) -> torch.Tensor:
    ctx = get_context()
    world_size = ctx.get_num_ranks()
    M = input.shape[0]
    N = input.shape[1] if input.ndim >= 2 else 1

    # Reshape to 2-D if needed (iris CCL expects 2-D tensors)
    inp = input if input.ndim >= 2 else input.unsqueeze(1)
    output = ctx.zeros(world_size * inp.shape[0], inp.shape[1], dtype=inp.dtype)
    ctx.ccl.all_gather(output, inp)

    if input.ndim < 2:
        output = output.squeeze(1)
    return output


@impl(iris_lib, "all_gather", "Meta")
def _all_gather_meta(input: torch.Tensor) -> torch.Tensor:
    # During tracing we don't have the real context, but we need world_size.
    # We read it from the singleton if available; otherwise fall back to 1.
    world_size = 1
    if _iris_ctx is not None:
        world_size = _iris_ctx.get_num_ranks()

    if input.ndim >= 2:
        return torch.empty(
            world_size * input.shape[0], *input.shape[1:],
            dtype=input.dtype, device=input.device,
        )
    else:
        return torch.empty(
            world_size * input.shape[0],
            dtype=input.dtype, device=input.device,
        )


# ---------------------------------------------------------------------------
# reduce_scatter
# ---------------------------------------------------------------------------

iris_lib.define("reduce_scatter(Tensor input, str op='sum') -> Tensor")


@impl(iris_lib, "reduce_scatter", "CUDA")
def _reduce_scatter_cuda(input: torch.Tensor, op: str = "sum") -> torch.Tensor:
    ctx = get_context()
    output = ctx.zeros_like(input)
    ctx.ccl.reduce_scatter(output, input)
    return output


@impl(iris_lib, "reduce_scatter", "Meta")
def _reduce_scatter_meta(input: torch.Tensor, op: str = "sum") -> torch.Tensor:
    return torch.empty_like(input)


# ---------------------------------------------------------------------------
# all_to_all
# ---------------------------------------------------------------------------

iris_lib.define("all_to_all(Tensor input) -> Tensor")


@impl(iris_lib, "all_to_all", "CUDA")
def _all_to_all_cuda(input: torch.Tensor) -> torch.Tensor:
    ctx = get_context()
    output = ctx.zeros_like(input)
    ctx.ccl.all_to_all(output, input)
    return output


@impl(iris_lib, "all_to_all", "Meta")
def _all_to_all_meta(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)
