# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Functional collective operations for iris compatible with torch.compile.

This module registers iris collective operations as custom operators using
torch.library, making them compatible with torch.compile tracing,
AOTAutograd, and fake tensor mode.

The approach follows the same pattern used by torch.distributed._functional_collectives
for NCCL/RCCL:

1. Define custom ops with torch.library.custom_op
2. Register "fake" (meta) implementations for tensor shape/dtype inference during tracing
3. The real implementations call into iris's existing CCL primitives
4. Operations are functional (return new tensors, don't modify inputs)

Algorithm:
    The underlying algorithm is iris's two-shot all-reduce, which is a port of
    RCCL's approach: a reduce-scatter phase followed by an all-gather phase.
    iris leverages its symmetric heap to perform these operations via direct
    remote memory access (RMA), avoiding the copies that RCCL must perform.

References:
    - RCCL source: src/collectives/all_reduce.cc (ring/tree algorithm dispatch)
    - RCCL algorithms: src/algorithms/ (ring, tree, recursive halving-doubling)
    - torch functional collectives: torch.distributed._functional_collectives
"""

import threading
from typing import Optional

import torch

# Thread-local storage for iris compile context.
# Each thread gets its own context reference, eliminating race conditions
# between concurrent dispatch (e.g. multi-threaded torch.compile, autograd backward).
_thread_local = threading.local()

# Global default context set by setup(), protected by a lock for safe publication.
_default_ctx = None
_default_ctx_lock = threading.Lock()


class IrisCompileContext:
    """
    Manages iris context for torch.compile integration.

    This class holds the iris context and provides workspace caching
    for repeated collective calls during compiled execution.

    Buffer allocation is per-call to avoid race conditions under concurrent
    dispatch (torch.compile multi-threaded backend, autograd backward).
    Workspace metadata (preamble results) is cached since they are
    configuration objects, not data buffers.

    Args:
        iris_instance: An initialized iris.Iris instance.
        config: Optional iris.ccl.Config for kernel parameters.
    """

    def __init__(self, iris_instance, config=None):
        self.ctx = iris_instance
        self._config = config
        self._ag_config = None
        self._rs_config = None
        self._workspaces = {}
        self._workspace_lock = threading.Lock()

    @property
    def config(self):
        if self._config is None:
            from iris.ccl.config import Config
            self._config = Config(
                block_size_m=32,
                block_size_n=64,
                all_reduce_variant="two_shot",
                all_reduce_distribution=1,
            )
        return self._config

    @property
    def ag_config(self):
        """Cached config for all-gather ops."""
        if self._ag_config is None:
            from iris.ccl.config import Config
            self._ag_config = Config(
                all_gather_variant="persistent",
                block_size_m=32,
                block_size_n=64,
            )
        return self._ag_config

    @property
    def rs_config(self):
        """Cached config for reduce-scatter ops."""
        if self._rs_config is None:
            from iris.ccl.config import Config
            self._rs_config = Config(
                reduce_scatter_variant="two_shot",
                block_size_m=32,
                block_size_n=64,
                all_reduce_distribution=1,
            )
        return self._rs_config

    def allocate_buffers(self, shape, dtype):
        """Allocate fresh input/output buffers on the symmetric heap.

        Returns a NEW pair of buffers each call to avoid race conditions
        when multiple threads dispatch ops with the same shape/dtype
        concurrently.
        """
        inp = self.ctx.zeros(shape, dtype=dtype)
        out = self.ctx.zeros(shape, dtype=dtype)
        return inp, out

    def get_workspace(self, key, output_tensor, input_tensor):
        """Get or create a workspace for a given tensor shape/dtype.

        Workspaces are configuration/metadata objects (not data buffers),
        so caching them is safe across concurrent calls.
        """
        with self._workspace_lock:
            if key not in self._workspaces:
                workspace = self.ctx.ccl.all_reduce_preamble(
                    output_tensor, input_tensor, config=self.config
                )
                self.ctx.barrier()
                self._workspaces[key] = workspace
            return self._workspaces[key]

    def clear_workspaces(self):
        """Clear cached workspaces."""
        with self._workspace_lock:
            self._workspaces.clear()


def setup(iris_instance, config=None):
    """
    Initialize the global iris compile context.

    Must be called before using iris functional collectives with torch.compile.
    This registers the iris instance as the backend for compiled collective ops.

    Args:
        iris_instance: An initialized iris.Iris instance.
        config: Optional iris.ccl.Config for kernel parameters.

    Returns:
        IrisCompileContext: The initialized compile context.

    Example:
        >>> import iris
        >>> from iris.compile import setup
        >>> ctx = iris.iris(heap_size=2**30)
        >>> compile_ctx = setup(ctx)
    """
    global _default_ctx
    ctx = IrisCompileContext(iris_instance, config)
    with _default_ctx_lock:
        _default_ctx = ctx
    # Also set on current thread
    _thread_local.iris_ctx = ctx
    return ctx


def _get_ctx():
    """Get the iris compile context for the current thread.

    Returns the thread-local context if set, otherwise falls back to the
    global default. This is safe for concurrent dispatch because each
    thread reads its own storage.
    """
    ctx = getattr(_thread_local, 'iris_ctx', None)
    if ctx is not None:
        return ctx
    # Fall back to global default
    with _default_ctx_lock:
        ctx = _default_ctx
    if ctx is None:
        raise RuntimeError(
            "iris compile context not initialized. "
            "Call iris.compile.setup(ctx) before using functional collectives."
        )
    return ctx


# ============================================================================
# Define custom ops using torch.library.custom_op
# ============================================================================

# --- All-Reduce ---

@torch.library.custom_op("iris::all_reduce", mutates_args=())
def _all_reduce_op(input: torch.Tensor) -> torch.Tensor:
    """
    Functional all-reduce: sum inputs across all ranks.

    Returns a new tensor with the reduced result. The input tensor
    is left unmodified.

    Uses iris's two-shot all-reduce algorithm (port of RCCL's approach):
    1. Reduce-scatter: each rank reduces its assigned tiles from all peers
    2. All-gather: each rank broadcasts its reduced tiles to all peers

    The symmetric heap enables direct remote memory access without copies.
    """
    ctx = _get_ctx()
    iris_ctx = ctx.ctx

    # Allocate fresh buffers per call to avoid race conditions under
    # concurrent dispatch (torch.compile multi-thread, autograd backward).
    inp, output = ctx.allocate_buffers(input.shape, input.dtype)

    # Check if input is already on the symmetric heap
    if iris_ctx.heap.on_symmetric_heap(input):
        inp = input
        # Cache workspace by buffer addresses since heap tensors can differ
        key = ("all_reduce", tuple(input.shape), input.dtype,
               inp.data_ptr(), output.data_ptr())
    else:
        inp.copy_(input)
        key = ("all_reduce", tuple(input.shape), input.dtype)

    workspace = ctx.get_workspace(key, output, inp)

    # Execute all-reduce
    iris_ctx.ccl.all_reduce(
        output, inp, config=ctx.config, workspace=workspace
    )

    return output.clone()


@_all_reduce_op.register_fake
def _all_reduce_fake(input: torch.Tensor) -> torch.Tensor:
    """Fake (meta) implementation for torch.compile tracing."""
    return torch.empty_like(input)


# --- All-Gather ---

@torch.library.custom_op("iris::all_gather", mutates_args=())
def _all_gather_op(input: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Functional all-gather: gather tensors from all ranks and concatenate.

    Returns a tensor of shape (world_size * M, N) containing
    concatenated data from all ranks.
    """
    ctx = _get_ctx()
    iris_ctx = ctx.ctx

    M = input.shape[0]
    N = input.shape[1] if input.dim() > 1 else 1

    # Allocate fresh buffers per call to avoid race conditions
    out_shape = (world_size * M, N) if input.dim() > 1 else (world_size * M,)
    inp = iris_ctx.zeros(input.shape, dtype=input.dtype)
    output = iris_ctx.zeros(out_shape, dtype=input.dtype)

    # Copy input to symmetric heap buffer
    inp.copy_(input)

    iris_ctx.ccl.all_gather(output, inp, config=ctx.ag_config)

    return output.clone()


@_all_gather_op.register_fake
def _all_gather_fake(input: torch.Tensor, world_size: int) -> torch.Tensor:
    """Fake (meta) implementation for torch.compile tracing."""
    M = input.shape[0]
    rest = input.shape[1:] if input.dim() > 1 else ()
    return torch.empty((world_size * M, *rest), dtype=input.dtype, device=input.device)


# --- Reduce-Scatter ---

@torch.library.custom_op("iris::reduce_scatter", mutates_args=())
def _reduce_scatter_op(input: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Functional reduce-scatter: reduce and scatter to each rank.

    Each rank gets a portion of the reduced result.
    Input shape: (M, N), output shape: (M // world_size, N).
    """
    ctx = _get_ctx()
    iris_ctx = ctx.ctx

    # Validate input shape is divisible by world_size
    if input.shape[0] % world_size != 0:
        raise ValueError(
            f"reduce_scatter requires input.shape[0] ({input.shape[0]}) "
            f"to be divisible by world_size ({world_size}), "
            f"but {input.shape[0]} % {world_size} = {input.shape[0] % world_size}"
        )

    # Output shape is input shape with first dim divided by world_size
    out_M = input.shape[0] // world_size
    out_shape = (out_M,) + input.shape[1:]

    # Allocate fresh buffers per call to avoid race conditions
    inp = iris_ctx.zeros(input.shape, dtype=input.dtype)
    output = iris_ctx.zeros(out_shape, dtype=input.dtype)

    # Copy input to symmetric heap buffer
    inp.copy_(input)

    iris_ctx.ccl.reduce_scatter(output, inp, config=ctx.rs_config)

    return output.clone()


@_reduce_scatter_op.register_fake
def _reduce_scatter_fake(input: torch.Tensor, world_size: int) -> torch.Tensor:
    """Fake (meta) implementation for torch.compile tracing."""
    if input.shape[0] % world_size != 0:
        raise ValueError(
            f"reduce_scatter requires input.shape[0] ({input.shape[0]}) "
            f"to be divisible by world_size ({world_size}), "
            f"but {input.shape[0]} % {world_size} = {input.shape[0] % world_size}"
        )
    out_M = input.shape[0] // world_size
    out_shape = (out_M,) + input.shape[1:]
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


# --- Wait Tensor (identity for synchronization) ---

@torch.library.custom_op("iris::wait_tensor", mutates_args=())
def _wait_tensor_op(input: torch.Tensor) -> torch.Tensor:
    """
    Wait for a collective to complete and return the result.

    For iris, collectives are synchronous (barrier at end), so this is
    essentially an identity operation that ensures the current stream is synced.
    Uses stream-specific synchronization to avoid serializing all GPU work.
    """
    torch.cuda.current_stream().synchronize()
    return input.clone()


@_wait_tensor_op.register_fake
def _wait_tensor_fake(input: torch.Tensor) -> torch.Tensor:
    """Fake implementation - identity."""
    return torch.empty_like(input)


# ============================================================================
# Public functional API
# ============================================================================

def all_reduce(input: torch.Tensor, ctx: Optional["IrisCompileContext"] = None) -> torch.Tensor:
    """
    Functional all-reduce that works with torch.compile.

    Reduces the tensor across all ranks using SUM. Returns a new tensor
    with the result. The input tensor is not modified.

    Args:
        input: Input tensor to reduce.
        ctx: Optional IrisCompileContext. If None, uses the global context.

    Returns:
        New tensor containing the sum across all ranks.

    Example:
        >>> import iris
        >>> from iris.compile import functional_collectives as fc
        >>> ctx = iris.iris(heap_size=2**30)
        >>> fc.setup(ctx)
        >>> result = fc.all_reduce(my_tensor)

        >>> # With torch.compile
        >>> @torch.compile
        ... def model(x):
        ...     return fc.all_reduce(x)
        >>> output = model(input_tensor)
    """
    if ctx is not None:
        # Set context on current thread only — no global mutation, no races.
        old = getattr(_thread_local, 'iris_ctx', None)
        _thread_local.iris_ctx = ctx
        try:
            return torch.ops.iris.all_reduce(input)
        finally:
            _thread_local.iris_ctx = old
    return torch.ops.iris.all_reduce(input)


def all_gather(
    input: torch.Tensor,
    world_size: Optional[int] = None,
    ctx: Optional["IrisCompileContext"] = None,
) -> torch.Tensor:
    """
    Functional all-gather that works with torch.compile.

    Gathers tensors from all ranks and concatenates along dim 0.

    Args:
        input: Input tensor (local shard).
        world_size: Number of ranks. If None, auto-detected from context.
        ctx: Optional IrisCompileContext.

    Returns:
        Tensor of shape (world_size * input.shape[0], ...).
    """
    if ctx is not None:
        old = getattr(_thread_local, 'iris_ctx', None)
        _thread_local.iris_ctx = ctx
        try:
            if world_size is None:
                world_size = ctx.ctx.get_num_ranks()
            return torch.ops.iris.all_gather(input, world_size)
        finally:
            _thread_local.iris_ctx = old
    if world_size is None:
        compile_ctx = _get_ctx()
        world_size = compile_ctx.ctx.get_num_ranks()
    return torch.ops.iris.all_gather(input, world_size)


def reduce_scatter(
    input: torch.Tensor,
    world_size: Optional[int] = None,
    ctx: Optional["IrisCompileContext"] = None,
) -> torch.Tensor:
    """
    Functional reduce-scatter that works with torch.compile.

    Reduces and scatters the result across ranks.

    Args:
        input: Input tensor.
        world_size: Number of ranks. If None, auto-detected from context.
        ctx: Optional IrisCompileContext.

    Returns:
        Tensor of shape (input.shape[0] // world_size, ...).
    """
    if ctx is not None:
        old = getattr(_thread_local, 'iris_ctx', None)
        _thread_local.iris_ctx = ctx
        try:
            if world_size is None:
                world_size = ctx.ctx.get_num_ranks()
            return torch.ops.iris.reduce_scatter(input, world_size)
        finally:
            _thread_local.iris_ctx = old
    if world_size is None:
        compile_ctx = _get_ctx()
        world_size = compile_ctx.ctx.get_num_ranks()
    return torch.ops.iris.reduce_scatter(input, world_size)


def wait_tensor(input: torch.Tensor) -> torch.Tensor:
    """
    Wait for async collective to complete.

    For iris, collectives are synchronous, so this is a pass-through.

    Args:
        input: Tensor from a collective operation.

    Returns:
        The completed tensor.
    """
    return torch.ops.iris.wait_tensor(input)


# ============================================================================
# Autograd support
# ============================================================================

# Register autograd formulas for backward pass support
def _all_reduce_backward(ctx, grad_output):
    """Backward for all-reduce is another all-reduce."""
    return torch.ops.iris.all_reduce(grad_output)


def _all_reduce_setup_context(ctx, inputs, output):
    """No context needed for all-reduce backward."""
    pass


torch.library.register_autograd(
    "iris::all_reduce",
    _all_reduce_backward,
    setup_context=_all_reduce_setup_context,
)


# Mark collective ops as having side effects to prevent DCE
torch.fx.node.has_side_effect(torch.ops.iris.all_reduce.default)
torch.fx.node.has_side_effect(torch.ops.iris.all_gather.default)
torch.fx.node.has_side_effect(torch.ops.iris.reduce_scatter.default)
torch.fx.node.has_side_effect(torch.ops.iris.wait_tensor.default)
