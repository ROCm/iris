# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Transport-neutral device functions for ``engines=("rdma",)``.

Pointers are symmetric-heap pointers in this rank's address space, as with
``iris.put``; ``heap_bases`` is ``ctx.get_heap_bases()``. ``rdma_ctx`` is the
backend's device context (for RCCL GIN, ``ctx.rdma.device_context(dc)``) and
``BACKEND`` its constexpr id (``ctx.rdma.BACKEND_ID``), which picks the
implementation at compile time: a kernel links only its own backend, and must
be launched with ``extern_libs=ctx.rdma.extern_libs()``.

Every function here is a *program-wide* operation: all threads of the program
must reach it in program-uniform control flow (branching on ``tl.program_id``
is fine, on a per-lane value is not).

Signals: ``put_signal`` adds ``value`` to signal ``signal`` on ``to_rank``, and
``wait_signal`` blocks until the local signal is at least ``least`` -- an
*absolute* value, not a count past the last wait. A caller that waits
repeatedly keeps its own running target (or reads a baseline first with
``ctx.rdma.read_signals``).
"""

import triton
import triton.language as tl

from iris.experimental import rccl_gin  # loads nothing until a kernel is compiled

RCCL_GIN = tl.constexpr(1)


@triton.jit
def _offset(ptr, from_rank, heap_bases):
    return tl.cast(ptr, tl.int64) - tl.load(heap_bases + from_rank)


@triton.jit
def put_signal(
    from_ptr, to_ptr, nbytes, from_rank, to_rank, heap_bases, rdma_ctx, context, signal, value, BACKEND: tl.constexpr
):
    """Write ``nbytes`` from ``from_ptr`` here to ``to_ptr``'s counterpart on
    ``to_rank``, then add ``value`` to ``signal`` there. Observing the signal
    guarantees this put's data is visible, and nothing about any other put."""
    tl.static_assert(BACKEND == RCCL_GIN, "unknown RDMA backend id")
    if BACKEND == RCCL_GIN:
        comm = tl.load(rdma_ctx)
        win = tl.load(rdma_ctx + 1)
        rccl_gin.put_signal(
            comm,
            context,
            to_rank,
            win,
            _offset(to_ptr, from_rank, heap_bases),
            win,
            _offset(from_ptr, from_rank, heap_bases),
            nbytes,
            signal,
            value,
        )


@triton.jit
def put(from_ptr, to_ptr, nbytes, from_rank, to_rank, heap_bases, rdma_ctx, context, BACKEND: tl.constexpr):
    """``put_signal`` without the signal."""
    tl.static_assert(BACKEND == RCCL_GIN, "unknown RDMA backend id")
    if BACKEND == RCCL_GIN:
        comm = tl.load(rdma_ctx)
        win = tl.load(rdma_ctx + 1)
        rccl_gin.put(
            comm,
            context,
            to_rank,
            win,
            _offset(to_ptr, from_rank, heap_bases),
            win,
            _offset(from_ptr, from_rank, heap_bases),
            nbytes,
        )


@triton.jit
def wait_signal(rdma_ctx, context, signal, least, BACKEND: tl.constexpr):
    """Block until local ``signal`` >= ``least`` (absolute); acquire for the
    whole program, so data carried by the puts that raised it is visible."""
    tl.static_assert(BACKEND == RCCL_GIN, "unknown RDMA backend id")
    if BACKEND == RCCL_GIN:
        rccl_gin.wait_signal(tl.load(rdma_ctx), context, signal, least)


@triton.jit
def flush(rdma_ctx, context, BACKEND: tl.constexpr):
    """Wait until every put this program issued on ``context`` has left its
    source buffer, so the buffer may be reused. Remote visibility is the
    signal's job. With RCCL GIN's default (fallback) build this is safe in the
    kernel that issued the puts; see iris/csrc/rccl_gin/build.sh STORE_B128."""
    tl.static_assert(BACKEND == RCCL_GIN, "unknown RDMA backend id")
    if BACKEND == RCCL_GIN:
        rccl_gin.flush(tl.load(rdma_ctx), context)
