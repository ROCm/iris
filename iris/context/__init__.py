# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side contexts for Iris kernels.

Provides ``TritonContext`` and ``GluonContext`` — device-side aggregates that
decode the context tensor from ``Iris.get_device_context()`` and expose RMA
operations (load, store, copy, get, put, atomics) inside Triton or Gluon
kernels respectively.

Each context also embeds a device-side tracing aggregate
(``TritonDeviceTracing`` / ``GluonDeviceTracing``) that records events into
SoA buffers when compiled with ``tracing=True``.

Usage (Triton)::

    import iris
    from iris.context import TritonContext

    ctx = iris.iris(heap_size=2**30)
    context_tensor = ctx.get_device_context()

    @triton.jit
    def kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr):
        ctx = TritonContext.initialize(context_tensor, rank, world_size)
        data = ctx.load(buffer + offsets, from_rank=1, mask=mask)

Usage (Gluon)::

    import iris
    from iris.context import GluonContext

    ctx = iris.iris(heap_size=2**30)
    context_tensor = ctx.get_device_context()

    @gluon.jit
    def kernel(GluonContext: gl.constexpr, context_tensor):
        ctx = GluonContext.initialize(context_tensor)
        data = ctx.load(buffer + offsets, 1, mask=mask)
"""

from .triton import TritonContext, TritonDeviceTracing, _translate_ptr

# Gluon is optional — don't fail if triton.experimental.gluon is missing
try:
    from .gluon import GluonContext, GluonDeviceTracing

    __all__ = [
        "TritonContext",
        "TritonDeviceTracing",
        "GluonContext",
        "GluonDeviceTracing",
        "_translate_ptr",
    ]
except ImportError:
    __all__ = [
        "TritonContext",
        "TritonDeviceTracing",
        "_translate_ptr",
    ]
