# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris Experimental Features

This module contains experimental features for Iris that may not be fully stable
or may undergo breaking changes in future releases.

Current experimental features:
- Gluon backend: @aggregate-based implementation via iris.device.gluon

Usage:
    >>> import iris
    >>> from iris.gluon import IrisDeviceCtx
    >>> from triton.experimental import gluon
    >>> from triton.experimental.gluon import language as gl
    >>>
    >>> # Host side
    >>> ctx = iris.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>>
    >>> # Device side
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     ctx.load(buffer, 1)
"""

from iris.device.gluon import context as gluon_context  # noqa: F401

__all__ = ["gluon_context"]
