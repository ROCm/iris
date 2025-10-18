# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris Experimental Features

This module contains experimental features for Iris that may not be fully stable
or may undergo breaking changes in future releases.

Current experimental features:
- iris_gluon: Gluon-based implementation using @aggregate and @gluon.jit

Usage:
    >>> import iris.experimental.iris_gluon as iris_gl
    >>> from triton.experimental import gluon
    >>> from triton.experimental.gluon import language as gl
    >>>
    >>> # Host side
    >>> ctx = iris_gl.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>>
    >>> # Device side
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     ctx.load(buffer, 1)
"""

from . import iris_gluon

__all__ = ["iris_gluon"]
