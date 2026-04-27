# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon backend availability check.

This is the single place where gluon availability is determined.
All other modules should import GLUON_AVAILABLE from here.
"""

try:
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401
    from iris.mem.gluon.context import Context as IrisDeviceCtx  # noqa: F401

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False
