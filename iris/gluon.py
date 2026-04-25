# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Convenience re-exports for Gluon device-side API.

Usage::

    from iris.gluon import IrisDeviceCtx, GluonDeviceTracing
"""

from iris.mem.gluon.context import Context as IrisDeviceCtx  # noqa: F401
from iris.mem.gluon.tracing import Tracing as GluonDeviceTracing  # noqa: F401
