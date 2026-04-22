# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Convenience re-exports for Gluon device-side API.

Usage::

    from iris.gluon import IrisDeviceCtx, GluonDeviceTracing
"""

from iris.device.gluon.context import IrisDeviceCtx  # noqa: F401
from iris.device.gluon.tracing import GluonDeviceTracing  # noqa: F401
