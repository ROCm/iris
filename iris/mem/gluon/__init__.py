# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""Gluon device-side context and tracing."""

from .context import Context  # noqa: F401
from .context import Context as IrisDeviceCtx  # noqa: F401  backward compat
from .tracing import Tracing  # noqa: F401
from .tracing import Tracing as GluonDeviceTracing  # noqa: F401  backward compat
