# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""Backward compat shim — canonical code moved to iris.mem.triton.context."""

from iris.mem.triton.context import *  # noqa: F401,F403
from iris.mem.triton.context import __translate  # noqa: F401 — dunder excluded from star import
