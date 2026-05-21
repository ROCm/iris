# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared driver package types for memory backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from iris.drivers.base import BaseDriver

__all__ = ["DriverStack"]


@dataclass
class DriverStack:
    """Driver available for a rank."""

    vendor: str
    driver: Optional[BaseDriver]
