# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Shared driver package types for fabric backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from iris.drivers.base import BaseFabricDriver

__all__ = ["DriverStack"]


@dataclass
class DriverStack:
    """Fabric drivers available for a rank."""

    vendor: str
    fabric: Optional[BaseFabricDriver]
