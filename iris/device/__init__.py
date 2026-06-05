# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side utilities for Iris.

This module provides low-level device-side functions for use in Triton kernels,
including SDMA queue management and packet construction utilities.
"""

from . import sdma_utils

__all__ = ["sdma_utils"]
