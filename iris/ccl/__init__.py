# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris-ccl: Collective Communication Library for Iris

This module provides configuration for collective communication primitives.
Collective operations are accessed through the Iris instance's ccl attribute:
    >>> shmem = iris.iris()
    >>> shmem.ccl.all_to_all(output_tensor, input_tensor)
"""

from .config import Config

__all__ = ["Config"]
