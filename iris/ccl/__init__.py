# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
iris-ccl: Collective Communication Library for Iris

This module provides standalone collective communication primitives
that match PyTorch's RCCL/NCCL interface.
"""

from .all_to_all import all_to_all
from .config import Config

__all__ = ["all_to_all", "Config"]


