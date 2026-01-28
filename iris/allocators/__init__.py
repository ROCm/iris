# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Allocator interfaces for Iris symmetric heap management.
"""

from .base import BaseAllocator
from .torch_allocator import TorchAllocator

__all__ = ["BaseAllocator", "TorchAllocator"]
