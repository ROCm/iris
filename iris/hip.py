# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
HIP-compatible API facade for Iris.

This module provides a HIP-compatible interface that transparently redirects
to either the HIP backend (AMD GPUs) or CUDA backend (NVIDIA GPUs) based on
auto-detection.
"""

import ctypes


# Detect backend
def _detect_backend():
    """Detect which backend to use based on available libraries."""
    # Auto-detect by trying to load libraries
    try:
        ctypes.cdll.LoadLibrary("libamdhip64.so")
        return "hip"
    except (OSError, FileNotFoundError):
        pass

    try:
        ctypes.cdll.LoadLibrary("libcudart.so")
        return "cuda"
    except (OSError, FileNotFoundError):
        pass

    # Default to hip for backward compatibility
    return "hip"


_backend = _detect_backend()

# Import all public symbols from the appropriate backend module
if _backend == "cuda":
    from iris._cuda import *  # noqa: F403, F401
else:
    from iris._hip import *  # noqa: F403, F401


# Make backend information available
def get_backend():
    """Get the currently active backend name ('hip' or 'cuda')."""
    return _backend
