# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
HIP-compatible API facade for Iris.

This module provides a HIP-compatible interface that transparently redirects
to either the HIP backend (AMD GPUs) or CUDA backend (NVIDIA GPUs) based on
runtime detection or configuration.

The backend is selected based on:
1. IRIS_BACKEND environment variable (set to 'cuda' or 'hip')
2. Auto-detection based on available libraries
"""

import os
import sys

# Detect backend
def _detect_backend():
    """Detect which backend to use based on environment and available libraries."""
    backend_env = os.environ.get("IRIS_BACKEND", "").lower()
    if backend_env in ("cuda", "nvidia"):
        return "cuda"
    elif backend_env in ("hip", "amd", "rocm"):
        return "hip"
    
    # Auto-detect by trying to load libraries
    import ctypes
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

# Import from the appropriate backend module
if _backend == "cuda":
    from iris.cuda import *
else:
    from iris._hip import *

# Make backend information available
def get_backend():
    """Get the currently active backend name ('hip' or 'cuda')."""
    return _backend
