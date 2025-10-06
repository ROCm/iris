# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
HIP-compatible API facade for Iris.

This module provides a HIP-compatible interface that transparently redirects
to either the HIP backend (AMD GPUs) or CUDA backend (NVIDIA GPUs) based on
build-time configuration, runtime detection, or auto-detection.

The backend is selected based on (in priority order):
1. Build-time configuration (set via pip install --config-settings backend=nvidia)
2. IRIS_BACKEND environment variable (set to 'cuda' or 'hip')
3. Auto-detection based on available libraries
"""

import os


# Detect backend
def _detect_backend():
    """Detect which backend to use based on build-time config, environment, and available libraries."""
    # 1. Check for build-time configuration file first
    try:
        from . import _backend_selected

        backend_config = _backend_selected.BACKEND.lower()
        if backend_config in ("cuda", "nvidia"):
            return "cuda"
        elif backend_config in ("hip", "amd", "rocm"):
            return "hip"
    except (ImportError, AttributeError):
        pass

    # 2. Check environment variable
    backend_env = os.environ.get("IRIS_BACKEND", "").lower()
    if backend_env in ("cuda", "nvidia"):
        return "cuda"
    elif backend_env in ("hip", "amd", "rocm"):
        return "hip"

    # 3. Auto-detect by trying to load libraries
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

# Import all public symbols from the appropriate backend module
if _backend == "cuda":
    from iris._cuda import *  # noqa: F403, F401
else:
    from iris._hip import *  # noqa: F403, F401


# Make backend information available
def get_backend():
    """Get the currently active backend name ('hip' or 'cuda')."""
    return _backend
