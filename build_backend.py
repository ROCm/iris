# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Custom build backend to support backend selection via --config-settings.

This allows users to install Iris with:
    pip install . --config-settings backend=nvidia
or:
    pip install . --config-settings backend=hip
"""

import os
from setuptools import build_meta as _orig

# Re-export all setuptools build_meta functions
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist


def _write_backend_config(config_settings):
    """Write backend configuration file based on --config-settings."""
    backend = None

    if config_settings:
        backend = config_settings.get("backend", "").lower()

    # Normalize backend names
    if backend in ("nvidia", "cuda"):
        backend = "cuda"
    elif backend in ("amd", "rocm", "hip"):
        backend = "hip"
    else:
        backend = None  # Auto-detect at runtime

    # Write backend selection to a Python file
    if backend:
        with open("iris/_backend_selected.py", "w") as f:
            f.write(f'BACKEND = "{backend}"\n')
        print(f"Iris: Configured to use {backend} backend")
    else:
        # Remove file if it exists (auto-detect mode)
        if os.path.exists("iris/_backend_selected.py"):
            os.remove("iris/_backend_selected.py")
        print("Iris: No backend specified, will auto-detect at runtime")


# Wrap build functions to inject backend configuration
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with backend configuration."""
    _write_backend_config(config_settings)
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build sdist with backend configuration."""
    _write_backend_config(config_settings)
    return _orig.build_sdist(sdist_directory, config_settings)
