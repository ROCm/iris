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

    # Also set environment variable as suggested in setup.py
    if backend:
        os.environ["IRIS_BACKEND"] = backend

    # Write configuration file
    config_dir = os.path.join("iris", ".config")
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(config_dir, "backend.txt")
    if backend:
        with open(config_file, "w") as f:
            f.write(backend)
        print(f"Iris: Configured to use {backend} backend")
    else:
        # Remove config file if it exists (auto-detect mode)
        if os.path.exists(config_file):
            os.remove(config_file)
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
