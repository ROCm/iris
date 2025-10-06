# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
from setuptools import setup


def build(setup_kwargs, config_settings=None):
    """
    Build hook to configure backend via --config-settings.

    This function is called during the build process to handle
    backend configuration from pip install --config-settings.
    """
    if config_settings is None:
        config_settings = {}

    backend = config_settings.get("backend", "amd")

    # Normalize backend names
    if backend.lower() in ("nvidia", "cuda"):
        backend = "cuda"
    elif backend.lower() in ("amd", "rocm", "hip"):
        backend = "hip"
    else:
        backend = "hip"  # Default to hip

    os.environ["IRIS_BACKEND"] = backend
    print(f"Building Iris with backend={backend}")

    return setup_kwargs


# This setup.py provides backward compatibility for legacy metadata fields
# that don't map directly from pyproject.toml's modern PEP 621 format.
setup(
    url="https://rocm.github.io/iris/",
    author="Muhammad Awad, Muhammad Osama, Brandon Potter",
)
