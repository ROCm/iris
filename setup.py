# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

from setuptools import setup

# This setup.py provides backward compatibility for legacy metadata fields
# that don't map directly from pyproject.toml's modern PEP 621 format.
setup(
    url="https://rocm.github.io/iris/",
    author="Muhammad Awad, Muhammad Osama, Brandon Potter",
)
