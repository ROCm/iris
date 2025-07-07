#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class HIPBuildPy(build_py):
    """Custom build command that also builds the HIP library."""

    def run(self):
        # Build the HIP library first
        self.build_hip_library()
        # Then run the normal Python build
        super().run()

    def build_hip_library(self):
        """Build the finegrained allocator using hipcc."""
        # Get the project root directory (where setup.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(project_root, "csrc", "finegrained_alloc")
        src_file = os.path.join(src_dir, "finegrained_allocator.hip")
        output_file = os.path.join(src_dir, "libfinegrained_allocator.so")

        # Check if source file exists
        if not os.path.exists(src_file):
            raise FileNotFoundError(
                f"Source file not found: {src_file}\n"
                "This might happen if the repository is incomplete or if you're "
                "installing from a source distribution that doesn't include the C++ source."
            )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        basic_warnings = ["-Wall", "-Wextra", "-Werror"]
        strict_warnings = [
            "-pedantic",
            "-Wshadow",
            "-Wnon-virtual-dtor",
            "-Wold-style-cast",
            "-Wcast-align",
            "-Woverloaded-virtual",
            "-Wconversion",
            "-Wsign-conversion",
            "-Wnull-dereference",
            "-Wdouble-promotion",
            "-Wformat=2",
        ]
        std_flags = ["-std=c++17"]
        output_flags = ["-shared", "-fPIC", "-o", output_file]

        cmd = ["hipcc"] + basic_warnings + strict_warnings + std_flags + output_flags + [src_file]

        print(f"Building finegrained allocator: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, cwd=src_dir, check=True, capture_output=True, text=True)
            print(f"Successfully built: {output_file}")

            # Copy the built library to the iris package directory for installation
            iris_package_dir = os.path.join(project_root, "iris")
            target_dir = os.path.join(iris_package_dir, "csrc", "finegrained_alloc")
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, "libfinegrained_allocator.so")
            shutil.copy2(output_file, target_file)
            print(f"Copied library to: {target_file}")

        except subprocess.CalledProcessError as e:
            print(f"Build failed with return code {e.returncode}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            print("hipcc not found. Please ensure ROCm/HIP is installed.")
            print(
                "You can install ROCm following the instructions at: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html"
            )
            raise


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_py": HIPBuildPy,
        },
        package_data={
            "iris": ["csrc/finegrained_alloc/libfinegrained_allocator.so"],
        },
    )
