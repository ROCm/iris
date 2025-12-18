# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake to build"""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that runs CMake"""
    
    def run(self):
        # Check if CMake is available
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build RDMA extensions")
        
        # Build each extension
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)
        
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        
        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        
        # Build arguments
        build_args = ["--config", "Release"]
        
        # Parallel build
        if hasattr(os, "cpu_count"):
            build_args += [f"-j{os.cpu_count()}"]
        
        # Create build directory
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp
        )


# Check if InfiniBand libraries are available (optional RDMA support)
def has_infiniband():
    """Check if InfiniBand development libraries are available"""
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "libibverbs"],
            capture_output=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        # pkg-config not available, try to find library directly
        for path in ["/usr/lib", "/usr/lib64", "/usr/local/lib"]:
            if os.path.exists(os.path.join(path, "libibverbs.so")):
                return True
        return False


# Build RDMA extension if InfiniBand is available
ext_modules = []
if has_infiniband():
    print("InfiniBand libraries detected - building RDMA backend")
    rdma_ext = CMakeExtension(
        "iris.experimental._iris_rdma_backend",
        sourcedir="iris/experimental/iris_rdma"
    )
    ext_modules.append(rdma_ext)
else:
    print("InfiniBand libraries not found - skipping RDMA backend")
    print("To enable RDMA support, install: libibverbs-dev (Ubuntu/Debian) or rdma-core-devel (RHEL/CentOS)")


# This setup.py provides backward compatibility for legacy metadata fields
# that don't map directly from pyproject.toml's modern PEP 621 format.
setup(
    url="https://rocm.github.io/iris/",
    author="Muhammad Awad, Muhammad Osama, Brandon Potter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild} if ext_modules else {},
)
