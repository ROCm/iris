# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Local (intra-node) memory drivers."""

from iris.drivers.local.amd import LocalHipDriver
from iris.drivers.local.nvidia import LocalCudaDriver

__all__ = ["LocalHipDriver", "LocalCudaDriver"]
