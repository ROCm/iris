#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Collect environment versions and emit JSON to stdout.

Runs inside the CI container. The calling shell script handles
$GITHUB_STEP_SUMMARY and stdout formatting on the host side.
"""

import json
import os
import subprocess
import sys


def collect():
    info = {}

    # Driver + ROCm via amd-smi version --json
    try:
        raw = subprocess.check_output(
            ["amd-smi", "version", "--json"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        smi = json.loads(raw)
        if isinstance(smi, list):
            smi = smi[0]
        info["driver"] = smi.get("amdgpu_version", "")
        info["rocm"] = smi.get("rocm_version", "")
    except Exception:
        pass

    # ROCm fallback from filesystem
    if not info.get("rocm"):
        for p in ["/opt/rocm/.info/version", "/opt/rocm/lib/rocm_version"]:
            if os.path.isfile(p):
                info["rocm"] = open(p).read().strip()
                break

    # Python
    v = sys.version_info
    info["python"] = f"{v.major}.{v.minor}.{v.micro}"

    # PyTorch, HIP, GPU
    try:
        import torch

        info["torch"] = torch.__version__
        info["hip"] = (
            getattr(torch.version, "hip", None)
            or getattr(torch.version, "cuda", None)
            or ""
        )
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_arch"] = getattr(props, "gcnArchName", "N/A")
    except Exception:
        pass

    # Triton
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        pass

    # Kernel
    try:
        info["kernel"] = os.uname().release
    except Exception:
        pass

    return info


if __name__ == "__main__":
    print(json.dumps(collect()))
