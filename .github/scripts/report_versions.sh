#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Report software and hardware versions for CI documentation.
# Collects versions from inside the container, prints to stdout,
# and writes a GitHub Actions job summary table.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU_DEVICES should be set by the workflow acquire_gpus.sh step
GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# Collect all versions as JSON from inside the container using Python
# shellcheck disable=SC2086
VERSION_JSON=$("$SCRIPT_DIR/container_exec.sh" $GPU_ARG '
python3 -c "
import json, subprocess, sys, os

info = {}

# Driver + ROCm via amd-smi version --json (structured, no regex)
try:
    raw = subprocess.check_output([\"amd-smi\", \"version\", \"--json\"],
                                  stderr=subprocess.DEVNULL, timeout=10)
    smi = json.loads(raw)
    if isinstance(smi, list):
        smi = smi[0]
    info[\"driver\"] = smi.get(\"amdgpu_version\", \"\")
    info[\"rocm\"] = smi.get(\"rocm_version\", \"\")
except Exception:
    pass

# ROCm fallback from filesystem
if not info.get(\"rocm\"):
    for p in [\"/opt/rocm/.info/version\", \"/opt/rocm/lib/rocm_version\"]:
        if os.path.isfile(p):
            info[\"rocm\"] = open(p).read().strip()
            break

# Python
info[\"python\"] = f\"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\"

# PyTorch, HIP, GPU
try:
    import torch
    info[\"torch\"] = torch.__version__
    info[\"hip\"] = getattr(torch.version, \"hip\", None) or getattr(torch.version, \"cuda\", None) or \"\"
    info[\"gpu_count\"] = torch.cuda.device_count()
    if torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        info[\"gpu_name\"] = props.name
        info[\"gpu_arch\"] = getattr(props, \"gcnArchName\", \"N/A\")
except Exception:
    pass

# Triton
try:
    import triton
    info[\"triton\"] = triton.__version__
except Exception:
    pass

# Kernel
try:
    info[\"kernel\"] = os.uname().release
except Exception:
    pass

print(json.dumps(info))
"
')

# Extract the JSON line (last line, skip any container startup noise)
VERSION_JSON=$(echo "$VERSION_JSON" | grep '^{' | tail -1)

if [ -z "$VERSION_JSON" ]; then
    echo "[WARN] Failed to collect version info"
    exit 0
fi

# Parse JSON fields with Python (no jq dependency needed)
read_field() {
    echo "$VERSION_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$1','unknown'))" 2>/dev/null || echo "unknown"
}

DRIVER=$(read_field driver)
ROCM=$(read_field rocm)
PYTHON=$(read_field python)
TORCH=$(read_field torch)
HIP=$(read_field hip)
TRITON=$(read_field triton)
GPU_NAME=$(read_field gpu_name)
GPU_ARCH=$(read_field gpu_arch)
GPU_COUNT=$(read_field gpu_count)
KERNEL=$(read_field kernel)

# Print to stdout
echo "============================================"
echo "  Iris CI — Environment Report"
echo "============================================"
echo "  Driver:   $DRIVER"
echo "  ROCm:     $ROCM"
echo "  Python:   $PYTHON"
echo "  PyTorch:  $TORCH"
echo "  HIP:      $HIP"
echo "  Triton:   $TRITON"
echo "  GPU:      $GPU_NAME ($GPU_ARCH) × $GPU_COUNT"
echo "  Kernel:   $KERNEL"
echo "============================================"

# Write GitHub Actions job summary
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
    cat >> "$GITHUB_STEP_SUMMARY" <<SUMMARY
### Environment

| Component | Version |
|-----------|---------|
| Driver | $DRIVER |
| ROCm | $ROCM |
| Python | $PYTHON |
| PyTorch | $TORCH |
| HIP | $HIP |
| Triton | $TRITON |
| GPU | $GPU_NAME ($GPU_ARCH) × $GPU_COUNT |
| Kernel | $KERNEL |

SUMMARY
fi
