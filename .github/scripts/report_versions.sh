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

# Collect versions as KEY=VALUE pairs from inside the container
# shellcheck disable=SC2086
VERSION_DATA=$("$SCRIPT_DIR/container_exec.sh" $GPU_ARG '
# Driver version — try multiple sources
DRIVER=""
if [ -f /sys/module/amdgpu/version ]; then
    DRIVER=$(cat /sys/module/amdgpu/version 2>/dev/null)
fi
if [ -z "$DRIVER" ] && command -v modinfo &> /dev/null; then
    DRIVER=$(modinfo amdgpu 2>/dev/null | grep "^version:" | head -1 | awk "{print \$2}")
fi
if [ -z "$DRIVER" ] && command -v amd-smi &> /dev/null; then
    DRIVER=$(amd-smi version 2>/dev/null | grep -iE "driver|AMDSMI" | head -1 | sed "s/.*: *//; s/^ *//; s/ *$//")
fi
if [ -z "$DRIVER" ] && command -v rocm-smi &> /dev/null; then
    DRIVER=$(rocm-smi --showdriverversion 2>/dev/null | grep -i "driver" | head -1 | sed "s/.*: *//")
fi
echo "DRIVER=${DRIVER:-unknown}"

# ROCm version
if [ -f /opt/rocm/.info/version ]; then
    echo "ROCM=$(cat /opt/rocm/.info/version)"
elif [ -f /opt/rocm/lib/rocm_version ]; then
    echo "ROCM=$(cat /opt/rocm/lib/rocm_version)"
else
    echo "ROCM=unknown"
fi

# Python
PYVER=$(python3 --version 2>/dev/null | sed "s/Python //")
echo "PYTHON=${PYVER:-unknown}"

# PyTorch, HIP, GPU info
python3 -c "
import torch
print(f\"TORCH={torch.__version__}\")
hip = torch.version.hip if hasattr(torch.version, \"hip\") and torch.version.hip else torch.version.cuda
print(f\"HIP={hip or \"unknown\"}\")
print(f\"GPU_COUNT={torch.cuda.device_count()}\")
if torch.cuda.device_count() > 0:
    props = torch.cuda.get_device_properties(0)
    print(f\"GPU_NAME={props.name}\")
    print(f\"GPU_ARCH={getattr(props, \"gcnArchName\", \"N/A\")}\")
else:
    print(\"GPU_NAME=none\")
    print(\"GPU_ARCH=N/A\")
" 2>/dev/null || echo "TORCH=unavailable"

# Triton
TRITON=$(python3 -c "import triton; print(triton.__version__)" 2>/dev/null)
echo "TRITON=${TRITON:-unavailable}"

# Kernel
echo "KERNEL=$(uname -r 2>/dev/null || echo unknown)"
')

# Parse KEY=VALUE pairs
declare -A V
while IFS='=' read -r key value; do
    [[ -n "$key" && "$key" != *" "* ]] && V["$key"]="$value"
done <<< "$VERSION_DATA"

# Print to stdout
echo "============================================"
echo "  Iris CI — Environment Report"
echo "============================================"
echo "  Driver:   ${V[DRIVER]:-unknown}"
echo "  ROCm:     ${V[ROCM]:-unknown}"
echo "  Python:   ${V[PYTHON]:-unknown}"
echo "  PyTorch:  ${V[TORCH]:-unknown}"
echo "  HIP:      ${V[HIP]:-unknown}"
echo "  Triton:   ${V[TRITON]:-unknown}"
echo "  GPU:      ${V[GPU_NAME]:-unknown} (${V[GPU_ARCH]:-N/A}) × ${V[GPU_COUNT]:-0}"
echo "  Kernel:   ${V[KERNEL]:-unknown}"
echo "============================================"

# Write GitHub Actions job summary
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
    cat >> "$GITHUB_STEP_SUMMARY" <<SUMMARY
### Environment

| Component | Version |
|-----------|---------|
| Driver | ${V[DRIVER]:-unknown} |
| ROCm | ${V[ROCM]:-unknown} |
| Python | ${V[PYTHON]:-unknown} |
| PyTorch | ${V[TORCH]:-unknown} |
| HIP | ${V[HIP]:-unknown} |
| Triton | ${V[TRITON]:-unknown} |
| GPU | ${V[GPU_NAME]:-unknown} (${V[GPU_ARCH]:-N/A}) × ${V[GPU_COUNT]:-0} |
| Kernel | ${V[KERNEL]:-unknown} |

SUMMARY
fi
