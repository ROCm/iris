#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Report software and hardware versions for CI documentation.
# Runs inside the container via container_exec.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU_DEVICES should be set by the workflow acquire_gpus.sh step
GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# shellcheck disable=SC2086
"$SCRIPT_DIR/container_exec.sh" $GPU_ARG '
echo "============================================"
echo "  Iris CI — Environment Report"
echo "============================================"
echo ""

echo "--- Driver & ROCm ---"
if command -v amd-smi &> /dev/null; then
    amd-smi version 2>/dev/null || true
    echo ""
    amd-smi static --asic 2>/dev/null | head -30 || true
elif command -v rocm-smi &> /dev/null; then
    rocm-smi --showdriverversion 2>/dev/null || true
    rocm-smi --showid 2>/dev/null | head -20 || true
else
    echo "No amd-smi or rocm-smi found"
fi

if [ -f /opt/rocm/.info/version ]; then
    echo "ROCm version: $(cat /opt/rocm/.info/version)"
elif [ -f /opt/rocm/lib/rocm_version ]; then
    echo "ROCm version: $(cat /opt/rocm/lib/rocm_version)"
fi
echo ""

echo "--- Python ---"
python3 --version 2>/dev/null || echo "python3 not found"
echo ""

echo "--- PyTorch ---"
python3 -c "
import torch
print(f\"torch:        {torch.__version__}\")
print(f\"CUDA/HIP:     {torch.version.hip if hasattr(torch.version, \"hip\") and torch.version.hip else torch.version.cuda}\")
print(f\"GPU count:    {torch.cuda.device_count()}\")
for i in range(min(torch.cuda.device_count(), 1)):
    props = torch.cuda.get_device_properties(i)
    print(f\"GPU 0:        {props.name} (gcnArchName={getattr(props, \"gcnArchName\", \"N/A\")})\")
" 2>/dev/null || echo "PyTorch not available"
echo ""

echo "--- Triton ---"
python3 -c "import triton; print(f\"triton:       {triton.__version__}\")" 2>/dev/null || echo "Triton not available"
echo ""

echo "--- Iris ---"
python3 -c "
try:
    import iris
    print(f\"iris:         {iris.__version__}\")
except Exception:
    print(\"iris not installed (will be installed during test)\")
" 2>/dev/null
echo ""

echo "--- System ---"
uname -r 2>/dev/null || true
echo "============================================"
'
