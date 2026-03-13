#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Acquire GPUs for CI workflows - to be called as a workflow step
# Usage: acquire_gpus.sh <num_gpus>
#
# Exports GPU_DEVICES environment variable to $GITHUB_ENV for use in subsequent steps

set -e

NUM_GPUS=$1

if [ -z "$NUM_GPUS" ]; then
    echo "[ERROR] Missing required argument"
    echo "Usage: $0 <num_gpus>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[ACQUIRE-GPUS] Acquiring $NUM_GPUS GPU(s)"
source "$SCRIPT_DIR/gpu_allocator.sh"
acquire_gpus "$NUM_GPUS"

echo "[ACQUIRE-GPUS] Allocated GPUs: $GPU_DEVICES"
echo "[ACQUIRE-GPUS] GPU allocation details:"
echo "  GPU_DEVICES=$GPU_DEVICES"
echo "  ALLOCATED_GPU_BITMAP=$ALLOCATED_GPU_BITMAP"

# Export to GITHUB_ENV so subsequent steps can use these variables
if [ -n "$GITHUB_ENV" ]; then
    {
        echo "GPU_DEVICES=$GPU_DEVICES"
        echo "ALLOCATED_GPU_BITMAP=$ALLOCATED_GPU_BITMAP"
    } >> "$GITHUB_ENV"
    echo "[ACQUIRE-GPUS] Exported variables to GITHUB_ENV"
else
    echo "[ACQUIRE-GPUS] WARNING: GITHUB_ENV not set, variables not exported"
fi
