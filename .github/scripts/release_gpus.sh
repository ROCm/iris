#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Release GPUs for CI workflows - to be called as a workflow step with if: always()
# Usage: release_gpus.sh
#
# Reads GPU allocation details from environment variables set by acquire_gpus.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we have GPU allocation details
if [ -z "$GPU_DEVICES" ] && [ -z "$ALLOCATED_GPU_BITMAP" ]; then
    echo "[RELEASE-GPUS] No GPU allocation found, nothing to release"
    exit 0
fi

echo "[RELEASE-GPUS] Releasing GPUs"
echo "[RELEASE-GPUS] GPU allocation details:"
echo "  GPU_DEVICES=$GPU_DEVICES"
echo "  ALLOCATED_GPU_BITMAP=$ALLOCATED_GPU_BITMAP"

source "$SCRIPT_DIR/gpu_allocator.sh"
release_gpus

echo "[RELEASE-GPUS] GPUs released successfully"
