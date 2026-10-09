#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Release GPUs for CI workflows - to be called as a workflow step with if: always()
# Usage: release_gpus.sh
#
# Returns whatever GPUs gpu_task_queue.py still holds if it was killed mid-run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# gpu_task_queue.py records the GPUs it still holds here. It normally returns
# them itself and leaves 0; anything else means it was killed mid-run.
QUEUE_HELD_FILE="${GPU_QUEUE_HELD_FILE:-${RUNNER_TEMP:-/tmp}/iris_gpu_queue_held}"
if [ -z "$ALLOCATED_GPU_BITMAP" ] && [ -s "$QUEUE_HELD_FILE" ]; then
    ALLOCATED_GPU_BITMAP=$(cat "$QUEUE_HELD_FILE")
    rm -f "$QUEUE_HELD_FILE"
    if [ "$ALLOCATED_GPU_BITMAP" = "0" ]; then
        unset ALLOCATED_GPU_BITMAP
    else
        export ALLOCATED_GPU_BITMAP
    fi
fi

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
