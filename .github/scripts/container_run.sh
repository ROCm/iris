#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Universal container run script that works with Apptainer or Docker

set -e

# Check which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNTIME="apptainer"
    echo "[INFO] Using Apptainer"
elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "[INFO] Using Docker"
else
    echo "[ERROR] Neither Apptainer nor Docker is available"
    echo "[ERROR] Please install either Apptainer or Docker to continue"
    exit 1
fi

# Run based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    echo "[INFO] Running with Apptainer..."
    bash apptainer/run.sh "$@"
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    echo "[INFO] Running with Docker..."
    IMAGE_NAME=${1:-"iris-dev-triton-aafec41"}
    WORKSPACE_DIR=${2:-"$(pwd)"}
    bash docker/run.sh "$IMAGE_NAME" "$WORKSPACE_DIR"
fi

