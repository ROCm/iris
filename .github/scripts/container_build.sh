#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Universal container build script that works with Apptainer or Docker

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

# Build based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    echo "[INFO] Building with Apptainer..."
    
    # Create persistent Apptainer directory
    mkdir -p ~/apptainer
    
    # Build Apptainer image from definition file (only if it doesn't exist)
    if [ ! -f ~/apptainer/iris-dev.sif ]; then
        echo "[INFO] Building new Apptainer image..."
        apptainer build ~/apptainer/iris-dev.sif apptainer/iris.def
    else
        echo "[INFO] Using existing Apptainer image at ~/apptainer/iris-dev.sif"
    fi
    
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    echo "[INFO] Checking Docker images..."
    # Use GitHub variable if set, otherwise default to iris-dev
    IMAGE_NAME=${DOCKER_IMAGE_NAME:-"iris-dev"}
    
    # Check if the image exists
    if docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "[INFO] Using existing Docker image: $IMAGE_NAME"
    else
        echo "[WARNING] Docker image $IMAGE_NAME not found"
        echo "[INFO] Please build it using: ./build_triton_image.sh"
        echo "[INFO] Or pull it if available from registry"
    fi
fi

echo "[INFO] Container build completed successfully with $CONTAINER_RUNTIME"

