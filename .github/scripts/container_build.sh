#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Check /dev/shm size
shm_size_gb=$(df -k /dev/shm | tail -1 | awk '{print int($2/1024/1024)}')
if [ "${shm_size_gb:-0}" -lt 64 ]; then
    echo "❌ ERROR: /dev/shm is too small (${shm_size_gb}GB < 64GB)"
    echo "Fix: mount -o remount,size=64G /dev/shm"
    exit 1
fi
echo "✅ /dev/shm size OK (${shm_size_gb}GB)"

# Build based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    echo "[INFO] Building with Apptainer..."
    
    # Verify def file exists
    DEF_FILE=apptainer/iris.def
    if [ ! -f "$DEF_FILE" ]; then
        echo "[ERROR] Definition file $DEF_FILE not found"
        exit 1
    fi
    
    # Calculate checksum of the def file to use as subdirectory name
    DEF_CHECKSUM=$(sha256sum "$DEF_FILE" | awk '{print $1}')
    
    # Create persistent Apptainer directory with checksum subdirectory
    mkdir -p "${HOME}/iris-apptainer-images/${DEF_CHECKSUM}"
    
    # Define paths
    IMAGE_PATH="${HOME}/iris-apptainer-images/${DEF_CHECKSUM}/iris-dev.sif"
    CHECKSUM_FILE="${HOME}/iris-apptainer-images/${DEF_CHECKSUM}/iris-dev.sif.checksum"
    
    # Check if image exists and has a valid checksum
    REBUILD_NEEDED=true
    if [ -f "$IMAGE_PATH" ] && [ -f "$CHECKSUM_FILE" ]; then
        OLD_CHECKSUM=$(head -n1 "$CHECKSUM_FILE" 2>/dev/null)
        # Validate checksum format (64 hex characters for SHA256)
        if [[ "$OLD_CHECKSUM" =~ ^[a-f0-9]{64}$ ]] && [ "$OLD_CHECKSUM" = "$DEF_CHECKSUM" ]; then
            echo "[INFO] Def file unchanged (checksum: $DEF_CHECKSUM)"
            echo "[INFO] Skipping rebuild, using existing image at $IMAGE_PATH"
            REBUILD_NEEDED=false
        else
            echo "[INFO] Def file changed (old: ${OLD_CHECKSUM:-<invalid>}, new: $DEF_CHECKSUM)"
            echo "[INFO] Rebuilding Apptainer image..."
        fi
    else
        echo "[INFO] Image or checksum not found, building new Apptainer image..."
    fi
    
    # Build the image if needed
    if [ "$REBUILD_NEEDED" = true ]; then
        if apptainer build --force "$IMAGE_PATH" "$DEF_FILE"; then
            # Store the checksum only if build succeeded
            echo "$DEF_CHECKSUM" > "$CHECKSUM_FILE"
            echo "[INFO] Built image: $IMAGE_PATH"
            echo "[INFO] Checksum saved: $DEF_CHECKSUM"
        else
            echo "[ERROR] Apptainer build failed"
            exit 1
        fi
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

