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
    CACHE_DIR="${HOME}/iris-apptainer-images/${DEF_CHECKSUM}"
    mkdir -p "$CACHE_DIR"
    
    # Define paths. $HOME is shared across the runners, so every job in a run
    # reads and writes this one directory.
    IMAGE_PATH="$CACHE_DIR/iris-dev.sif"
    CHECKSUM_FILE="$CACHE_DIR/iris-dev.sif.checksum"
    LOCK_FILE="$CACHE_DIR/.build.lock"
    
    image_is_current() {
        [ -f "$IMAGE_PATH" ] && [ -f "$CHECKSUM_FILE" ] || return 1
        local old
        old=$(head -n1 "$CHECKSUM_FILE" 2>/dev/null)
        # Validate checksum format (64 hex characters for SHA256)
        [[ "$old" =~ ^[a-f0-9]{64}$ ]] && [ "$old" = "$DEF_CHECKSUM" ]
    }
    
    # Build to a private path and rename into place. Renaming is atomic and
    # leaves the inode alone, so a job already executing the old image keeps
    # running against it; `apptainer build --force` straight to IMAGE_PATH would
    # truncate the file out from under it.
    build_image() {
        local tmp
        tmp=$(mktemp -u "$CACHE_DIR/.iris-dev.XXXXXXXX.sif")
        if apptainer build --force "$tmp" "$DEF_FILE"; then
            mv -f "$tmp" "$IMAGE_PATH"
            # Store the checksum only if build succeeded
            echo "$DEF_CHECKSUM" > "$CHECKSUM_FILE"
            echo "[INFO] Built image: $IMAGE_PATH"
            echo "[INFO] Checksum saved: $DEF_CHECKSUM"
        else
            rm -f "$tmp"
            echo "[ERROR] Apptainer build failed"
            exit 1
        fi
    }
    
    if image_is_current; then
        echo "[INFO] Def file unchanged (checksum: $DEF_CHECKSUM)"
        echo "[INFO] Skipping rebuild, using existing image at $IMAGE_PATH"
    else
        echo "[INFO] Image or checksum not found, building new Apptainer image..."
        # Serialize builders. Without this, jobs that start together all see no
        # image and all build concurrently into the same path -- observed, with
        # two runners building at once. The re-check inside the lock is the
        # point: whoever waits usually finds the image already built and skips a
        # redundant half-hour build.
        if command -v flock > /dev/null 2>&1; then
            exec 9> "$LOCK_FILE"
            if ! flock -w 5400 9; then
                echo "[ERROR] Timed out waiting for the image build lock"
                exit 1
            fi
            if image_is_current; then
                echo "[INFO] Another job built it while we waited; using $IMAGE_PATH"
            else
                build_image
            fi
            exec 9>&-
        else
            echo "[WARN] flock not available; building without a lock"
            build_image
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
        echo "[INFO] Docker image $IMAGE_NAME not found, building..."
        REPO_ROOT="$(dirname "$(realpath "$0")")/../.."
        # Build from the repo root, not docker/, so the Dockerfile can COPY in
        # .github/scripts/install_rocshmem.sh -- the same installer the Apptainer
        # def file pulls in via %files. A docker/-only context cannot see it.
        if docker build -t "$IMAGE_NAME" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"; then
            echo "[INFO] Built Docker image: $IMAGE_NAME"
        else
            echo "[ERROR] Docker build failed"
            exit 1
        fi
    fi
fi

echo "[INFO] Container build completed successfully with $CONTAINER_RUNTIME"

