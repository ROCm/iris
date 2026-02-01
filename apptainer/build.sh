#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

set -e

# Build the SIF image with the specified ROCm version
IMAGE_NAME="iris.sif"
IMAGE_DIR="apptainer/images"
IMAGE_PATH="$IMAGE_DIR/$IMAGE_NAME"
DEF_FILE="apptainer/iris.def"
CHECKSUM_FILE="$IMAGE_DIR/${IMAGE_NAME}.checksum"

# Create images directory if it doesn't exist
mkdir -p "$IMAGE_DIR"

# Verify def file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "Error: Definition file $DEF_FILE not found"
    exit 1
fi

# Calculate checksum of the def file
NEW_CHECKSUM=$(sha256sum "$DEF_FILE" | awk '{print $1}')

# Check if image exists and has a valid checksum
REBUILD_NEEDED=true
if [ -f "$IMAGE_PATH" ] && [ -f "$CHECKSUM_FILE" ]; then
    OLD_CHECKSUM=$(head -n1 "$CHECKSUM_FILE" 2>/dev/null)
    # Validate checksum format (64 hex characters for SHA256)
    if [[ "$OLD_CHECKSUM" =~ ^[a-f0-9]{64}$ ]] && [ "$OLD_CHECKSUM" = "$NEW_CHECKSUM" ]; then
        echo "Def file unchanged (checksum: $NEW_CHECKSUM)"
        echo "Skipping rebuild of $IMAGE_NAME"
        REBUILD_NEEDED=false
    else
        echo "Def file changed (old: ${OLD_CHECKSUM:-<invalid>}, new: $NEW_CHECKSUM)"
        echo "Rebuilding $IMAGE_NAME"
    fi
else
    echo "Image or checksum not found, building $IMAGE_NAME"
fi

# Build the image if needed
if [ "$REBUILD_NEEDED" = true ]; then
    if apptainer build --force "$IMAGE_PATH" "$DEF_FILE"; then
        # Store the checksum only if build succeeded
        echo "$NEW_CHECKSUM" > "$CHECKSUM_FILE"
        echo "Built image: $IMAGE_NAME"
        echo "Checksum saved: $NEW_CHECKSUM"
    else
        echo "Error: Apptainer build failed"
        exit 1
    fi
fi