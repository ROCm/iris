#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Universal container exec script - thin wrapper that executes commands in either Apptainer or Docker
# Usage: container_exec.sh [--gpus GPUS] [--image IMAGE] <command>


# Parse optional arguments
GPU_DEVICES=""
CUSTOM_IMAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --image)
            CUSTOM_IMAGE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Remaining args are the command
COMMAND="$@"
if [ -z "$COMMAND" ]; then
    echo "[ERROR] No command provided" >&2
    echo "Usage: $0 [--gpus GPUS] [--image IMAGE] <command>" >&2
    exit 1
fi

# Check which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNTIME="apptainer"
    echo "[INFO] Using Apptainer"
elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "[INFO] Using Docker"
else
    echo "[ERROR] Neither Apptainer nor Docker is available" >&2
    exit 1
fi

# Execute based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    # Find image
    if [ -n "$CUSTOM_IMAGE" ]; then
        IMAGE="$CUSTOM_IMAGE"
    elif [ -f ~/apptainer/iris-dev.sif ]; then
        IMAGE=~/apptainer/iris-dev.sif
    elif [ -f apptainer/images/iris.sif ]; then
        IMAGE="apptainer/images/iris.sif"
    else
        echo "[ERROR] Apptainer image not found" >&2
        exit 1
    fi
    
    # Create temporary overlay in workspace with unique name based on PID and timestamp
    OVERLAY="./iris_overlay_$$_$(date +%s%N).img"
    if ! apptainer overlay create --size 16384 --create-dir /var/cache/iris "${OVERLAY}" > /dev/null 2>&1; then
        echo "[ERROR] Failed to create Apptainer overlay"
        exit 1
    fi
    
    # Build exec command
    EXEC_CMD="apptainer exec --overlay ${OVERLAY} --no-home --cleanenv"
    
    # Set required RCCL environment variable for ROCm
    EXEC_CMD="$EXEC_CMD --env HSA_NO_SCRATCH_RECLAIM=1"
    
    # Add GPU selection if specified
    if [ -n "$GPU_DEVICES" ]; then
        EXEC_CMD="$EXEC_CMD --env HIP_VISIBLE_DEVICES=${GPU_DEVICES}"
    fi
    
    # Add standard flags
    EXEC_CMD="$EXEC_CMD --bind ${PWD}:/iris_workspace --cwd /iris_workspace"
    
    # Execute with cleanup of overlay file
    EXIT_CODE=0
    $EXEC_CMD "$IMAGE" bash -c "$COMMAND" || EXIT_CODE=$?
    # Clean up overlay file (always cleanup, even on failure)
    rm -f "${OVERLAY}" 2>/dev/null || true
    exit $EXIT_CODE
    
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    # Use custom image if provided, otherwise use GitHub variable or default
    # GitHub Actions sets DOCKER_IMAGE_NAME, locally defaults to iris-dev
    IMAGE_NAME=${CUSTOM_IMAGE:-${DOCKER_IMAGE_NAME:-"iris-dev"}}
    
    if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "[ERROR] Docker image $IMAGE_NAME not found" >&2
        exit 1
    fi
    
    # Build run command with proper GPU access
    # Get video and render group IDs from host
    VIDEO_GID=$(getent group video | cut -d: -f3)
    RENDER_GID=$(getent group render | cut -d: -f3)
    
    RUN_CMD="docker run --rm --network=host --device=/dev/kfd --device=/dev/dri"
    RUN_CMD="$RUN_CMD --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    RUN_CMD="$RUN_CMD -v ${PWD}:/iris_workspace -w /iris_workspace"
    RUN_CMD="$RUN_CMD --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864"
    RUN_CMD="$RUN_CMD --user $(id -u):$(id -g)"
    
    # Add video and render groups for GPU access
    if [ -n "$VIDEO_GID" ]; then
        RUN_CMD="$RUN_CMD --group-add $VIDEO_GID"
    fi
    if [ -n "$RENDER_GID" ]; then
        RUN_CMD="$RUN_CMD --group-add $RENDER_GID"
    fi
    
    RUN_CMD="$RUN_CMD -e HOME=/iris_workspace"
    RUN_CMD="$RUN_CMD --entrypoint bash"
    
    # Set required RCCL environment variable for ROCm
    RUN_CMD="$RUN_CMD -e HSA_NO_SCRATCH_RECLAIM=1"
    
    # Add GPU selection if specified
    if [ -n "$GPU_DEVICES" ]; then
        RUN_CMD="$RUN_CMD -e HIP_VISIBLE_DEVICES=${GPU_DEVICES}"
    fi
    
    # Execute and capture exit code
    EXIT_CODE=0
    $RUN_CMD "$IMAGE_NAME" -c "$COMMAND" || EXIT_CODE=$?
    exit $EXIT_CODE
fi

