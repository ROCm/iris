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
    echo "[INFO] Checking Docker access..."
    
    # Check if Docker daemon is accessible
    if ! docker ps &> /dev/null; then
        echo "[INFO] Cannot access Docker, checking if daemon is running..."
        
        # Try to start Docker daemon if it's not running
        if command -v sudo &> /dev/null; then
            DOCKER_STARTED=false
            
            # Try systemd first
            if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
                if ! sudo systemctl is-active --quiet docker; then
                    echo "[INFO] Starting Docker daemon with systemctl..."
                    sudo systemctl start docker
                    DOCKER_STARTED=true
                fi
            # Try service command (SysVinit/Upstart)
            elif command -v service &> /dev/null; then
                if ! sudo service docker status &> /dev/null; then
                    echo "[INFO] Starting Docker daemon with service..."
                    sudo service docker start
                    DOCKER_STARTED=true
                fi
            fi
            
            if [ "$DOCKER_STARTED" = true ]; then
                sleep 2
            fi
            
            # If still can't access, it's a permission issue
            if ! docker ps &> /dev/null; then
                echo "[INFO] Docker daemon running but no permission, adding user to docker group..."
                sudo usermod -aG docker $USER
                echo "[INFO] User added to docker group."
                echo "[ERROR] Please restart the GitHub Actions runner for changes to take effect"
                exit 1
            fi
        else
            echo "[ERROR] Cannot access Docker and sudo is not available"
            exit 1
        fi
    else
        echo "[INFO] Docker is accessible and running"
    fi
    
    IMAGE_NAME="iris-dev-triton-aafec41"
    
    # Check if the triton image exists
    echo "[INFO] Building Docker image..."
    docker build -t "$IMAGE_NAME" -f docker/Dockerfile .
    echo "[INFO] Successfully built Docker image: $IMAGE_NAME"
fi

echo "[INFO] Container build completed successfully with $CONTAINER_RUNTIME"

