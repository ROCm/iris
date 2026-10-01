#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Builds the iris-dev-custom Docker image from source using TheRock.
# Always uses Dockerfile.dev.
#
# If TheRock/ or rocm-systems/ directories already exist in the current
# folder, they are reused. Otherwise they are cloned from the given URLs.
#
# Usage:
#   ./build-dev.sh [OPTIONS]
#
# Options:
#   -n, --name IMAGE_NAME         Docker image name (default: iris-dev-custom)
#   -g, --gpu-family FAMILY       THEROCK_AMDGPU_FAMILIES value (default: gfx110X-all)
#   -t, --therock-repo URL        TheRock git repo URL
#                                 (default: https://github.com/ROCm/TheRock.git)
#   -r, --rocm-systems-repo URL   Custom rocm-systems git repo URL
#                                 (replaces TheRock's built-in rocm-systems)
#   -h, --help                    Show this help message
#
# Examples:
#   # Default: uses existing dirs or clones, gfx110X-all
#   ./build-dev.sh
#
#   # Custom GPU family
#   ./build-dev.sh -g gfx942
#
#   # Custom rocm-systems repo (clones if rocm-systems/ doesn't exist)
#   ./build-dev.sh -r git@github.com:ROCm/rocm-systems.git
#
#   # Everything custom
#   ./build-dev.sh -g gfx942 \
#       -t git@github.com:ROCm/TheRock.git \
#       -r git@github.com:ROCm/rocm-systems.git

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
DOCKERFILE="Dockerfile.dev"

# Defaults
IMAGE_NAME="iris-dev-custom"
AMDGPU_FAMILIES="gfx110X-all"
THEROCK_REPO="git@github.com:ROCm/TheRock.git"
ROCM_SYSTEMS_REPO=""
SWAP_ROCM_SYSTEMS="false"

usage() {
    sed -n '2,/^$/s/^#//p' "$0"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -g|--gpu-family)
            AMDGPU_FAMILIES="$2"
            shift 2
            ;;
        -t|--therock-repo)
            THEROCK_REPO="$2"
            shift 2
            ;;
        -r|--rocm-systems-repo)
            ROCM_SYSTEMS_REPO="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

pushd "$SCRIPT_DIR" > /dev/null

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Error: $DOCKERFILE not found in $SCRIPT_DIR"
    exit 1
fi

# --- TheRock ---
if [[ -d "TheRock" ]]; then
    echo "==> Using existing TheRock directory."
else
    echo "==> Cloning TheRock from: $THEROCK_REPO"
    git clone "$THEROCK_REPO" TheRock
fi

# --- Custom rocm-systems ---
# If -r is passed, clone from that URL (unless rocm-systems/ already exists)
# If -r is not passed but rocm-systems/ exists locally, use it
# The custom rocm-systems is staged as TheRock/custom-rocm-systems and
# swapped inside the Dockerfile AFTER fetch_sources.py
if [[ -n "$ROCM_SYSTEMS_REPO" ]]; then
    if [[ -d "rocm-systems" ]]; then
        echo "==> Using existing rocm-systems directory."
    else
        echo "==> Cloning custom rocm-systems from: $ROCM_SYSTEMS_REPO"
        git clone --recursive "$ROCM_SYSTEMS_REPO" rocm-systems
    fi
    echo "==> Staging custom rocm-systems into TheRock/custom-rocm-systems"
    rm -rf TheRock/custom-rocm-systems
    cp -a rocm-systems TheRock/custom-rocm-systems
    SWAP_ROCM_SYSTEMS="true"
elif [[ -d "rocm-systems" ]]; then
    echo "==> Found existing rocm-systems directory, using it."
    echo "==> Staging custom rocm-systems into TheRock/custom-rocm-systems"
    rm -rf TheRock/custom-rocm-systems
    cp -a rocm-systems TheRock/custom-rocm-systems
    SWAP_ROCM_SYSTEMS="true"
else
    echo "==> No custom rocm-systems, using TheRock default."
fi

echo ""
echo "========================================="
echo "  Building Docker image"
echo "========================================="
echo "  Image name:      $IMAGE_NAME"
echo "  Dockerfile:      $DOCKERFILE"
echo "  GPU family:      $AMDGPU_FAMILIES"
echo "  TheRock:         $THEROCK_REPO"
if [[ "$SWAP_ROCM_SYSTEMS" == "true" ]]; then
    echo "  rocm-systems:    custom (from ./rocm-systems)"
else
    echo "  rocm-systems:    (TheRock default)"
fi
echo "========================================="
echo ""

docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" \
    --build-arg AMDGPU_FAMILIES="$AMDGPU_FAMILIES" \
    --build-arg SWAP_ROCM_SYSTEMS="$SWAP_ROCM_SYSTEMS" \
    .

popd > /dev/null
