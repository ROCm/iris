#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Run Iris tests in a container with automatic GPU allocation
# Usage: run_tests.sh <test_dir> <num_ranks> [gpu_devices] [install_method]
#   test_dir: subdirectory under tests/ (e.g., examples, unittests, ccl)
#   num_ranks: number of GPU ranks (1, 2, 4, or 8)
#   gpu_devices: comma-separated GPU device IDs (optional, if not provided will use allocator)
#   install_method: pip install method - "git", "editable", or "install" (optional, default: "editable")
#     - "git": pip install git+https://github.com/${{ github.repository }}.git@${{ github.sha }}
#     - "editable": pip install -e .
#     - "install": pip install .

set -e

TEST_DIR=$1
NUM_RANKS=$2
GPU_DEVICES=${3:-${GPU_DEVICES:-""}}
INSTALL_METHOD=${4:-"editable"}

if [ -z "$TEST_DIR" ] || [ -z "$NUM_RANKS" ]; then
    echo "[ERROR] Missing required arguments"
    echo "Usage: $0 <test_dir> <num_ranks> [gpu_devices] [install_method]"
    echo "  test_dir: examples, unittests, x or ccl"
    echo "  num_ranks: 1, 2, 4, or 8"
    echo "  install_method: git, editable, or install (default: editable)"
    exit 1
fi

# Validate test directory
if [ ! -d "tests/$TEST_DIR" ]; then
    echo "[ERROR] Test directory tests/$TEST_DIR does not exist"
    exit 1
fi

# Validate install method
if [ "$INSTALL_METHOD" != "git" ] && [ "$INSTALL_METHOD" != "editable" ] && [ "$INSTALL_METHOD" != "install" ]; then
    echo "[ERROR] Invalid install_method: $INSTALL_METHOD"
    echo "  Must be one of: git, editable, install"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU_DEVICES should be provided by workflow-level acquire_gpus.sh step
# or via command-line argument for backward compatibility
if [ -z "$GPU_DEVICES" ]; then
    echo "[RUN-TESTS] WARNING: No GPUs allocated. GPU_DEVICES not set."
    echo "[RUN-TESTS] Tests may fail if they require GPUs."
fi

# Build GPU argument
GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# Build install command based on method
INSTALL_CMD=""
if [ "$INSTALL_METHOD" = "git" ]; then
    # For git install, we need the repository and SHA from environment or use defaults
    REPO=${GITHUB_REPOSITORY:-"ROCm/iris"}
    SHA=${GITHUB_SHA:-"HEAD"}
    INSTALL_CMD="pip install git+https://github.com/${REPO}.git@${SHA}"
elif [ "$INSTALL_METHOD" = "editable" ]; then
    INSTALL_CMD="pip install -e ."
elif [ "$INSTALL_METHOD" = "install" ]; then
    INSTALL_CMD="pip install ."
fi

# Run tests in container
EXIT_CODE=0
# shellcheck disable=SC2086
"$SCRIPT_DIR/container_exec.sh" $GPU_ARG "
    set -e
    
    echo \"Installing iris using method: $INSTALL_METHOD\"
    $INSTALL_CMD
    
    # Run tests in the specified directory
    for test_file in tests/$TEST_DIR/test_*.py; do
        if [ -f \"\$test_file\" ]; then
            echo \"Testing: \$test_file with $NUM_RANKS ranks (install: $INSTALL_METHOD)\"
            torchrun --nproc_per_node=$NUM_RANKS --standalone tests/run_tests_distributed.py \"\$test_file\" -v --tb=short --durations=10
        fi
    done
" || { EXIT_CODE=$?; }

# GPU cleanup is now handled by workflow-level release_gpus.sh step
exit $EXIT_CODE