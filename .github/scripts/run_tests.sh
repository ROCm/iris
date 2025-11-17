#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Run Iris tests in a container
# Usage: run_tests.sh <test_dir> <num_ranks> [gpu_devices]
#   test_dir: subdirectory under tests/ (e.g., examples, unittests, ccl)
#   num_ranks: number of GPU ranks (1, 2, 4, or 8)
#   gpu_devices: comma-separated GPU device IDs (optional)

set -e

TEST_DIR=$1
NUM_RANKS=$2
GPU_DEVICES=${3:-""}

if [ -z "$TEST_DIR" ] || [ -z "$NUM_RANKS" ]; then
    echo "[ERROR] Missing required arguments"
    echo "Usage: $0 <test_dir> <num_ranks> [gpu_devices]"
    echo "  test_dir: examples, unittests, or ccl"
    echo "  num_ranks: 1, 2, 4, or 8"
    exit 1
fi

# Validate test directory
if [ ! -d "tests/$TEST_DIR" ]; then
    echo "[ERROR] Test directory tests/$TEST_DIR does not exist"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build GPU argument if provided
GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# Run tests in container
"$SCRIPT_DIR/container_exec.sh" $GPU_ARG "
    set -e
    pip install -e .
    
    # Run tests in the specified directory
    for test_file in tests/$TEST_DIR/test_*.py; do
        if [ -f \"\$test_file\" ]; then
            echo \"Testing: \$test_file with $NUM_RANKS ranks\"
            python tests/run_tests_distributed.py --num_ranks $NUM_RANKS \"\$test_file\" -v --tb=short --durations=10
        fi
    done
"

