#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Run Iris tests in a container
# Usage: run_tests.sh <num_ranks> [gpu_devices]

set -e

NUM_RANKS=$1
GPU_DEVICES=${2:-""}

if [ -z "$NUM_RANKS" ]; then
    echo "[ERROR] NUM_RANKS not provided"
    echo "Usage: $0 <num_ranks> [gpu_devices]"
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
    
    # Run examples tests
    for test_file in tests/examples/test_*.py; do
        echo \"Testing: \$test_file with $NUM_RANKS ranks\"
        python tests/run_tests_distributed.py --num_ranks $NUM_RANKS \"\$test_file\" -v --tb=short --durations=10
    done
    
    # Run unit tests
    for test_file in tests/unittests/test_*.py; do
        echo \"Testing: \$test_file with $NUM_RANKS ranks\"
        python tests/run_tests_distributed.py --num_ranks $NUM_RANKS \"\$test_file\" -v --tb=short --durations=10
    done

    # Run ccl tests
    for test_file in tests/ccl/test_*.py; do
        echo \"Testing: \$test_file with $NUM_RANKS ranks\"
        python tests/run_tests_distributed.py --num_ranks $NUM_RANKS \"\$test_file\" -v --tb=short --durations=10
    done
"

