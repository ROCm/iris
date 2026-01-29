#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Run performance benchmark in a container
# Usage: run_perf_benchmark.sh <example_path> <tflops_threshold> <benchmark_args...>

set -e

EXAMPLE_PATH=$1
TFLOPS_THRESHOLD=$2
shift 2
BENCHMARK_ARGS="$@"

if [ -z "$EXAMPLE_PATH" ] || [ -z "$TFLOPS_THRESHOLD" ]; then
    echo "[ERROR] Missing required arguments"
    echo "Usage: $0 <example_path> <tflops_threshold> <benchmark_args...>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run benchmark in container
"$SCRIPT_DIR/container_exec.sh" --gpus "0,1,2,3,4,5,6,7" "
    set -e
    
    # Install tritonBLAS (required dependency)
    echo \"Installing tritonBLAS...\"
    if [ ! -d \"/tmp/tritonBLAS\" ]; then
        cd /tmp && git clone https://github.com/ROCm/tritonBLAS.git 2>&1 | tail -3
    fi
    if [ -d \"/tmp/tritonBLAS\" ]; then
        cd /tmp/tritonBLAS
        git checkout 47768c93acb7f89511d797964b84544c30ab81ad 2>&1 | tail -2
        pip install -e . 2>&1 | tail -3
    else
        echo \"Warning: Could not clone tritonBLAS, trying pip install from git...\"
        pip install git+https://github.com/ROCm/tritonBLAS.git@47768c93acb7f89511d797964b84544c30ab81ad 2>&1 | tail -3
    fi
    
    cd /iris_workspace
    pip install -e .
    python examples/${EXAMPLE_PATH}/benchmark.py \
        --benchmark \
        --validate \
        -r 8 \
        ${BENCHMARK_ARGS} \
        --output_file perf_result.json
"

# Validate performance (runs outside container)
echo "Validating performance results..."

SUCCESS=$(jq -r '.success' perf_result.json)
if [ "$SUCCESS" != "true" ]; then
    echo "[ERROR] Benchmark failed (success: $SUCCESS)"
    jq '.' perf_result.json
    exit 1
fi

TFLOPS=$(jq -r '.tflops' perf_result.json)

if [ -z "$TFLOPS" ] || [ "$TFLOPS" = "null" ]; then
    echo "[ERROR] Failed to extract tflops from benchmark output"
    jq '.' perf_result.json
    exit 1
fi

echo "[INFO] Achieved TFLOPs: $TFLOPS"

# Convert to integer for comparison
TFLOPS_INT=${TFLOPS%.*}
if (( TFLOPS_INT < TFLOPS_THRESHOLD )); then
    echo "[ERROR] Performance regression detected! TFLOPs ($TFLOPS) is below threshold ($TFLOPS_THRESHOLD)"
    jq '.' perf_result.json
    exit 1
fi

echo "✅ Performance test passed! TFLOPs: $TFLOPS (threshold: >$TFLOPS_THRESHOLD)"

