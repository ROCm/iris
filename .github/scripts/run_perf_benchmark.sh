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

# Use GPU_DEVICES from environment if set, otherwise default to all 8 GPUs
GPU_DEVICES=${GPU_DEVICES:-"0,1,2,3,4,5,6,7"}
echo "[PERF-BENCHMARK] Using GPUs: $GPU_DEVICES"

# Run benchmark in container
"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    
    cd /iris_workspace
    pip install -e .
    torchrun --nproc_per_node=8 examples/${EXAMPLE_PATH}/benchmark.py \
        --benchmark \
        --validate \
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

