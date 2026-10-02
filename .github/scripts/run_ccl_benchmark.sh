#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Sweep the CCL collectives for Iris and RCCL and render a comparison.
# Usage: run_ccl_benchmark.sh <num_ranks> <sizes_csv> <n_repeat>
#
# Each bench_*.py carries a "backend" axis with values iris and rccl, so one
# sweep produces both series over identical shapes and the same timing harness.
# reduce_scatter is not included: iris.ccl.reduce_scatter is (M, N) -> (M, N)
# whereas torch's reduce_scatter_tensor is (W*M, N) -> (M, N), so a side-by-side
# number would compare two different operations.

set -e

NUM_RANKS=${1:-8}
SIZES=${2:-"1024,4096,16384"}
N_REPEAT=${3:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DEVICES=${GPU_DEVICES:-"0,1,2,3,4,5,6,7"}

echo "[CCL-BENCHMARK] ranks=${NUM_RANKS} sizes=${SIZES} n_repeat=${N_REPEAT}"
echo "[CCL-BENCHMARK] Using GPUs: $GPU_DEVICES"

mkdir -p ccl_results

# The default axes sweep M and N over 2^10..2^14 independently, which is 25
# shape combinations per op per backend. Pin M=N to the requested sizes instead:
# a square sweep along the diagonal, which is what the comparison plot shows.
AXIS_ARGS=""
for s in ${SIZES//,/ }; do
    AXIS_ARGS="${AXIS_ARGS} ${s}"
done

"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    cd /iris_workspace
    pip install -e .

    for op in all_reduce all_gather all_to_all; do
        echo \"::group::\${op}\"
        for size in ${AXIS_ARGS}; do
            python benchmark/ccl/bench_\${op}.py \
                --axis_num_ranks=${NUM_RANKS} \
                --axis_M=\${size} \
                --axis_N=\${size} \
                --axis_dtype=fp16 \
                --n_repeat=${N_REPEAT} \
                --benchmark_format=csv \
                --benchmark_out=ccl_results/\${op}_\${size}.csv \
              || echo \"[WARN] \${op} at size \${size} failed; continuing\"
        done
        echo '::endgroup::'
    done
"

echo "[CCL-BENCHMARK] rendering comparison"
"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    cd /iris_workspace
    pip install matplotlib >/dev/null 2>&1 || true
    python benchmark/ccl/plot_sweep_results.py ccl_results/*.csv \
        --output ccl_results/iris_vs_rccl.png \
        --markdown_out ccl_results/summary.md \
        --title 'CCL Benchmark: Iris vs RCCL (${NUM_RANKS} ranks)'
"

echo "[CCL-BENCHMARK] artifacts:"
ls -la ccl_results/
