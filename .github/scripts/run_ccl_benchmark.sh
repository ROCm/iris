#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Sweep the CCL collectives for Iris and RCCL and render a comparison.
# Usage: run_ccl_benchmark.sh <ranks_csv> <dtypes_csv> <m_csv> <n_csv> <n_repeat>
#   e.g. run_ccl_benchmark.sh 2,4,8 fp16,bf16,fp8 1024,4096,16384,65536 1024,2048,4096,8192 50
#
# Each bench_*.py carries a "backend" axis with values iris and rccl, so one
# sweep produces both series over identical shapes and the same timing harness.
# RCCL has no fp8, so fp8 points are Iris-only. reduce_scatter is not included:
# iris.ccl.reduce_scatter is (M, N) -> (M, N) whereas torch's
# reduce_scatter_tensor is (W*M, N) -> (M, N), so a side-by-side number would
# compare two different operations.

set -e

RANKS=${1:-"2,4,8"}
DTYPES=${2:-"fp16,bf16,fp8"}
M_VALUES=${3:-"1024,4096,16384,65536"}
N_VALUES=${4:-"1024,2048,4096,8192"}
N_REPEAT=${5:-50}

# These are spliced into the container command below, so only allow the
# characters the values can legitimately contain.
for v in "$RANKS" "$M_VALUES" "$N_VALUES" "$N_REPEAT"; do
    [[ "$v" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "[ERROR] expected comma-separated integers, got '$v'"; exit 1; }
done
[[ "$DTYPES" =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || { echo "[ERROR] bad dtype list '$DTYPES'"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DEVICES=${GPU_DEVICES:-"0,1,2,3,4,5,6,7"}

echo "[CCL-BENCHMARK] ranks=${RANKS} dtypes=${DTYPES} M=${M_VALUES} N=${N_VALUES} n_repeat=${N_REPEAT}"
echo "[CCL-BENCHMARK] Using GPUs: $GPU_DEVICES"

mkdir -p ccl_results/csv

# One process per (collective, M, dtype). The benchmarks do not free symmetric
# buffers between points, so a process's heap use is the sum over everything it
# sweeps: all_to_all at 8 ranks and M=65536 needs ~30 GiB per dtype across the N
# values, so fp16+bf16+fp8 together would overflow even a 64 GiB heap. Rank
# counts can share a process: iris.bench launches each one as its own group.
"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    cd /iris_workspace
    pip install -e .

    for op in all_reduce all_gather all_to_all; do
        echo \"::group::\${op}\"
        for m in ${M_VALUES//,/ }; do
            for dtype in ${DTYPES//,/ }; do
                python benchmark/ccl/bench_\${op}.py \
                    --axis_num_ranks=${RANKS} \
                    --axis_M=\${m} \
                    --axis_N=${N_VALUES} \
                    --axis_dtype=\${dtype} \
                    --heap_size=$((1 << 36)) \
                    --n_repeat=${N_REPEAT} \
                    --benchmark_format=csv \
                    --benchmark_out=ccl_results/csv/\${op}_M\${m}_\${dtype}.csv \
                  || echo \"[WARN] \${op} M=\${m} \${dtype} failed; continuing\"
            done
        done
        echo '::endgroup::'
    done
"

echo "[CCL-BENCHMARK] rendering comparison"
"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    cd /iris_workspace
    pip install matplotlib >/dev/null 2>&1 || true
    python benchmark/ccl/plot_sweep_results.py ccl_results/csv/*.csv \
        --output_dir ccl_results \
        --markdown_out ccl_results/summary.md \
        --title 'Iris vs RCCL'
"

echo "[CCL-BENCHMARK] artifacts:"
ls -la ccl_results/
