#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Runs inside the existing Iris GPU container. Both revisions use this harness
# and the same installed dependencies, allocated GPUs and fixed configuration.
set -euo pipefail
cd /iris_workspace
pip install -e .
python -m pytest tests/bench/test_workspace_benchmark_report.py -q

base_sha=$(cat .workspace_benchmark/base_sha)
head_sha=$(cat .workspace_benchmark/head_sha)
for ranks in 2 4 8; do
    for label in base head; do
        if [ "$label" = base ]; then
            source_dir=/iris_workspace/.workspace_benchmark/base
            source_sha=$base_sha
        else
            source_dir=/iris_workspace
            source_sha=$head_sha
        fi
        timeout --signal=TERM --kill-after=30s 15m \
            torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
            --nnodes=1 --nproc_per_node="$ranks" \
            benchmark/ops/bench_matmul_all_reduce_workspace.py \
            --source-dir "$source_dir" --commit "$source_sha" --label "$label" \
            --output ".workspace_benchmark/results/${label}_${ranks}.json"
    done
done
python benchmark/ops/summarize_workspace_benchmark.py .workspace_benchmark/results \
    --output .workspace_benchmark/results/summary.md
