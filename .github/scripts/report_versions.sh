#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Collect environment versions from inside the container and write
# a GitHub Actions job summary table. Version collection logic lives
# in collect_versions.py; this script handles container exec + summary.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# Run collect_versions.py inside the container, extract JSON line
# shellcheck disable=SC2086
RAW=$("$SCRIPT_DIR/container_exec.sh" $GPU_ARG "python3 /iris_workspace/.github/scripts/collect_versions.py")
VERSION_JSON=$(echo "$RAW" | grep '^{' | tail -1)

if [ -z "$VERSION_JSON" ]; then
    echo "[WARN] Failed to collect version info"
    exit 0
fi

# Format and print with Python on the host (available on all runners)
python3 -c "
import json, sys, os

d = json.loads(sys.argv[1])

def g(k):
    return str(d.get(k, 'unknown'))

gpu = f\"{g('gpu_name')} ({g('gpu_arch')}) x {g('gpu_count')}\"

print('============================================')
print('  Iris CI - Environment Report')
print('============================================')
for label, key in [('Driver', 'driver'), ('ROCm', 'rocm'), ('Python', 'python'),
                    ('PyTorch', 'torch'), ('HIP', 'hip'), ('Triton', 'triton')]:
    print(f'  {label:9s} {g(key)}')
print(f'  {\"GPU\":9s} {gpu}')
print(f'  {\"Kernel\":9s} {g(\"kernel\")}')
print('============================================')

summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
if summary_path:
    with open(summary_path, 'a') as f:
        f.write('### Environment\n\n')
        f.write('| Component | Version |\n')
        f.write('|-----------|--------|\n')
        for label, key in [('Driver', 'driver'), ('ROCm', 'rocm'), ('Python', 'python'),
                            ('PyTorch', 'torch'), ('HIP', 'hip'), ('Triton', 'triton')]:
            f.write(f'| {label} | {g(key)} |\n')
        f.write(f'| GPU | {gpu} |\n')
        f.write(f'| Kernel | {g(\"kernel\")} |\n\n')
" "$VERSION_JSON"
