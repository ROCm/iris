#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Run new example scripts (numbered 24+) directly with torchrun.
# Usage: run_new_examples.sh <num_ranks> [install_method]
#   num_ranks: number of GPU ranks (2, 4, or 8)
#   install_method: "git", "editable", or "install" (default: "editable")

set -e

NUM_RANKS=${1:-2}
INSTALL_METHOD=${2:-"editable"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU_DEVICES should be set by the workflow-level acquire_gpus.sh step
GPU_ARG=""
if [ -n "$GPU_DEVICES" ]; then
    GPU_ARG="--gpus $GPU_DEVICES"
fi

# Build install command based on method
if [ "$INSTALL_METHOD" = "git" ]; then
    REPO=${GITHUB_REPOSITORY:-"ROCm/iris"}
    SHA=${GITHUB_SHA:-"HEAD"}
    INSTALL_CMD="pip install git+https://github.com/${REPO}.git@${SHA}"
elif [ "$INSTALL_METHOD" = "editable" ]; then
    INSTALL_CMD="pip install -e ."
elif [ "$INSTALL_METHOD" = "install" ]; then
    INSTALL_CMD="pip install ."
else
    echo "[ERROR] Invalid install_method: $INSTALL_METHOD"
    exit 1
fi

EXIT_CODE=0
# shellcheck disable=SC2086
"$SCRIPT_DIR/container_exec.sh" $GPU_ARG "
    set -e

    echo \"Installing iris using method: $INSTALL_METHOD\"
    $INSTALL_CMD

    # Run new examples (numbered 24 and above)
    for example_file in examples/2[4-9]_*/example.py examples/3[0-9]_*/example.py; do
        if [ -f \"\$example_file\" ]; then
            echo \"Running: \$example_file with $NUM_RANKS ranks\"
            torchrun --nproc_per_node=$NUM_RANKS --standalone \"\$example_file\"
        fi
    done
" || { EXIT_CODE=$?; }

exit $EXIT_CODE
