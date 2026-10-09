#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Install Iris from this commit and run an external validation test on 2 ranks.
# Usage: run_external_validation.sh <triton|gluon>
#
# The tests live in gists, outside the repo, so they exercise Iris the way an
# external user would: a pip install from git rather than the checkout.

set -e

case "$1" in
    triton)
        TEST_URL="https://gist.githubusercontent.com/mawad-amd/6375dc078e39e256828f379e03310ec7/raw/0827d023eaf8e9755b17cbe8ab06f2ce258e746a/test_iris_distributed.py"
        ;;
    gluon)
        TEST_URL="https://gist.githubusercontent.com/mawad-amd/2666dde8ebe2755eb0c4f2108709fcd5/raw/c5544943e2832c75252160bd9084600bf01a6b06/test_iris_gluon_distributed.py"
        ;;
    *)
        echo "[ERROR] Usage: $0 <triton|gluon>"
        exit 1
        ;;
esac

if [ -z "$GPU_DEVICES" ]; then
    echo "[ERROR] GPU_DEVICES is not set; run this through gpu_task_queue.py"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${GITHUB_REPOSITORY:-"ROCm/iris"}
SHA=${GITHUB_SHA:-"HEAD"}
TEST_FILE=$(basename "$TEST_URL")

"$SCRIPT_DIR/container_exec.sh" --gpus "$GPU_DEVICES" "
    set -e
    cd /iris_workspace
    pip install git+https://github.com/${REPO}.git@${SHA}
    wget -O ${TEST_FILE} ${TEST_URL}
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 ${TEST_FILE}
"

echo "✅ External $1 validation test passed!"
