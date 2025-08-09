#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

SCRIPT_DIR=$(dirname "$(realpath "$0")")

IMAGE_NAME=${1:-"iris-dev"}

pushd "$SCRIPT_DIR" > /dev/null

docker build -t $IMAGE_NAME .

popd > /dev/null