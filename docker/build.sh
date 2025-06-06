#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


IMAGE_NAME=${1:-"iris-dev"}

docker build -t $IMAGE_NAME .