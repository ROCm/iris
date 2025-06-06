#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.



# Build the SIF image with the specified ROCm version
IMAGE_NAME="iris.sif"
apptainer build apptainer/images/"$IMAGE_NAME" apptainer/iris.def

echo "Built image: $IMAGE_NAME"