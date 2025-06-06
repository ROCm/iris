#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


ROCM_VERSION=${1:-6.2.3}

# Set image name
IMAGE_NAME="rocshmem_rocm_${ROCM_VERSION}.sif"
IMAGE_PATH="apptainer/images/${IMAGE_NAME}"

# Check if the image exists
if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Error: Image $IMAGE_PATH does not exist."
  exit 1
fi

export ROCM_VERSION

apptainer exec --cleanenv --pwd "$(pwd)/rocSHMEM" "$IMAGE_PATH" jupyter notebook --ip=0.0.0.0 --no-browser --port=8888

echo "Executed image: $IMAGE_NAME with ROCm version: $ROCM_VERSION"