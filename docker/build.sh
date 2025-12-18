#!/bin/bash
# Build miniQP Docker image

SCRIPT_DIR=$(dirname "$(realpath "$0")")
IMAGE_NAME=${1:-"iris-rdma"}

pushd "$SCRIPT_DIR" > /dev/null

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME --network=host .

popd > /dev/null

