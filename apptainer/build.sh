#!/bin/bash


# Build the SIF image with the specified ROCm version
IMAGE_NAME="iris.sif"
apptainer build apptainer/images/"$IMAGE_NAME" apptainer/iris.def

echo "Built image: $IMAGE_NAME"
