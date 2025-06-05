#!/bin/bash

IMAGE_NAME=${1:-"iris-dev"}

docker build -t $IMAGE_NAME .