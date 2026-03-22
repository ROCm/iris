#!/bin/bash
set -e

echo "=== Building Docker image iris-dev-gluon ==="
cd /home/muhaawad/git/amd/iris
docker build -t iris-dev-gluon -f docker/Dockerfile docker/

echo "=== Docker image built successfully ==="

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

echo "=== Running tests in container ==="
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$(pwd)":/iris_workspace -w /iris_workspace \
  --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 \
  --group-add "$VIDEO_GID" \
  --group-add "$RENDER_GID" \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint bash \
  iris-dev-gluon /iris_workspace/run_inside_container.sh
