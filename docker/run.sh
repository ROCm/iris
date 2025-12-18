#!/bin/bash
# Run Iris RDMA Docker container with InfiniBand support

IMAGE_NAME=${1:-"iris-rdma"}
WORKSPACE_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "Starting miniQP container..."
echo "  Image: $IMAGE_NAME"
echo "  Workspace: $WORKSPACE_DIR"

# Auto-detect InfiniBand devices
IB_DEVICES=""
if [ -d /dev/infiniband ]; then
    for dev in /dev/infiniband/uverbs*; do
        if [ -e "$dev" ]; then
            IB_DEVICES="$IB_DEVICES --device=$dev"
        fi
    done
    if [ -n "$IB_DEVICES" ]; then
        echo "  InfiniBand devices: $(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l) found"
    fi
else
    echo "  Warning: No InfiniBand devices found"
fi
echo ""

docker run -it --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    $IB_DEVICES \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    -v "$WORKSPACE_DIR:$WORKSPACE_DIR" \
    -w "$WORKSPACE_DIR" \
    --shm-size=16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $IMAGE_NAME

