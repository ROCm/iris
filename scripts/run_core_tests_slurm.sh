#!/bin/bash
#SBATCH --job-name=iris-core-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=iris_core_tests_%j.out

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_SRC=${REPO_SRC:-${SLURM_SUBMIT_DIR:-$(realpath "$SCRIPT_DIR/..")}}
IMAGE_NAME=${IMAGE_NAME:-iris-dev}
if [ -d "/scratch" ]; then
    DEFAULT_WORK_PARENT="/scratch/$USER"
else
    DEFAULT_WORK_PARENT="/tmp/$USER"
fi
WORK_ROOT=${WORK_ROOT:-$DEFAULT_WORK_PARENT/iris-core-tests-$SLURM_JOB_ID}
WORKSPACE_DIR="$WORK_ROOT/iris"
PERSIST_LOG_ROOT=${PERSIST_LOG_ROOT:-$HOME/slurm-logs/iris-core-tests-$SLURM_JOB_ID}
CONTAINER_NAME="${USER}-iris-core-tests-${SLURM_JOB_ID}"
CONTAINER_LABEL="user=${USER}"

copy_logs_and_cleanup() {
    local exit_code=$1

    if [ -d "$WORKSPACE_DIR/logs" ]; then
        mkdir -p "$PERSIST_LOG_ROOT"
        if ! rsync -a "$WORKSPACE_DIR/logs/" "$PERSIST_LOG_ROOT/"; then
            echo "Failed to copy logs to $PERSIST_LOG_ROOT" >&2
            if [ "$exit_code" -eq 0 ]; then
                exit_code=1
            fi
        fi
    fi

    rm -rf "$WORK_ROOT"
    exit "$exit_code"
}

trap 'copy_logs_and_cleanup $?' EXIT

mkdir -p "$WORK_ROOT"
rsync -a --delete \
    --exclude=".git/" \
    --exclude=".cache/" \
    --exclude=".pytest_cache/" \
    --exclude=".venv/" \
    --exclude="iris.egg-info/" \
    --exclude="logs/" \
    --exclude="results/" \
    "$REPO_SRC/" "$WORKSPACE_DIR/"

cd "$WORKSPACE_DIR"

echo "Repository source: $REPO_SRC"
echo "Scratch workspace: $WORKSPACE_DIR"
echo "Running on node: $(hostname)"
echo "Image name: $IMAGE_NAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available on $(hostname)" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Docker image $IMAGE_NAME not found on $(hostname)" >&2
    exit 1
fi

GPU_ENV_ARGS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_ENV_ARGS+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    GPU_ENV_ARGS+=(-e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}")
    GPU_ENV_ARGS+=(-e "HIP_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}")
fi

docker run --rm \
    --name "$CONTAINER_NAME" \
    --label "$CONTAINER_LABEL" \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${GPU_ENV_ARGS[@]}" \
    -e HOME="$WORKSPACE_DIR" \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e IRIS_MAX_NUM_RANKS=4 \
    -v "$WORKSPACE_DIR:$WORKSPACE_DIR" \
    -w "$WORKSPACE_DIR" \
    --entrypoint bash \
    "$IMAGE_NAME" \
    -lc 'set -euo pipefail; git config --global --add safe.directory "$PWD"; python3 -m pip install -e ".[dev]"; bash scripts/run_core_tests.sh'

echo "Logs copied to $PERSIST_LOG_ROOT"
