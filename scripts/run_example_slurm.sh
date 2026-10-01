#!/bin/bash
#SBATCH --job-name=iris-example
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=iris_example_%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/run_example_slurm.sh <example_script> [example args...]

Examples:
  sbatch scripts/run_example_slurm.sh examples/14_all_gather_gemm/example_run_pull.py --num_ranks 4
  sbatch --export=ALL,IMAGE_NAME=my-iris-image scripts/run_example_slurm.sh \
      examples/14_all_gather_gemm/example_run_push.py --num_ranks 4 --dtype bfloat16

Environment overrides:
  IMAGE_NAME        Docker image name (default: iris-dev)
  INSTALL_METHOD    editable | install | git (default: editable)
  WORK_ROOT         Node-local working directory for the staged repo
  PERSIST_LOG_ROOT  Directory where logs/results are copied after the job
EOF
}

if [ $# -lt 1 ]; then
    usage >&2
    exit 1
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_SRC=${REPO_SRC:-${SLURM_SUBMIT_DIR:-$(realpath "$SCRIPT_DIR/..")}}
IMAGE_NAME=${IMAGE_NAME:-iris-dev}
INSTALL_METHOD=${INSTALL_METHOD:-editable}
if [ -d "/scratch" ]; then
    DEFAULT_WORK_PARENT="/scratch/$USER"
else
    DEFAULT_WORK_PARENT="/tmp/$USER"
fi
WORK_ROOT=${WORK_ROOT:-$DEFAULT_WORK_PARENT/iris-example-$SLURM_JOB_ID}
WORKSPACE_DIR="$WORK_ROOT/iris"
PERSIST_LOG_ROOT=${PERSIST_LOG_ROOT:-$HOME/slurm-logs/iris-example-$SLURM_JOB_ID}
CONTAINER_NAME="${USER}-iris-example-${SLURM_JOB_ID}"
CONTAINER_LABEL="user=${USER}"

EXAMPLE_SCRIPT_INPUT=$1
shift
EXAMPLE_ARGS=("$@")

if [[ "$EXAMPLE_SCRIPT_INPUT" = /* ]]; then
    case "$EXAMPLE_SCRIPT_INPUT" in
        "$REPO_SRC"/*)
            EXAMPLE_SCRIPT=${EXAMPLE_SCRIPT_INPUT#"$REPO_SRC"/}
            ;;
        *)
            echo "Example script must be inside the repository: $EXAMPLE_SCRIPT_INPUT" >&2
            exit 1
            ;;
    esac
else
    EXAMPLE_SCRIPT=$EXAMPLE_SCRIPT_INPUT
fi

if [ ! -f "$REPO_SRC/$EXAMPLE_SCRIPT" ]; then
    echo "Example script not found: $EXAMPLE_SCRIPT" >&2
    exit 1
fi

case "$INSTALL_METHOD" in
    editable)
        INSTALL_CMD='python3 -m pip install -e ".[dev]"'
        ;;
    install)
        INSTALL_CMD='python3 -m pip install .'
        ;;
    git)
        REPO=${GITHUB_REPOSITORY:-ROCm/iris}
        SHA=${GITHUB_SHA:-HEAD}
        INSTALL_CMD="python3 -m pip install git+https://github.com/${REPO}.git@${SHA}"
        ;;
    *)
        echo "Unsupported INSTALL_METHOD: $INSTALL_METHOD" >&2
        exit 1
        ;;
esac

printf -v EXAMPLE_SCRIPT_ESCAPED '%q' "$EXAMPLE_SCRIPT"
printf -v EXAMPLE_ARGS_ESCAPED '%q ' "${EXAMPLE_ARGS[@]}"

copy_artifacts_and_cleanup() {
    local exit_code=$1

    mkdir -p "$PERSIST_LOG_ROOT"
    if [ -d "$WORKSPACE_DIR/logs" ]; then
        rsync -a "$WORKSPACE_DIR/logs/" "$PERSIST_LOG_ROOT/logs/" || exit_code=$?
    fi
    if [ -d "$WORKSPACE_DIR/results" ]; then
        rsync -a "$WORKSPACE_DIR/results/" "$PERSIST_LOG_ROOT/results/" || exit_code=$?
    fi

    rm -rf "$WORK_ROOT"
    exit "$exit_code"
}

trap 'copy_artifacts_and_cleanup $?' EXIT

mkdir -p "$WORK_ROOT"
rsync -a --delete \
    --exclude=".git/" \
    --exclude=".cache/" \
    --exclude=".pytest_cache/" \
    --exclude=".venv/" \
    --exclude=".triton/" \
    --exclude="iris.egg-info/" \
    --exclude="logs/" \
    --exclude="results/" \
    "$REPO_SRC/" "$WORKSPACE_DIR/"

cd "$WORKSPACE_DIR"

echo "Repository source: $REPO_SRC"
echo "Scratch workspace: $WORKSPACE_DIR"
echo "Running on node: $(hostname)"
echo "Image name: $IMAGE_NAME"
echo "Example script: $EXAMPLE_SCRIPT"
echo "Example args: ${EXAMPLE_ARGS[*]:-<none>}"
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
    --user "$(id -u):$(id -g)" \
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
    -v "$WORKSPACE_DIR:$WORKSPACE_DIR" \
    -w "$WORKSPACE_DIR" \
    --entrypoint bash \
    "$IMAGE_NAME" \
    -lc "set -euo pipefail; git config --global --add safe.directory \"\$PWD\"; $INSTALL_CMD; python3 $EXAMPLE_SCRIPT_ESCAPED $EXAMPLE_ARGS_ESCAPED"

echo "Artifacts copied to $PERSIST_LOG_ROOT"
