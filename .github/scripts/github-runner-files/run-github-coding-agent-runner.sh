#!/bin/bash

# SLURM job script to run GitHub Coding Agent Runner (Iris + Apptainer)

#SBATCH --job-name=github-coding-agent-runner
#SBATCH --output=github-coding-agent-runner-%j.out
#SBATCH --error=github-coding-agent-runner-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH -p mi3008x  # MI300X partition

# Adjust the above SLURM parameters as needed for your system
#
# Two ways to run:
#   1) Standalone with flags (required):
#        ./run-github-coding-agent-runner.sh --github-token='...' --github-repository='owner/repo' --script-dir="$(pwd)" --runner-base="$(pwd)/runner-data"
#   2) Via sbatch with env (SLURM-only fallback): set GITHUB_TOKEN, GITHUB_REPOSITORY; SCRIPT_DIR/RUNNER_BASE default from SLURM_SUBMIT_DIR and WORK
#        export GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo
#        sbatch run-github-coding-agent-runner.sh

set -e

# Parse input flags first. When running under SLURM with no args, env and SLURM defaults are used for any unset value.
while [[ $# -gt 0 ]]; do
    case $1 in
        --github-token=*)      GITHUB_TOKEN="${1#*=}"; shift ;;
        --github-token)        GITHUB_TOKEN="${2:-}"; shift 2 ;;
        --github-repository=*) GITHUB_REPOSITORY="${1#*=}"; shift ;;
        --github-repository)   GITHUB_REPOSITORY="${2:-}"; shift 2 ;;
        --runner-name=*)       RUNNER_NAME="${1#*=}"; shift ;;
        --runner-name)         RUNNER_NAME="${2:-}"; shift 2 ;;
        --cluster-name=*)      CLUSTER_NAME="${1#*=}"; shift ;;
        --cluster-name)        CLUSTER_NAME="${2:-}"; shift 2 ;;
        --runner-labels=*)     RUNNER_LABELS="${1#*=}"; shift ;;
        --runner-labels)       RUNNER_LABELS="${2:-}"; shift 2 ;;
        --script-dir=*)        SCRIPT_DIR="${1#*=}"; shift ;;
        --script-dir)          SCRIPT_DIR="${2:-}"; shift 2 ;;
        --runner-base=*)       RUNNER_BASE="${1#*=}"; shift ;;
        --runner-base)         RUNNER_BASE="${2:-}"; shift 2 ;;
        --sif=*)               SIF_PATH="${1#*=}"; shift ;;
        --sif)                 SIF_PATH="${2:-}"; shift 2 ;;
        --runner-tmp=*)        RUNNER_TMP="${1#*=}"; shift ;;
        --runner-tmp)          RUNNER_TMP="${2:-}"; shift 2 ;;
        --use-overlay=*)       USE_OVERLAY="${1#*=}"; shift ;;
        --use-overlay)         USE_OVERLAY="${2:-}"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options (--option=value or --option value):"
            echo "  --github-token=TOKEN       GitHub token (required)"
            echo "  --github-repository=OWNER/REPO   e.g. Jose/Iris (required)"
            echo "  --script-dir=DIR           Directory with container and scripts (required)"
            echo "  --runner-base=DIR          Runner data base directory (required)"
            echo "  --sif=PATH                 Path to .sif container (default: script-dir/github-copilot-coding-agent-runner.sif)"
            echo "  --runner-name=NAME         Runner name (default: repo-runner-cluster-YYYYMMDD-HHMMSS)"
            echo "  --cluster-name=NAME        Cluster name for default runner name (default: hostname)"
            echo "  --runner-labels=LABELS     Comma-separated labels (default: copilot)"
            echo "  --runner-tmp=DIR           Bind DIR to /tmp in container (e.g. Triton cache)"
            echo "  --use-overlay=0|1          Use overlay (1) or bind mounts only (0)"
            exit 0
            ;;
        *) break ;;
    esac
done

# SLURM-only env fallback: when running under sbatch with no args, use env and SLURM defaults
if [ -n "${SLURM_JOB_ID}" ]; then
    if [ -z "$SCRIPT_DIR" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR:-}"
        [ -z "$SCRIPT_DIR" ] && SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
    if [ -z "$RUNNER_BASE" ]; then
        [ -n "${WORK}" ] && RUNNER_BASE="${WORK}/github-runner-data" || RUNNER_BASE="${SCRIPT_DIR}/github-runner-data"
    fi
    [ -z "$USE_OVERLAY" ] && USE_OVERLAY="${USE_OVERLAY:-1}"
    [ -z "$SIF_PATH" ]    && SIF_PATH="${SIF_PATH:-}"
    [ -z "$RUNNER_NAME" ] && RUNNER_NAME="${RUNNER_NAME:-}"
    [ -z "$RUNNER_LABELS" ] && RUNNER_LABELS="${RUNNER_LABELS:-}"
    [ -z "$RUNNER_TMP" ]  && RUNNER_TMP="${RUNNER_TMP:-}"
fi

# Required: pass as flags when standalone, or set env when using sbatch
[ -n "$GITHUB_TOKEN" ]     || { echo "Error: pass --github-token=TOKEN or set GITHUB_TOKEN (when using sbatch)"; exit 1; }
[ -n "$GITHUB_REPOSITORY" ] || { echo "Error: pass --github-repository=owner/repo or set GITHUB_REPOSITORY (when using sbatch)"; exit 1; }
[ -n "$SCRIPT_DIR" ]       || { echo "Error: pass --script-dir=DIR or set SCRIPT_DIR (when using sbatch)"; exit 1; }
[ -d "$SCRIPT_DIR" ]      || { echo "Error: SCRIPT_DIR must be an existing directory"; exit 1; }
[ -n "$RUNNER_BASE" ]      || { echo "Error: pass --runner-base=DIR or set RUNNER_BASE (when using sbatch)"; exit 1; }

# SIF path: default under script-dir if not passed; relative paths under script-dir
SIF_PATH="${SIF_PATH:-${SCRIPT_DIR}/github-copilot-coding-agent-runner.sif}"
[ "${SIF_PATH#/}" = "$SIF_PATH" ] && SIF_PATH="${SCRIPT_DIR}/${SIF_PATH}"

# Subdirectories of runner base only (no env or separate flags)
RUNNER_WORKDIR="${RUNNER_BASE}/_work"
OVERLAY_DIR="${RUNNER_BASE}/overlay"

# Default runner name: repo-runner-clustername-YYYYMMDD-HHMMSS (e.g. iris-runner-vultr-k8-20260214-025830)
if [ -z "$RUNNER_NAME" ]; then
    REPO_NAME="${GITHUB_REPOSITORY##*/}"
    REPO_NAME="$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')"
    [ -z "$CLUSTER_NAME" ] && CLUSTER_NAME="$(hostname 2>/dev/null || echo local)"
    RUNNER_NAME="${REPO_NAME}-runner-${CLUSTER_NAME}-$(date +%Y%m%d)-$(date +%H%M%S)"
fi
RUNNER_LABELS="${RUNNER_LABELS:-copilot}"
mkdir -p "${RUNNER_WORKDIR}" "${RUNNER_WORKDIR}/.home" "${RUNNER_WORKDIR}/.pip-cache" "${RUNNER_WORKDIR}/.tmp" "${RUNNER_WORKDIR}/.cache"
[ -n "${USE_OVERLAY}" ] && [ "${USE_OVERLAY}" != "0" ] && mkdir -p "${OVERLAY_DIR}"

echo "=========================================="
echo "GitHub Coding Agent Runner - SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo "Repository: $GITHUB_REPOSITORY"
echo "Runner Name: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "Script/container directory: $SCRIPT_DIR"
echo "Runner base: $RUNNER_BASE"
echo "Container SIF: $SIF_PATH"
echo "Overlay directory: $OVERLAY_DIR"
echo "Work directory: $RUNNER_WORKDIR"
echo "TMP bind: ${RUNNER_TMP:-<none>}"
echo "Overlay: ${USE_OVERLAY:-0} (use USE_OVERLAY=1 to enable in non-SLURM)"
echo "=========================================="

# Change to the directory containing the container
cd "${SCRIPT_DIR}"

# Container must exist (build first with: sbatch build-github-coding-agent-runner.sh)
if [ ! -f "$SIF_PATH" ]; then
    echo "Error: container not found: $SIF_PATH"
    echo "Build first: cd ${SCRIPT_DIR} && sbatch build-github-coding-agent-runner.sh"
    exit 1
fi

# Writable runner install dir (start.sh installs runner here if missing)
RUNNER_HOME_HOST="${RUNNER_BASE}/.github-runner"
mkdir -p "${RUNNER_HOME_HOST}"
# When running as root (e.g. in a K8s pod), chown so start.sh can re-exec as nobody and still write
if [ "$(id -u)" = "0" ]; then
    chown -R 65534:65534 "${RUNNER_HOME_HOST}" "${RUNNER_WORKDIR}" 2>/dev/null || true
fi

# Show GPU info
echo "GPU Information:"
rocm-smi --showproductname || echo "Warning: Could not get GPU info"
echo "=========================================="

# Run github-copilot-coding-agent-runner.sif: mount start.sh and writable dirs.
# RUNNER_HOME=/runner-home so start.sh installs/runs the runner there (no HOME override).
#
# Options (overlay not available in Kubernetes):
# - USE_OVERLAY=1 (SLURM): use --overlay for a writable layer (needs overlayfs).
# - USE_OVERLAY=0 (default in K8s/pods): no overlay; only bind mounts. Writable paths:
#   RUNNER_HOME_HOST (runner config), RUNNER_WORKDIR (job work), and optionally
#   RUNNER_TMP (bind to /tmp) if set, so /tmp is writable (e.g. Triton cache).
RUNNER_TMP_BIND=""
if [ -n "${RUNNER_TMP:-}" ] && [ -d "${RUNNER_TMP}" ]; then
    RUNNER_TMP_BIND="--bind ${RUNNER_TMP}:/tmp:rw"
fi

if [ -n "${USE_OVERLAY}" ] && [ "${USE_OVERLAY}" != "0" ] && [ -d "${OVERLAY_DIR}" ]; then
    apptainer exec \
        --no-home \
        --overlay "${OVERLAY_DIR}" \
        --bind "${SCRIPT_DIR}:/runner-scripts:ro" \
        --bind "${RUNNER_HOME_HOST}:/runner-home:rw" \
        --bind "${RUNNER_WORKDIR}:${RUNNER_WORKDIR}" \
        --env "RUNNER_HOME=/runner-home" \
        --env "GITHUB_TOKEN=${GITHUB_TOKEN}" \
        --env "GITHUB_REPOSITORY=${GITHUB_REPOSITORY}" \
        --env "RUNNER_NAME=${RUNNER_NAME}" \
        --env "RUNNER_LABELS=${RUNNER_LABELS}" \
        --env "RUNNER_WORKDIR=${RUNNER_WORKDIR}" \
        --rocm \
        "$SIF_PATH" \
        /bin/bash -c "/runner-scripts/start.sh"
else
    # No overlay (Kubernetes or USE_OVERLAY=0): bind mounts only
    # Optional: set RUNNER_TMP to a writable dir (e.g. pod emptyDir) to bind /tmp for Triton/cache
    apptainer exec \
        --no-home \
        --bind "${SCRIPT_DIR}:/runner-scripts:ro" \
        --bind "${RUNNER_HOME_HOST}:/runner-home:rw" \
        --bind "${RUNNER_WORKDIR}:${RUNNER_WORKDIR}" \
        ${RUNNER_TMP_BIND:+"$RUNNER_TMP_BIND"} \
        --env "RUNNER_HOME=/runner-home" \
        --env "GITHUB_TOKEN=${GITHUB_TOKEN}" \
        --env "GITHUB_REPOSITORY=${GITHUB_REPOSITORY}" \
        --env "RUNNER_NAME=${RUNNER_NAME}" \
        --env "RUNNER_LABELS=${RUNNER_LABELS}" \
        --env "RUNNER_WORKDIR=${RUNNER_WORKDIR}" \
        --rocm \
        "$SIF_PATH" \
        /bin/bash -c "/runner-scripts/start.sh"
fi

echo "=========================================="
echo "GitHub Coding Agent Runner stopped"
echo "=========================================="
