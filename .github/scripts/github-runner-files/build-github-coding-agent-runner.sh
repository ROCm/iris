#!/bin/bash

# SLURM job script to build GitHub Coding Agent Runner container

#SBATCH --job-name=build-github-coding-agent-runner
#SBATCH --output=build-github-coding-agent-runner-%j.out
#SBATCH --error=build-github-coding-agent-runner-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH -p mi3001x

set -e

# Parse flags for definition file (and optional output)
# Usage: ./build-github-coding-agent-runner.sh [--def=FILE] [--output=SIF]
#   or:  sbatch build-github-coding-agent-runner.sh  (uses DEF_FILE env or default iris.def)
while [[ $# -gt 0 ]]; do
    case $1 in
        --def=*)         DEF_FILE="${1#*=}"; shift ;;
        --def)           DEF_FILE="${2:-}"; shift 2 ;;
        --definition=*) DEF_FILE="${1#*=}"; shift ;;
        --definition)    DEF_FILE="${2:-}"; shift 2 ;;
        -d)              DEF_FILE="${2:-}"; shift 2 ;;
        --output=*)      OUTPUT_SIF="${1#*=}"; shift ;;
        --output)        OUTPUT_SIF="${2:-}"; shift 2 ;;
        -o)              OUTPUT_SIF="${2:-}"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --def=FILE, --definition=FILE, -d FILE   Apptainer definition file (default: iris.def)"
            echo "  --output=FILE, -o FILE                   Output .sif file (default: github-copilot-coding-agent-runner.sif)"
            exit 0
            ;;
        *) break ;;
    esac
done

# Defaults: when under SLURM with no args, use env; else use script default
DEF_FILE="${DEF_FILE:-iris.def}"
OUTPUT_SIF="${OUTPUT_SIF:-github-copilot-coding-agent-runner.sif}"

echo "=========================================="
echo "GitHub Coding Agent Runner Container Build"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Run from script directory so build and def file are in the right place
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    BUILD_DIR="${SLURM_SUBMIT_DIR}"
else
    BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${BUILD_DIR}"
echo "Build directory: ${BUILD_DIR}"

# Resolve def file path if relative
[ "${DEF_FILE#/}" = "$DEF_FILE" ] && DEF_FILE="${BUILD_DIR}/${DEF_FILE}"
[ "${OUTPUT_SIF#/}" = "$OUTPUT_SIF" ] && OUTPUT_SIF="${BUILD_DIR}/${OUTPUT_SIF}"

if [ ! -f "$DEF_FILE" ]; then
    echo "Error: definition file not found: $DEF_FILE"
    exit 1
fi

# Temp and cache under build dir (avoids /tmp filling up)
export APPTAINER_TMPDIR="${BUILD_DIR}/.apptainer-tmp"
export APPTAINER_CACHEDIR="${BUILD_DIR}/.apptainer-cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo ""
echo "=========================================="
echo "Building container image..."
echo "Definition file: $DEF_FILE"
echo "Output file: $OUTPUT_SIF"
echo "=========================================="

apptainer build --force --fakeroot "$OUTPUT_SIF" "$DEF_FILE"

# Clean build temp to free space (cache is kept for faster rebuilds; remove .apptainer-cache to reclaim that too).
rm -rf "$APPTAINER_TMPDIR"
echo "Cleaned temporary directory: $APPTAINER_TMPDIR"

echo ""
echo "=========================================="
echo "Build completed"
echo "=========================================="

echo ""
echo "Finished: $(date)"
