#!/bin/bash

# GitHub Actions Runner startup script for Apptainer (SLURM, standalone, or Kubernetes)
#
# Usage: env only. Required: GITHUB_TOKEN, GITHUB_REPOSITORY, RUNNER_HOME
# Optional: RUNNER_NAME, RUNNER_LABELS, RUNNER_WORKDIR, RUNNER_ENV_FILE, etc.
# See runner-container.env.example and README for details.

set -e

# Required: set when launching the runner (e.g. by run-github-coding-agent-runner.sh or pod spec)
[ -n "$GITHUB_TOKEN" ]     || { echo "Error: GITHUB_TOKEN is required"; exit 1; }
[ -n "$GITHUB_REPOSITORY" ] || { echo "Error: GITHUB_REPOSITORY is required (owner/repo)"; exit 1; }
[ -n "${RUNNER_HOME:-}" ]  || { echo "Error: RUNNER_HOME is required"; exit 1; }

# Default values (set early so env file can use RUNNER_WORKDIR / RUNNER_HOME)
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-$(date +%s)}"
RUNNER_LABELS="${RUNNER_LABELS:-copilot}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-$(dirname "${RUNNER_HOME}")/_work}"

# Source container env file so variables can be set or sourced (override with RUNNER_ENV_FILE)
if [ -n "${RUNNER_ENV_FILE:-}" ] && [ -f "${RUNNER_ENV_FILE}" ]; then
    echo "Sourcing env file: ${RUNNER_ENV_FILE}"
    set -a
    # shellcheck source=/dev/null
    . "${RUNNER_ENV_FILE}"
    set +a
elif [ -f "${RUNNER_HOME}/runner-container.env" ]; then
    echo "Sourcing env file: ${RUNNER_HOME}/runner-container.env"
    set -a
    # shellcheck source=/dev/null
    . "${RUNNER_HOME}/runner-container.env"
    set +a
else
    RUNNER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [ -n "${RUNNER_SCRIPT_DIR:-}" ] && [ -f "${RUNNER_SCRIPT_DIR}/runner-container.env" ]; then
        echo "Sourcing env file: ${RUNNER_SCRIPT_DIR}/runner-container.env"
        set -a
        # shellcheck source=/dev/null
        . "${RUNNER_SCRIPT_DIR}/runner-container.env"
        set +a
    fi
fi

# Runner-only defaults (use RUNNER_WORKDIR; no host-specific paths here).
# PATH, PYTHONPATH, ROCM_PATH, LD_LIBRARY_PATH, etc. come from the container
# image or from runner-container.env (see runner-container.env.example).
# Copy and edit that file per host/container so workflows see the right tools.
export RUNNER_ALLOW_RUNASROOT="${RUNNER_ALLOW_RUNASROOT:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNNER_WORKDIR}/.triton_cache}"

# Writable HOME/TMPDIR for job steps (run-github-coding-agent-runner.sh may already create dirs on host)
mkdir -p "${RUNNER_WORKDIR}/.home" "${RUNNER_WORKDIR}/.tmp"
export HOME="${RUNNER_WORKDIR}/.home"
export TMPDIR="${RUNNER_WORKDIR}/.tmp"

mkdir -p "${RUNNER_HOME}"

echo "=========================================="
echo "GitHub Actions Runner - Apptainer Edition"
echo "=========================================="
echo "Repository: $GITHUB_REPOSITORY"
echo "Runner Name: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "Work Directory: $RUNNER_WORKDIR"
echo "Runner Home: $RUNNER_HOME"
echo "=========================================="

# Install runner binaries if not already present
if [ ! -f "${RUNNER_HOME}/run.sh" ]; then
    echo "Setting up runner in ${RUNNER_HOME}..."
    if [ -d /opt/actions-runner ] && [ -f /opt/actions-runner/run.sh ]; then
        cp -r /opt/actions-runner/* "${RUNNER_HOME}/"
        chmod +x "${RUNNER_HOME}"/*.sh
    else
        RUNNER_VERSION="${RUNNER_VERSION:-2.313.0}"
        echo "Downloading Actions runner v${RUNNER_VERSION}..."
        (cd "${RUNNER_HOME}" && curl -sL -o runner.tgz \
            "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
            && tar xzf runner.tgz && rm -f runner.tgz)
        chmod +x "${RUNNER_HOME}"/*.sh 2>/dev/null || true
    fi
fi

# Change to writable runner directory
cd "${RUNNER_HOME}"

# Create work directory if it doesn't exist
mkdir -p "$RUNNER_WORKDIR"

# Get registration token
echo "Getting registration token..."
REGISTRATION_RESPONSE=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runners/registration-token")
if command -v jq >/dev/null 2>&1; then
    REGISTRATION_TOKEN=$(echo "$REGISTRATION_RESPONSE" | jq -r .token)
else
    REGISTRATION_TOKEN=$(echo "$REGISTRATION_RESPONSE" | grep -o '"token":"[^"]*"' | head -1 | cut -d'"' -f4)
fi

if [ "$REGISTRATION_TOKEN" == "null" ] || [ -z "$REGISTRATION_TOKEN" ]; then
    echo "Error: Failed to get registration token."
    echo "Please check:"
    echo "  1. GITHUB_TOKEN has 'repo' scope"
    echo "  2. Token has not expired"
    echo "  3. GITHUB_REPOSITORY format is correct (owner/repo)"
    exit 1
fi

echo "Registration token obtained successfully"

# Check if already configured (cleanup any previous config)
if [ -f ".runner" ]; then
    echo "Found existing runner configuration, removing..."
    ./config.sh remove --token "$REGISTRATION_TOKEN" || true
fi

# Configure the runner
echo "Configuring runner..."
./config.sh \
    --url "https://github.com/$GITHUB_REPOSITORY" \
    --token "$REGISTRATION_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work "$RUNNER_WORKDIR" \
    --unattended \
    --replace

# Cleanup function
cleanup() {
    # Kill any stale MCP processes left over from cancelled jobs
    pkill -f "mcp/dist/index.js" 2>/dev/null || true
    pkill -f "mcp-server-playwright" 2>/dev/null || true
    pkill -f "playwright-mcp" 2>/dev/null || true

    # Only run removal once; skip if config already removed
    if [ ! -f "${RUNNER_HOME}/.runner" ]; then
        echo "Runner config already removed or not configured. Skipping cleanup."
        return 0
    fi

    echo ""
    echo "Shutting down... Removing runner from GitHub..."

    REMOVE_RESPONSE=$(curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runners/remove-token")
    if command -v jq >/dev/null 2>&1; then
        REMOVE_TOKEN=$(echo "$REMOVE_RESPONSE" | jq -r .token)
    else
        REMOVE_TOKEN=$(echo "$REMOVE_RESPONSE" | grep -o '"token":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ "$REMOVE_TOKEN" != "null" ] && [ -n "$REMOVE_TOKEN" ]; then
        ./config.sh remove --token "$REMOVE_TOKEN"
        echo "Runner removed successfully"
    else
        echo "Warning: Could not remove runner automatically"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Fix git safe directory issues (common when running as root in containers)
# Point git config to a writable location (can be overridden by env file)
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-${RUNNER_WORKDIR}/.gitconfig}"
mkdir -p "$(dirname "$GIT_CONFIG_GLOBAL")"
git config --global --add safe.directory '*'

# Start the runner
echo "Starting GitHub Actions Runner..."
echo "Press Ctrl+C to stop"
echo "=========================================="
command -v rocminfo >/dev/null 2>&1 && rocminfo || true
./run.sh
