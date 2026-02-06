#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -euo pipefail

# This script runs on the HOST (via devcontainer.json "initializeCommand").
# It ensures there is an ssh-agent with a stable socket at:
#   ~/.ssh/ssh-agent.sock
#
# It also tries to load ~/.ssh/id_rsa if present.
# If your key is passphrase-protected and you're non-interactive, it may fail silently.

SOCK="${HOME}/.ssh/ssh-agent.sock"

mkdir -p "${HOME}/.ssh"

# Check if socket exists AND has keys loaded
if [[ -S "${SOCK}" ]]; then
  if SSH_AUTH_SOCK="${SOCK}" ssh-add -l >/dev/null 2>&1; then
    # Agent is running and has keys loaded
    exit 0
  fi
  
  # Check if agent is alive but just has no keys
  if SSH_AUTH_SOCK="${SOCK}" ssh-add -l 2>&1 | grep -q "no identities"; then
    # Agent is alive, just needs keys loaded - continue to key loading below
    :
  else
    # Agent is dead or socket is stale, remove it
    rm -f "${SOCK}" 2>/dev/null || true
  fi
fi

# Start agent if socket doesn't exist
if [[ ! -S "${SOCK}" ]]; then
  ssh-agent -a "${SOCK}" -t 8h >/dev/null || true
fi

# Load SSH key if it exists
if [[ -f "${HOME}/.ssh/id_rsa" ]]; then
  SSH_AUTH_SOCK="${SOCK}" ssh-add "${HOME}/.ssh/id_rsa" >/dev/null 2>&1 || true
fi

SSH_AUTH_SOCK="${SOCK}" ssh-add -l >/dev/null 2>&1 || true
