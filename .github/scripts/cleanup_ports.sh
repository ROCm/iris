#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

set -e

# Script to clean up any lingering test processes and ports
# This is useful when tests segfault and leave processes/ports open

echo "========================================"
echo "Port Cleanup Script - Starting"
echo "========================================"

# Show initial state of listening ports
echo ""
echo "Initial state - Listening TCP ports:"
echo "------------------------------------"
ss -tulpn 2>/dev/null | grep LISTEN | grep -E "python|pt_main_thread" || echo "No Python/PyTorch processes listening on ports"
echo ""

echo "Cleaning up lingering test processes and ports..."

# Clean up Python test processes that might be stuck
# Look for processes related to run_tests_distributed.py, pytest, and torch distributed tests
echo "Checking for lingering Python test processes..."
PYTHON_TEST_PIDS=$(pgrep -f "run_tests_distributed.py|pytest.*test_|torch.distributed" 2>/dev/null || true)

if [ -n "$PYTHON_TEST_PIDS" ]; then
    echo "Found Python test processes: $PYTHON_TEST_PIDS"
    echo "Killing Python test processes..."
    echo "$PYTHON_TEST_PIDS" | xargs kill -9 2>/dev/null || true
    echo "Cleaned up Python test processes"
fi

# Clean up pt_main_thread processes (PyTorch multiprocessing spawned processes)
echo "Checking for lingering PyTorch processes (multiprocessing.spawn)..."
PT_PIDS=$(pgrep -f "multiprocessing.spawn" 2>/dev/null || true)

if [ -n "$PT_PIDS" ]; then
    echo "Found PyTorch processes: $PT_PIDS"
    echo "Killing PyTorch processes..."
    echo "$PT_PIDS" | xargs kill -9 2>/dev/null || true
    echo "Cleaned up PyTorch processes"
fi

# Clean up any processes listening on TCP ports in the common test range
# PyTorch distributed typically uses ports in the 29500+ range, but can use any available port
echo "Checking for processes using TCP ports..."
LISTENING_PIDS=$(lsof -ti tcp -sTCP:LISTEN 2>/dev/null | sort -u || true)

if [ -n "$LISTENING_PIDS" ]; then
    # Filter to only Python/PyTorch processes to avoid killing system services
    for PID in $LISTENING_PIDS; do
        PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null || true)
        # Check for python or pt_main_thread processes
        if [[ "$PROCESS_NAME" == *"python"* ]] || [[ "$PROCESS_NAME" == *"pt_main_thread"* ]]; then
            PORT=$(lsof -Pan -p $PID -i tcp -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $9}' | cut -d':' -f2 | head -1)
            if [ -n "$PORT" ]; then
                echo "Found process $PROCESS_NAME (PID $PID) listening on port $PORT"
                kill -9 $PID 2>/dev/null || true
                echo "Cleaned up process $PID on port $PORT"
            fi
        fi
    done
fi

echo ""
echo "========================================"
echo "Port Cleanup Script - Completed"
echo "========================================"

# Show final state of listening ports
echo ""
echo "Final state - Listening TCP ports:"
echo "------------------------------------"
ss -tulpn 2>/dev/null | grep LISTEN | grep -E "python|pt_main_thread" || echo "No Python/PyTorch processes listening on ports"
echo ""
echo "Port cleanup complete."
