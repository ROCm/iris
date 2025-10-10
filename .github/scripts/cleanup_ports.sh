#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Script to clean up any lingering processes using common test ports
# This is useful when tests segfault and leave ports open

echo "Cleaning up lingering processes on test ports..."

# Common ports used by distributed tests
PORTS=(29500 29501 29502 29503 29504 29505)

for PORT in "${PORTS[@]}"; do
    # Find processes listening on the port
    PIDS=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    
    if [ -n "$PIDS" ]; then
        echo "Found processes using port $PORT: $PIDS"
        echo "Killing processes: $PIDS"
        kill -9 $PIDS 2>/dev/null || true
        echo "Cleaned up port $PORT"
    fi
done

echo "Port cleanup complete."
