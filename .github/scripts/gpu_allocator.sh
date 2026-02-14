#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Lightweight GPU allocator for CI jobs
# Provides isolation and efficient utilization for variable GPU requests
#
# Design:
# - Uses flock for atomic state management
# - Maintains shared state file with count of available GPUs
# - Supports variable GPU requests (1, 2, 4, 8 GPUs)
# - Throughput-oriented: first-available scheduling (non-FIFO)
# - Automatic cleanup on job exit
#
# Usage:
#   source gpu_allocator.sh
#   acquire_gpus <num_gpus>  # Blocks until GPUs available, sets ALLOCATED_GPUS_COUNT
#   # ... run your job ...
#   release_gpus             # Releases allocated GPUs back to pool

set -e

# Configuration
GPU_STATE_FILE="${GPU_STATE_FILE:-/tmp/iris_gpu_state}"
GPU_LOCK_FILE="${GPU_STATE_FILE}.lock"
MAX_GPUS="${MAX_GPUS:-8}"
RETRY_DELAY="${RETRY_DELAY:-2}"
MAX_RETRIES="${MAX_RETRIES:-300}"  # 10 minutes with 2s delay

# Initialize GPU state file if it doesn't exist
init_gpu_state() {
    # Use flock to ensure atomic initialization
    (
        flock -x 200
        if [ ! -f "$GPU_STATE_FILE" ]; then
            # Initialize with all GPUs available
            echo "$MAX_GPUS" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Initialized GPU state: $MAX_GPUS GPUs available" >&2
        fi
    ) 200>"$GPU_LOCK_FILE"
}

# Acquire N GPUs from the pool
# Sets ALLOCATED_GPUS_COUNT environment variable with number of GPUs allocated
# Blocks until requested GPUs are available
acquire_gpus() {
    local num_gpus="$1"
    
    # Validate input is provided and is numeric
    if [ -z "$num_gpus" ]; then
        echo "[GPU-ALLOC ERROR] GPU count not specified" >&2
        return 1
    fi
    
    # Check if numeric
    if ! [[ "$num_gpus" =~ ^[0-9]+$ ]]; then
        echo "[GPU-ALLOC ERROR] GPU count must be a number: $num_gpus" >&2
        return 1
    fi
    
    # Validate range
    if [ "$num_gpus" -lt 1 ] || [ "$num_gpus" -gt "$MAX_GPUS" ]; then
        echo "[GPU-ALLOC ERROR] Invalid GPU count: $num_gpus (must be 1-$MAX_GPUS)" >&2
        return 1
    fi
    
    # Initialize state if needed
    init_gpu_state
    
    local attempt=0
    
    echo "[GPU-ALLOC] Requesting $num_gpus GPU(s)..." >&2
    
    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        # Try to allocate GPUs atomically
        local success=0
        (
            flock -x 200
            
            # Read current available GPU count
            local available
            available=$(cat "$GPU_STATE_FILE")
            
            # Check if we have enough GPUs
            if [ "$available" -ge "$num_gpus" ]; then
                # Allocate GPUs by reducing the count
                local remaining=$((available - num_gpus))
                echo "$remaining" > "$GPU_STATE_FILE"
                
                echo "[GPU-ALLOC] Allocated $num_gpus GPU(s) (remaining: $remaining)" >&2
                exit 0
            else
                # Not enough GPUs available
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && success=1 || success=0
        
        if [ $success -eq 1 ]; then
            # Store allocated count
            ALLOCATED_GPUS_COUNT="$num_gpus"
            export ALLOCATED_GPUS_COUNT
            return 0
        fi
        
        # Sleep before retry
        attempt=$((attempt + 1))
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            echo "[GPU-ALLOC] Retrying... (attempt $((attempt + 1))/$MAX_RETRIES)" >&2
            sleep "$RETRY_DELAY"
        fi
    done
    
    # If we got here, allocation failed
    echo "[GPU-ALLOC ERROR] Failed to allocate $num_gpus GPU(s) after $MAX_RETRIES attempts" >&2
    return 1
}

# Release allocated GPUs back to the pool
# Uses ALLOCATED_GPUS_COUNT environment variable
release_gpus() {
    if [ -z "$ALLOCATED_GPUS_COUNT" ]; then
        echo "[GPU-ALLOC] No GPUs to release" >&2
        return 0
    fi
    
    echo "[GPU-ALLOC] Releasing $ALLOCATED_GPUS_COUNT GPU(s)" >&2
    
    # Save the count to release before entering subshell
    local gpus_to_release="$ALLOCATED_GPUS_COUNT"
    
    # Unset immediately to prevent double-release
    unset ALLOCATED_GPUS_COUNT
    
    (
        flock -x 200
        
        # Read current available count
        local available
        available=$(cat "$GPU_STATE_FILE")
        
        # Add released GPUs back to pool
        local new_count=$((available + gpus_to_release))
        
        echo "$new_count" > "$GPU_STATE_FILE"
        echo "[GPU-ALLOC] Released GPUs. Available GPUs: $new_count" >&2
    ) 200>"$GPU_LOCK_FILE"
}

# Clean up function to ensure GPUs are released
cleanup_gpus() {
    if [ -n "$ALLOCATED_GPUS_COUNT" ]; then
        echo "[GPU-ALLOC] Cleanup: releasing GPUs on exit" >&2
        release_gpus
    fi
}

# Register cleanup handler
trap cleanup_gpus EXIT
