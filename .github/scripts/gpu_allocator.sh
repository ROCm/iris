#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Lightweight GPU allocator for CI jobs
# Provides isolation and efficient utilization for variable GPU requests
#
# Design:
# - Uses flock for atomic state management
# - Maintains shared state file with next available GPU index
# - Supports variable GPU requests (1, 2, 4, 8 GPUs)
# - Throughput-oriented: first-available scheduling (non-FIFO)
# - Automatic cleanup on job exit
#
# Usage:
#   source gpu_allocator.sh
#   acquire_gpus <num_gpus>  # Blocks until GPUs available, sets GPU_DEVICES and ALLOCATED_GPU_START
#   # ... run your job with HIP_VISIBLE_DEVICES=$GPU_DEVICES ...
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
            # Initialize with next available GPU index at 0
            echo "0" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Initialized GPU state: next available GPU 0" >&2
        fi
    ) 200>"$GPU_LOCK_FILE"
}

# Acquire N GPUs from the pool
# Sets GPU_DEVICES environment variable with comma-separated GPU IDs
# Sets ALLOCATED_GPU_START and ALLOCATED_GPU_COUNT for cleanup
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
            
            # Read next available GPU index
            local next_gpu
            next_gpu=$(cat "$GPU_STATE_FILE")
            
            # Check if we have enough contiguous GPUs available
            local end_idx=$((next_gpu + num_gpus))
            if [ "$end_idx" -le "$MAX_GPUS" ]; then
                # Allocate GPUs by updating next available index
                echo "$end_idx" > "$GPU_STATE_FILE"
                
                echo "[GPU-ALLOC] Allocated GPUs $next_gpu-$((end_idx - 1)) (next available: $end_idx)" >&2
                exit 0
            else
                # Not enough GPUs available
                echo "[GPU-ALLOC] Need GPUs $next_gpu-$((end_idx - 1)) but only 0-$((MAX_GPUS - 1)) available" >&2
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && success=1 || success=0
        
        if [ $success -eq 1 ]; then
            # Calculate the actual start index from the updated state
            local next_available
            next_available=$(cat "$GPU_STATE_FILE")
            local allocated_start=$((next_available - num_gpus))
            
            # Build GPU_DEVICES string
            local gpu_devices=""
            for ((i=0; i<num_gpus; i++)); do
                if [ -z "$gpu_devices" ]; then
                    gpu_devices="$((allocated_start + i))"
                else
                    gpu_devices="$gpu_devices,$((allocated_start + i))"
                fi
            done
            
            # Export variables
            GPU_DEVICES="$gpu_devices"
            ALLOCATED_GPU_START="$allocated_start"
            ALLOCATED_GPU_COUNT="$num_gpus"
            export GPU_DEVICES ALLOCATED_GPU_START ALLOCATED_GPU_COUNT
            
            echo "[GPU-ALLOC] Set GPU_DEVICES=$GPU_DEVICES" >&2
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
# Uses ALLOCATED_GPU_START and ALLOCATED_GPU_COUNT environment variables
release_gpus() {
    if [ -z "$ALLOCATED_GPU_COUNT" ]; then
        echo "[GPU-ALLOC] No GPUs to release" >&2
        return 0
    fi
    
    echo "[GPU-ALLOC] Releasing $ALLOCATED_GPU_COUNT GPU(s) starting at index $ALLOCATED_GPU_START" >&2
    
    # Save the values to release before entering subshell
    local start_to_release="$ALLOCATED_GPU_START"
    local count_to_release="$ALLOCATED_GPU_COUNT"
    
    # Unset immediately to prevent double-release
    unset GPU_DEVICES ALLOCATED_GPU_START ALLOCATED_GPU_COUNT
    
    (
        flock -x 200
        
        # Read current next available GPU index
        local next_gpu
        next_gpu=$(cat "$GPU_STATE_FILE")
        
        # Check if the GPUs we're releasing are at the end of the allocated range
        local expected_next=$((start_to_release + count_to_release))
        if [ "$next_gpu" -eq "$expected_next" ]; then
            # We're releasing the most recently allocated GPUs
            # Move the next available index back
            echo "$start_to_release" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Released GPUs. Next available GPU: $start_to_release" >&2
        else
            # GPUs released out of order - this can happen with parallel jobs
            # For simplicity, we just reset to 0 when we detect this
            # This isn't perfect but ensures we don't leak GPUs
            echo "0" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Released GPUs (out of order). Reset next available to 0" >&2
        fi
    ) 200>"$GPU_LOCK_FILE"
}

# Clean up function to ensure GPUs are released
cleanup_gpus() {
    if [ -n "$ALLOCATED_GPU_COUNT" ]; then
        echo "[GPU-ALLOC] Cleanup: releasing GPUs on exit" >&2
        release_gpus
    fi
}

# Register cleanup handler
trap cleanup_gpus EXIT
