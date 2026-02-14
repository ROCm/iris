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
#   enable_gpu_cleanup_trap  # Optional: enable automatic cleanup on EXIT
#   # ... run your job with HIP_VISIBLE_DEVICES=$GPU_DEVICES ...
#   release_gpus             # Releases allocated GPUs back to pool

# Note: Do not modify caller's shell options (e.g., set -e) when sourced.

# Configuration
GPU_STATE_FILE="${GPU_STATE_FILE:-/tmp/iris_gpu_state}"
GPU_LOCK_FILE="${GPU_STATE_FILE}.lock"
MAX_GPUS="${MAX_GPUS:-8}"
RETRY_DELAY="${RETRY_DELAY:-2}"
MAX_RETRIES="${MAX_RETRIES:-300}"  # 10 minutes with 2s delay

# Initialize GPU state file and validate its contents
init_gpu_state() {
    # Use flock to ensure atomic initialization and validation
    (
        flock -x 200

        if [ ! -f "$GPU_STATE_FILE" ]; then
            # Initialize with next available GPU index at 0
            echo "0" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Initialized GPU state: next available GPU 0" >&2
        else
            # Validate existing state file contents
            local current_state
            current_state=$(cat "$GPU_STATE_FILE" 2>/dev/null || echo "")

            # Ensure the state is a non-negative integer
            if ! [[ "$current_state" =~ ^[0-9]+$ ]]; then
                echo "0" > "$GPU_STATE_FILE"
                echo "[GPU-ALLOC] Detected invalid GPU state ('$current_state'); reset to 0" >&2
            # Ensure the state is within [0, MAX_GPUS]
            elif [ "$current_state" -lt 0 ] || [ "$current_state" -gt "$MAX_GPUS" ]; then
                echo "0" > "$GPU_STATE_FILE"
                echo "[GPU-ALLOC] Detected out-of-range GPU state ($current_state); reset to 0" >&2
            fi
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
    
    echo "[GPU-ALLOC] Configuration: MAX_GPUS=$MAX_GPUS, MAX_RETRIES=$MAX_RETRIES, RETRY_DELAY=$RETRY_DELAY" >&2
    echo "[GPU-ALLOC] Requesting $num_gpus GPU(s)..." >&2
    
    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        # Try to allocate GPUs atomically and capture the start index
        local allocated_start=""
        local result_file
        local lock_exit_code
        result_file=$(mktemp)
        
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
                
                # Write the starting index to the result file while holding the lock
                echo "$next_gpu" > "$result_file"
                
                echo "[GPU-ALLOC] Allocated GPUs $next_gpu-$((end_idx - 1)) (next available: $end_idx)" >&2
                exit 0
            else
                # Not enough GPUs available
                local available_count=$((MAX_GPUS - next_gpu))
                echo "[GPU-ALLOC] Not enough GPUs: need $num_gpus, only $available_count available (next free GPU: $next_gpu)" >&2
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && lock_exit_code=0 || lock_exit_code=$?
        
        if [ "$lock_exit_code" -eq 0 ]; then
            # Read the allocated start index from the result file
            allocated_start=$(cat "$result_file")
            rm -f "$result_file"
            
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
        else
            rm -f "$result_file"
        fi
        
        # Sleep before retry
        attempt=$((attempt + 1))
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            echo "[GPU-ALLOC] Retrying... (attempt $((attempt + 1))/$MAX_RETRIES)" >&2
            sleep "$RETRY_DELAY"
        fi
    done
    
    # If we got here, allocation failed
    echo "[GPU-ALLOC ERROR] Failed to allocate $num_gpus GPU(s) after $attempt attempts (MAX_RETRIES=$MAX_RETRIES)" >&2
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
            # With only a single "next index" pointer, we cannot safely reuse these
            # GPUs without risking overlapping allocations. Leave the state unchanged
            # to preserve isolation; this may underutilize some GPUs but is safe.
            echo "[GPU-ALLOC] Released GPUs (out of order). Leaving next available at $next_gpu to avoid overlap" >&2
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

# Enable cleanup handler for the caller's shell.
# This should be called after a successful acquire_gpus invocation.
# It composes with any existing EXIT trap instead of overwriting it.
enable_gpu_cleanup_trap() {
    # Avoid installing the trap multiple times
    if [ "${GPU_ALLOC_CLEANUP_TRAP_ENABLED:-0}" -eq 1 ]; then
        return 0
    fi

    GPU_ALLOC_CLEANUP_TRAP_ENABLED=1
    export GPU_ALLOC_CLEANUP_TRAP_ENABLED

    # Capture any existing EXIT trap so we can chain it
    local existing_exit_trap
    existing_exit_trap=$(trap -p EXIT | sed -n "s/^trap -- '\(.*\)' EXIT$/\1/p")

    if [ -n "$existing_exit_trap" ]; then
        # First run cleanup_gpus, then the previously registered EXIT handler
        # shellcheck disable=SC2064
        trap "cleanup_gpus; $existing_exit_trap" EXIT
    else
        trap cleanup_gpus EXIT
    fi
}
