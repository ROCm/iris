#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Lightweight GPU allocator for CI jobs
# Provides isolation and efficient utilization for variable GPU requests
#
# Design:
# - Uses flock for atomic state management
# - Maintains shared state file tracking free GPUs
# - Supports variable GPU requests (1, 2, 4, 8 GPUs)
# - Throughput-oriented: first-available scheduling (non-FIFO)
# - Automatic cleanup on job exit
#
# Usage:
#   source gpu_allocator.sh
#   acquire_gpus <num_gpus>  # Blocks until GPUs available, sets ALLOCATED_GPUS
#   # ... run your job with HIP_VISIBLE_DEVICES=$ALLOCATED_GPUS ...
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
            # Create state file with all GPUs free (0-7 for 8-GPU system)
            local all_gpus=""
            for i in $(seq 0 $((MAX_GPUS - 1))); do
                if [ -z "$all_gpus" ]; then
                    all_gpus="$i"
                else
                    all_gpus="$all_gpus,$i"
                fi
            done
            echo "$all_gpus" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Initialized GPU state: $all_gpus" >&2
        fi
    ) 200>"$GPU_LOCK_FILE"
}

# Acquire N GPUs from the pool
# Sets ALLOCATED_GPUS environment variable with comma-separated GPU IDs
# Blocks until requested GPUs are available
acquire_gpus() {
    local num_gpus="$1"
    
    if [ -z "$num_gpus" ] || [ "$num_gpus" -lt 1 ] || [ "$num_gpus" -gt "$MAX_GPUS" ]; then
        echo "[GPU-ALLOC ERROR] Invalid GPU count: $num_gpus (must be 1-$MAX_GPUS)" >&2
        return 1
    fi
    
    # Initialize state if needed
    init_gpu_state
    
    local attempt=0
    local allocated=""
    
    echo "[GPU-ALLOC] Requesting $num_gpus GPU(s)..." >&2
    
    while [ $attempt -lt $MAX_RETRIES ]; do
        # Try to allocate GPUs atomically
        (
            flock -x 200
            
            # Read current free GPUs
            local free_gpus=$(cat "$GPU_STATE_FILE")
            
            # Convert comma-separated list to array
            IFS=',' read -ra gpu_array <<< "$free_gpus"
            
            # Check if we have enough free GPUs
            if [ ${#gpu_array[@]} -ge $num_gpus ]; then
                # Allocate first N GPUs
                allocated=""
                local remaining=""
                
                for i in "${!gpu_array[@]}"; do
                    if [ $i -lt $num_gpus ]; then
                        # Allocate this GPU
                        if [ -z "$allocated" ]; then
                            allocated="${gpu_array[$i]}"
                        else
                            allocated="$allocated,${gpu_array[$i]}"
                        fi
                    else
                        # Keep this GPU in free pool
                        if [ -z "$remaining" ]; then
                            remaining="${gpu_array[$i]}"
                        else
                            remaining="$remaining,${gpu_array[$i]}"
                        fi
                    fi
                done
                
                # Update state file with remaining GPUs
                echo "$remaining" > "$GPU_STATE_FILE"
                
                # Export allocated GPUs
                echo "$allocated"
                echo "[GPU-ALLOC] Allocated GPUs: $allocated (remaining: $remaining)" >&2
                exit 0
            else
                # Not enough GPUs available
                echo "[GPU-ALLOC] Only ${#gpu_array[@]} GPU(s) available, need $num_gpus. Retrying..." >&2
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && break || true
        
        # Sleep before retry
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_RETRIES ]; then
            sleep $RETRY_DELAY
        fi
    done
    
    # Check if we successfully allocated
    allocated=$(
        flock -s 200
        # Re-read to get the allocation result from the subshell
        # We need to track this differently
        cat "$GPU_STATE_FILE" 2>/dev/null || true
    ) 200>"$GPU_LOCK_FILE" || true
    
    # If we got here without breaking, allocation failed
    if [ $attempt -ge $MAX_RETRIES ]; then
        echo "[GPU-ALLOC ERROR] Failed to allocate $num_gpus GPU(s) after $MAX_RETRIES attempts" >&2
        return 1
    fi
    
    # The allocation was successful - we need to return the allocated GPUs
    # This is a bit tricky because we're in a subshell
    # Let's refactor to use a different approach
}

# Better implementation of acquire_gpus that properly returns the allocated GPUs
acquire_gpus() {
    local num_gpus="$1"
    
    if [ -z "$num_gpus" ] || [ "$num_gpus" -lt 1 ] || [ "$num_gpus" -gt "$MAX_GPUS" ]; then
        echo "[GPU-ALLOC ERROR] Invalid GPU count: $num_gpus (must be 1-$MAX_GPUS)" >&2
        return 1
    fi
    
    # Initialize state if needed
    init_gpu_state
    
    local attempt=0
    
    echo "[GPU-ALLOC] Requesting $num_gpus GPU(s)..." >&2
    
    while [ $attempt -lt $MAX_RETRIES ]; do
        # Create temporary file for allocation result
        local result_file=$(mktemp)
        
        # Try to allocate GPUs atomically
        local success=0
        (
            flock -x 200
            
            # Read current free GPUs
            local free_gpus=$(cat "$GPU_STATE_FILE")
            
            # Convert comma-separated list to array
            IFS=',' read -ra gpu_array <<< "$free_gpus"
            
            # Check if we have enough free GPUs
            if [ ${#gpu_array[@]} -ge $num_gpus ]; then
                # Allocate first N GPUs
                local allocated=""
                local remaining=""
                
                for i in "${!gpu_array[@]}"; do
                    if [ $i -lt $num_gpus ]; then
                        # Allocate this GPU
                        if [ -z "$allocated" ]; then
                            allocated="${gpu_array[$i]}"
                        else
                            allocated="$allocated,${gpu_array[$i]}"
                        fi
                    else
                        # Keep this GPU in free pool
                        if [ -z "$remaining" ]; then
                            remaining="${gpu_array[$i]}"
                        else
                            remaining="$remaining,${gpu_array[$i]}"
                        fi
                    fi
                done
                
                # Update state file with remaining GPUs
                echo "$remaining" > "$GPU_STATE_FILE"
                
                # Write allocated GPUs to result file
                echo "$allocated" > "$result_file"
                
                echo "[GPU-ALLOC] Allocated GPUs: $allocated (remaining: $remaining)" >&2
                exit 0
            else
                # Not enough GPUs available
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && success=1 || success=0
        
        if [ $success -eq 1 ]; then
            # Read allocated GPUs from result file
            ALLOCATED_GPUS=$(cat "$result_file")
            rm -f "$result_file"
            export ALLOCATED_GPUS
            return 0
        fi
        
        rm -f "$result_file"
        
        # Sleep before retry
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "[GPU-ALLOC] Retrying... (attempt $((attempt + 1))/$MAX_RETRIES)" >&2
            sleep $RETRY_DELAY
        fi
    done
    
    # If we got here, allocation failed
    echo "[GPU-ALLOC ERROR] Failed to allocate $num_gpus GPU(s) after $MAX_RETRIES attempts" >&2
    return 1
}

# Release allocated GPUs back to the pool
# Uses ALLOCATED_GPUS environment variable
release_gpus() {
    if [ -z "$ALLOCATED_GPUS" ]; then
        echo "[GPU-ALLOC] No GPUs to release" >&2
        return 0
    fi
    
    echo "[GPU-ALLOC] Releasing GPUs: $ALLOCATED_GPUS" >&2
    
    (
        flock -x 200
        
        # Read current free GPUs
        local free_gpus=$(cat "$GPU_STATE_FILE")
        
        # Add allocated GPUs back to free pool
        if [ -z "$free_gpus" ]; then
            free_gpus="$ALLOCATED_GPUS"
        else
            free_gpus="$free_gpus,$ALLOCATED_GPUS"
        fi
        
        # Sort GPUs numerically for consistency
        # Convert to array, sort, convert back
        IFS=',' read -ra gpu_array <<< "$free_gpus"
        IFS=$'\n' sorted=($(sort -n <<<"${gpu_array[*]}"))
        unset IFS
        
        # Join back with commas
        local sorted_gpus=""
        for gpu in "${sorted[@]}"; do
            if [ -z "$sorted_gpus" ]; then
                sorted_gpus="$gpu"
            else
                sorted_gpus="$sorted_gpus,$gpu"
            fi
        done
        
        echo "$sorted_gpus" > "$GPU_STATE_FILE"
        echo "[GPU-ALLOC] Released GPUs. Free GPUs: $sorted_gpus" >&2
    ) 200>"$GPU_LOCK_FILE"
    
    unset ALLOCATED_GPUS
}

# Clean up function to ensure GPUs are released
cleanup_gpus() {
    if [ -n "$ALLOCATED_GPUS" ]; then
        echo "[GPU-ALLOC] Cleanup: releasing GPUs on exit" >&2
        release_gpus
    fi
}

# Register cleanup handler
trap cleanup_gpus EXIT
