#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Lightweight GPU allocator for CI jobs
# Provides isolation and efficient utilization for variable GPU requests
#
# Design:
# - Uses flock for atomic state management
# - Maintains shared state file with 8-bit bitmap (one bit per GPU)
# - Supports variable GPU requests (1, 2, 4, 8 GPUs)
# - Non-contiguous allocation: any available GPUs can be used
# - Out-of-order release safe: each GPU tracked independently
# - Throughput-oriented: first-available scheduling (non-FIFO)
# - Automatic cleanup on job exit
#
# Usage:
#   source gpu_allocator.sh
#   acquire_gpus <num_gpus>  # Blocks until GPUs available, sets GPU_DEVICES and ALLOCATED_GPU_BITMAP
#   enable_gpu_cleanup_trap  # Optional: enable automatic cleanup on EXIT
#   # ... run your job with HIP_VISIBLE_DEVICES=$GPU_DEVICES ...
#   release_gpus             # Releases allocated GPUs back to pool

# Note: Do not modify caller's shell options (e.g., set -e) when sourced.

# Configuration
GPU_STATE_FILE="${GPU_STATE_FILE:-/tmp/iris_gpu_state}"
GPU_LOCK_FILE="${GPU_STATE_FILE}.lock"
MAX_GPUS="${MAX_GPUS:-8}"
RETRY_DELAY="${RETRY_DELAY:-60}"   # 1 minute between checks
MAX_RETRIES="${MAX_RETRIES:-180}"  # 3 hours total wait time (180 * 1 min)

# Initialize GPU state file and validate its contents
# State format: 8-bit bitmap where bit N=1 means GPU N is allocated
init_gpu_state() {
    # Use flock to ensure atomic initialization and validation
    (
        flock -x 200

        if [ ! -f "$GPU_STATE_FILE" ]; then
            # Initialize with all GPUs free (bitmap = 0)
            echo "0" > "$GPU_STATE_FILE"
            echo "[GPU-ALLOC] Initialized GPU bitmap: 0 (all GPUs free)" >&2
        else
            # Validate existing state file contents
            local current_state
            current_state=$(cat "$GPU_STATE_FILE" 2>/dev/null || echo "")

            # Ensure the state is a non-negative integer
            if ! [[ "$current_state" =~ ^[0-9]+$ ]]; then
                echo "0" > "$GPU_STATE_FILE"
                echo "[GPU-ALLOC] Detected invalid GPU bitmap ('$current_state'); reset to 0" >&2
            # Ensure the bitmap is within valid range (0-255 for 8 GPUs)
            elif [ "$current_state" -lt 0 ] || [ "$current_state" -gt 255 ]; then
                echo "0" > "$GPU_STATE_FILE"
                echo "[GPU-ALLOC] Detected out-of-range GPU bitmap ($current_state); reset to 0" >&2
            fi
        fi
    ) 200>"$GPU_LOCK_FILE"
}

# Acquire N GPUs from the pool using bitmap allocation
# Sets GPU_DEVICES environment variable with comma-separated GPU IDs
# Sets ALLOCATED_GPU_BITMAP for cleanup (bitmap of allocated GPUs)
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
        # Try to allocate GPUs atomically using bitmap
        local allocated_gpus=""
        local allocated_bitmap=0
        local result_file
        local lock_exit_code
        result_file=$(mktemp)
        
        (
            flock -x 200
            
            # Read current bitmap
            local bitmap
            bitmap=$(cat "$GPU_STATE_FILE")
            
            # Find N free GPUs (bits that are 0)
            local found_gpus=()
            local gpu_id
            for gpu_id in $(seq 0 $((MAX_GPUS - 1))); do
                # Check if bit gpu_id is 0 (GPU is free)
                if [ $(( (bitmap >> gpu_id) & 1 )) -eq 0 ]; then
                    found_gpus+=("$gpu_id")
                    if [ "${#found_gpus[@]}" -eq "$num_gpus" ]; then
                        break
                    fi
                fi
            done
            
            # Check if we found enough GPUs
            if [ "${#found_gpus[@]}" -eq "$num_gpus" ]; then
                # Mark these GPUs as allocated in the bitmap
                local new_bitmap=$bitmap
                local allocated_mask=0
                for gpu_id in "${found_gpus[@]}"; do
                    new_bitmap=$(( new_bitmap | (1 << gpu_id) ))
                    allocated_mask=$(( allocated_mask | (1 << gpu_id) ))
                done
                
                # Update state file with new bitmap
                echo "$new_bitmap" > "$GPU_STATE_FILE"
                
                # Write results to file while holding the lock
                # Format: "gpu_ids|allocated_mask"
                local gpu_list
                gpu_list=$(IFS=,; echo "${found_gpus[*]}")
                echo "${gpu_list}|${allocated_mask}" > "$result_file"
                
                echo "[GPU-ALLOC] Allocated GPUs: $gpu_list (bitmap: $new_bitmap)" >&2
                exit 0
            else
                # Not enough GPUs available
                local available_count="${#found_gpus[@]}"
                echo "[GPU-ALLOC] Not enough GPUs: need $num_gpus, only $available_count available (bitmap: $bitmap)" >&2
                exit 1
            fi
        ) 200>"$GPU_LOCK_FILE" && lock_exit_code=0 || lock_exit_code=$?
        
        if [ "$lock_exit_code" -eq 0 ]; then
            # Read the allocated GPU IDs and mask from the result file
            local result_line
            result_line=$(cat "$result_file")
            rm -f "$result_file"
            
            allocated_gpus="${result_line%|*}"
            allocated_bitmap="${result_line#*|}"
            
            # Export variables
            GPU_DEVICES="$allocated_gpus"
            ALLOCATED_GPU_BITMAP="$allocated_bitmap"
            export GPU_DEVICES ALLOCATED_GPU_BITMAP
            
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

# Release allocated GPUs back to the pool using bitmap
# Uses ALLOCATED_GPU_BITMAP environment variable
release_gpus() {
    if [ -z "$ALLOCATED_GPU_BITMAP" ]; then
        echo "[GPU-ALLOC] No GPUs to release" >&2
        return 0
    fi
    
    echo "[GPU-ALLOC] Releasing GPUs (bitmap mask: $ALLOCATED_GPU_BITMAP)" >&2
    
    # Save the bitmap to release before entering subshell
    local bitmap_to_release="$ALLOCATED_GPU_BITMAP"
    
    # Unset immediately to prevent double-release
    unset GPU_DEVICES ALLOCATED_GPU_BITMAP
    
    (
        flock -x 200
        
        # Read current bitmap
        local bitmap
        bitmap=$(cat "$GPU_STATE_FILE")
        
        # Clear the bits for the GPUs we're releasing (bitwise AND with inverse of mask)
        local new_bitmap
        new_bitmap=$(( bitmap & ~bitmap_to_release ))
        
        # Update state file
        echo "$new_bitmap" > "$GPU_STATE_FILE"
        
        echo "[GPU-ALLOC] Released GPUs. New bitmap: $new_bitmap" >&2
    ) 200>"$GPU_LOCK_FILE"
}

# Clean up function to ensure GPUs are released
cleanup_gpus() {
    if [ -n "$ALLOCATED_GPU_BITMAP" ]; then
        echo "[GPU-ALLOC] Cleanup: releasing GPUs on exit" >&2
        release_gpus
    fi
}


