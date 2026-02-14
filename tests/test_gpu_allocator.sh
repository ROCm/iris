#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Test script for GPU allocator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ALLOCATOR="${SCRIPT_DIR}/../.github/scripts/gpu_allocator.sh"

# Use test-specific state file
export GPU_STATE_FILE="/tmp/test_gpu_state_$$"
export GPU_LOCK_FILE="${GPU_STATE_FILE}.lock"
export MAX_GPUS=8
export RETRY_DELAY=0.1
export MAX_RETRIES=10

# Clean up test files on exit
cleanup() {
    rm -f "$GPU_STATE_FILE" "$GPU_LOCK_FILE"
}
trap cleanup EXIT

# Source the allocator
source "$GPU_ALLOCATOR"

echo "========================================"
echo "Testing GPU Allocator"
echo "========================================"

# Test 1: Initialize and verify state
echo ""
echo "Test 1: Initialize GPU state"
init_gpu_state
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "0,1,2,3,4,5,6,7" ]; then
    echo "✅ PASS: State initialized correctly: $state"
else
    echo "❌ FAIL: Expected '0,1,2,3,4,5,6,7', got '$state'"
    exit 1
fi

# Test 2: Allocate 1 GPU
echo ""
echo "Test 2: Allocate 1 GPU"
acquire_gpus 1
if [ "$ALLOCATED_GPUS" = "0" ]; then
    echo "✅ PASS: Allocated 1 GPU: $ALLOCATED_GPUS"
else
    echo "❌ FAIL: Expected GPU '0', got '$ALLOCATED_GPUS'"
    exit 1
fi
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "1,2,3,4,5,6,7" ]; then
    echo "✅ PASS: Remaining GPUs correct: $state"
else
    echo "❌ FAIL: Expected '1,2,3,4,5,6,7', got '$state'"
    exit 1
fi

# Test 3: Release GPU
echo ""
echo "Test 3: Release GPU"
release_gpus
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "0,1,2,3,4,5,6,7" ]; then
    echo "✅ PASS: GPU released, state: $state"
else
    echo "❌ FAIL: Expected '0,1,2,3,4,5,6,7', got '$state'"
    exit 1
fi

# Test 4: Allocate multiple GPUs
echo ""
echo "Test 4: Allocate 4 GPUs"
acquire_gpus 4
if [ "$ALLOCATED_GPUS" = "0,1,2,3" ]; then
    echo "✅ PASS: Allocated 4 GPUs: $ALLOCATED_GPUS"
else
    echo "❌ FAIL: Expected '0,1,2,3', got '$ALLOCATED_GPUS'"
    exit 1
fi
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "4,5,6,7" ]; then
    echo "✅ PASS: Remaining GPUs correct: $state"
else
    echo "❌ FAIL: Expected '4,5,6,7', got '$state'"
    exit 1
fi
release_gpus

# Test 5: Allocate all GPUs
echo ""
echo "Test 5: Allocate all 8 GPUs"
acquire_gpus 8
if [ "$ALLOCATED_GPUS" = "0,1,2,3,4,5,6,7" ]; then
    echo "✅ PASS: Allocated 8 GPUs: $ALLOCATED_GPUS"
else
    echo "❌ FAIL: Expected '0,1,2,3,4,5,6,7', got '$ALLOCATED_GPUS'"
    exit 1
fi
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "" ]; then
    echo "✅ PASS: No remaining GPUs (empty state)"
else
    echo "❌ FAIL: Expected empty state, got '$state'"
    exit 1
fi
release_gpus

# Test 6: Test allocation failure (more than available)
echo ""
echo "Test 6: Test allocation when not enough GPUs"
acquire_gpus 4
first_allocation="$ALLOCATED_GPUS"
echo "  First allocation: $first_allocation"

# Try to allocate 8 GPUs (should fail quickly due to low MAX_RETRIES)
if acquire_gpus 8 2>/dev/null; then
    echo "❌ FAIL: Should have failed to allocate 8 GPUs when only 4 available"
    exit 1
else
    echo "✅ PASS: Correctly failed to allocate 8 GPUs when only 4 available"
fi

# Verify first allocation is still intact
if [ "$ALLOCATED_GPUS" = "$first_allocation" ]; then
    echo "✅ PASS: First allocation unchanged after failed allocation"
else
    echo "❌ FAIL: First allocation changed after failed allocation"
    exit 1
fi
release_gpus

# Test 7: Concurrent allocations (simulate)
echo ""
echo "Test 7: Sequential allocations (simulating concurrent usage)"
# Allocate 2 GPUs
acquire_gpus 2
alloc1="$ALLOCATED_GPUS"
echo "  Allocation 1: $alloc1"

# Simulate another process by manually updating state
# Save current allocation
saved_alloc="$ALLOCATED_GPUS"
unset ALLOCATED_GPUS

# Allocate 2 more GPUs
acquire_gpus 2
alloc2="$ALLOCATED_GPUS"
echo "  Allocation 2: $alloc2"

# Verify they don't overlap
if [[ "$alloc1" == *"$alloc2"* ]] || [[ "$alloc2" == *"$alloc1"* ]]; then
    echo "❌ FAIL: Allocations overlap: $alloc1 and $alloc2"
    exit 1
else
    echo "✅ PASS: Allocations don't overlap"
fi

# Clean up both allocations
release_gpus
ALLOCATED_GPUS="$saved_alloc"
release_gpus

# Test 8: Verify proper cleanup
echo ""
echo "Test 8: Verify final state"
state=$(cat "$GPU_STATE_FILE")
if [ "$state" = "0,1,2,3,4,5,6,7" ]; then
    echo "✅ PASS: All GPUs returned to pool: $state"
else
    echo "❌ FAIL: Expected all GPUs in pool, got '$state'"
    exit 1
fi

echo ""
echo "========================================"
echo "All tests passed! ✅"
echo "========================================"
