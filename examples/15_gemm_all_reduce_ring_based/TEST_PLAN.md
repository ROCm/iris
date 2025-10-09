# Test Plan for Ring-Based All-Reduce Validation Fix

## Issue Summary
The ring-based all-reduce GEMM was failing validation when run with more than 1 GPU due to a race condition in the cross-GPU synchronization mechanism.

## Root Cause
The original implementation used:
- `iris.store()` to set flags on remote GPUs
- `tl.load()` busy-wait to check local flags

This pattern had a race condition where the flag write could become visible before the data write completed, causing ranks to read stale data from the ring buffer.

## Fix Applied
Changed the synchronization to use atomic operations with proper memory ordering:

1. **Flag Signaling (Producer):**
   - Changed from: `iris.store(locks + tile_id, 1, ...)`
   - Changed to: `iris.atomic_add(locks + tile_id, 1, ..., sem="release", scope="sys")`
   - Effect: Ensures all prior data writes are visible before the flag is set

2. **Flag Waiting (Consumer):**
   - Changed from: `while tl.load(locks + tile_id, ...) != 1: pass`
   - Changed to: `while result != 1: result = iris.atomic_cas(locks + tile_id, 1, 0, ..., sem="acquire", scope="sys")`
   - Effect: Ensures data is visible after the flag is seen, and atomically resets the flag

## Testing Instructions

### Prerequisites
- AMD GPU with ROCm support (e.g., MI300X, MI350X, MI355X)
- Multiple GPUs available (minimum 2)
- Iris installed with dev dependencies

### Test Cases

#### Test 1: Basic Validation with 2 GPUs
```bash
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2
```
**Expected Result:**
- No validation errors
- Output should show: `Final C validation passed`
- Max absolute difference should be small (< atol=2)

#### Test 2: Validation with Different Matrix Sizes
```bash
# Small matrices
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 -m 1024 -n 1024 -k 1024

# Medium matrices
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 -m 4096 -n 2304 -k 18432

# Large matrices (default)
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 -m 8192 -n 4608 -k 36864
```
**Expected Result:**
- All validations should pass
- Results should match expected values within tolerance

#### Test 3: Validation with Different Number of GPUs
```bash
# 2 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2

# 4 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 4

# 8 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 8
```
**Expected Result:**
- All configurations should pass validation
- K and N dimensions must be divisible by num_ranks

#### Test 4: Validation with Different Data Types
```bash
# FP16 (default)
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 --datatype fp16

# FP32
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 --datatype fp32

# BF16
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2 --datatype bf16
```
**Expected Result:**
- All data types should pass validation
- Different data types may have different tolerance requirements

#### Test 5: Combined Validation and Benchmarking
```bash
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --benchmark --num_ranks 2
```
**Expected Result:**
- Validation passes
- Performance metrics are displayed
- TFLOPS value is positive and reasonable

#### Test 6: Stress Test with Many Iterations
```bash
# Run multiple times to check for race conditions
for i in {1..10}; do
    echo "Iteration $i"
    python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2
    if [ $? -ne 0 ]; then
        echo "Failed on iteration $i"
        exit 1
    fi
done
```
**Expected Result:**
- All iterations should pass
- No intermittent failures

## Validation Criteria

### Success Criteria
1. ✅ Validation passes with 2+ GPUs
2. ✅ Max absolute difference is within tolerance (< 2 for FP16)
3. ✅ No mismatch errors reported
4. ✅ Results are consistent across multiple runs
5. ✅ Works with different matrix sizes and data types

### Failure Indicators
1. ❌ "Final C validation failed" message
2. ❌ Large max absolute difference (> tolerance)
3. ❌ Mismatch at specific indices
4. ❌ Intermittent failures across runs
5. ❌ Crashes or hangs

## Comparison with Before Fix

### Before (Original Code)
```
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2
[Iris] [1/2] Max absolute difference: 646.0
[Iris] [0/2] Max absolute difference: 605.5
[Iris] [0/2] Mismatch at index (4641, 1056): C=45.90625, expected=55.25
[Iris] [0/2] Final C validation failed.
```

### After (Fixed Code)
```
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2
[Iris] [0/2] Validating...
[Iris] [1/2] Validating...
[Iris] [0/2] Final C validation passed.
[Iris] [1/2] Final C validation passed.
```

## Technical Details

### Memory Ordering Guarantees

1. **Release Semantics (Producer)**
   - All writes before the atomic_add are visible to other threads
   - The flag write acts as a release barrier
   - Ensures data is committed before signaling completion

2. **Acquire Semantics (Consumer)**
   - All writes before a release operation are visible after the acquire
   - The flag read acts as an acquire barrier
   - Ensures data is visible after the flag is seen

3. **System Scope**
   - Guarantees visibility across all GPUs in the system
   - Required for cross-GPU synchronization
   - Stronger than GPU scope (single GPU) or CTA scope (single compute unit)

### Why This Fix Works

The atomic operations with release/acquire semantics form a synchronization pair:
- Producer releases the data with the flag write
- Consumer acquires the data with the flag read
- The hardware guarantees that all data visible before the release is visible after the acquire

This is a standard pattern used in concurrent programming and is similar to:
- C++ `std::atomic` with `memory_order_release` and `memory_order_acquire`
- Java `volatile` variables
- CUDA cooperative groups synchronization

## References

- Example 09 (gemm_one_shot_all_reduce) uses the same pattern
- Iris documentation on atomic operations
- ROCm memory ordering guarantees
- HIP atomic operation semantics

## Notes

- The fix follows the same pattern used in other Iris examples
- The atomic_cas operation atomically reads and resets the flag, eliminating the need for a separate flag reset
- The system scope is essential for cross-GPU visibility
- The fix is minimal and surgical, changing only the synchronization mechanism
