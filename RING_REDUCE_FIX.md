# Ring-Based All-Reduce Validation Fix

## Problem
The ring-based all-reduce implementation was failing validation with errors like:
```
[Iris] [0/2] Max absolute difference: 605.5
[Iris] [0/2] Mismatch at index (4641, 1056): C=45.90625, expected=55.25
```

## Root Cause
The issue was caused by improper memory ordering and visibility in the lock-based synchronization mechanism used for the ring all-reduce algorithm. The original implementation used:
1. Regular `iris.store` for setting locks
2. Regular `tl.load` with cache modifiers for polling locks
3. Regular `tl.store` for resetting locks

This approach did not guarantee proper memory ordering and visibility across ranks, leading to:
- Data races where the lock might be read before the data is visible
- Cache coherence issues where stale data might be read
- Missing synchronization guarantees needed for multi-GPU communication

## Solution
Replaced the lock synchronization mechanism with proper atomic operations:

1. **Setting locks**: Changed from `iris.store` to `iris.atomic_xchg` with:
   - `sem="release"`: Ensures all previous memory operations (including the data write) are visible before the lock is set
   - `scope="sys"`: Ensures visibility across the entire system, not just the local GPU

2. **Waiting on locks**: Changed from `tl.load` to `tl.atomic_cas` with:
   - `sem="acquire"`: Ensures the data write is visible after the lock is observed
   - `scope="sys"`: Ensures proper synchronization across all ranks

3. **Resetting locks**: Changed from `tl.store` to `tl.atomic_xchg` with:
   - `sem="release"`: Ensures proper ordering for subsequent iterations
   - `scope="sys"`: Maintains consistency across the system

4. **Initialization**: Added explicit zeroing of locks and ring_buffer before each run to ensure clean state

## Technical Details

### Memory Ordering
The acquire-release semantics provide the necessary happens-before relationships:
- The release fence on the lock write ensures the data write happens-before the lock write
- The acquire fence on the lock read ensures the lock read happens-before the data read
- This creates a synchronization point between the producer (writer) and consumer (reader)

### System Scope
Using `scope="sys"` instead of the default `scope="gpu"` is critical because:
- The communication happens across different GPUs (ranks)
- GPU scope only guarantees visibility within a single GPU
- System scope ensures visibility across all GPUs in the system

## Files Changed
1. `examples/15_gemm_all_reduce_ring_based/benchmark.py`: Added initialization of locks and ring_buffer
2. `examples/15_gemm_all_reduce_ring_based/gemm_all_reduce_ring_based.py`: Updated synchronization primitives

## Testing
To test the fix, run:
```bash
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2
```

Expected output:
```
[Iris] [0/2] Validating...
[Iris] [1/2] Validating...
[Iris] [0/2] Final C validation passed.
[Iris] [1/2] Final C validation passed.
```

## Related Patterns
This fix aligns with similar synchronization patterns used in other examples:
- `examples/13_flash_decode/decode_kernels.py` uses `tl.atomic_cas` with acquire semantics for flag polling
- The use of system-scope atomics is standard practice for multi-GPU synchronization
