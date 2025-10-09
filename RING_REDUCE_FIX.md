# Ring-Based All-Reduce Validation Fix

## Problem
The ring-based all-reduce implementation was failing validation with errors like:
```
[Iris] [0/2] Max absolute difference: 605.5
[Iris] [0/2] Mismatch at index (4641, 1056): C=45.90625, expected=55.25
```

Additionally, when scaling to 8 GPUs, the implementation would hang/deadlock.

## Root Causes

### Issue 1: Improper Memory Ordering (Fixed in initial commits)
The lock-based synchronization used regular memory operations without proper acquire-release semantics, leading to data races and cache coherence issues.

### Issue 2: Race Condition in Ring Protocol (Fixed in this commit)
The ring all-reduce algorithm had a critical race condition when scaling beyond 2 GPUs:

**The Problem**: Without waiting for the next rank to finish consuming data from the previous step, a rank could overwrite the ring buffer while the next rank was still reading it.

**Example with 8 GPUs**:
- Step 0: Rank 0 writes data to Rank 1's buffer and sets lock to 1
- Step 0: Rank 1 eventually reads the data and resets lock to 0
- Step 1: Rank 0 immediately writes NEW data to Rank 1's buffer
- **RACE**: If Rank 0 writes before Rank 1 finishes reading from Step 0, data corruption occurs

This race condition became more likely with more GPUs because:
- With 2 GPUs: Only 1 step (world_size - 1), less time for race
- With 8 GPUs: 7 steps, much more opportunity for the race to manifest as a hang/deadlock

## Solution

### Phase 1: Proper Atomic Operations (Commits 1-3)
Replaced lock operations with proper atomic primitives:

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

### Phase 2: Fix Ring Protocol Race (This commit)
Added a wait-before-write check at the beginning of each ring step:

```python
# Before writing to next rank's buffer, wait for it to finish reading from previous step
while iris.atomic_cas(locks + tile_id, 0, 0, cur_rank, next_rank, heap_bases, sem="acquire", scope="sys") != 0:
    pass
```

This ensures:
- The next rank has reset its lock to 0 (finished consuming previous data)
- The ring buffer is ready to receive new data
- No data corruption from overlapping writes/reads

**Flow for each rank in each step**:
1. Wait for next rank's lock to be 0 (ready for new data)
2. Write data to next rank's buffer
3. Set next rank's lock to 1 (data ready)
4. Wait for own lock to be 1 (data from prev rank ready)
5. Read data from own buffer
6. Reset own lock to 0 (ready for next step)

This creates proper producer-consumer synchronization in the ring.

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

### Ring Protocol
The ring protocol now properly handles the circular dependency:
- Each rank waits for both prev (to receive) AND next (to send)
- This prevents buffer overwrites and ensures data integrity
- The double-barrier pattern (wait before write, signal after write) is standard for producer-consumer

## Files Changed
1. `examples/15_gemm_all_reduce_ring_based/benchmark.py`: Added initialization of locks and ring_buffer
2. `examples/15_gemm_all_reduce_ring_based/gemm_all_reduce_ring_based.py`: Updated synchronization primitives and fixed ring protocol

## Testing
To test the fix, run:
```bash
# Test with 2 GPUs
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 2

# Test with 8 GPUs (previously would hang)
python examples/15_gemm_all_reduce_ring_based/benchmark.py --validate --num_ranks 8
```

Expected output:
```
[Iris] [0/N] Validating...
[Iris] [1/N] Validating...
...
[Iris] [0/N] Final C validation passed.
[Iris] [1/N] Final C validation passed.
```

## Related Patterns
This fix aligns with similar synchronization patterns used in other examples:
- `examples/13_flash_decode/decode_kernels.py` uses `tl.atomic_cas` with acquire semantics for flag polling
- The use of system-scope atomics is standard practice for multi-GPU synchronization
- The wait-before-write pattern is a standard producer-consumer synchronization technique
