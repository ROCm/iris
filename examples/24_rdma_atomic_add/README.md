# RDMA Atomic Add Example

This example demonstrates RDMA atomic fetch-and-add operations using Iris RDMA.

## Overview

In this example:
- **Rank 0** maintains a shared counter in its symmetric heap
- **All ranks** (0 through N-1) atomically increment rank 0's counter
- Each rank adds its own rank number + 1 (i.e., rank 0 adds 1, rank 1 adds 2, etc.)
- The atomic operation returns the old value before incrementing
- Rank 0 verifies the final sum

## Key Concepts

### Atomic Fetch-and-Add
```python
iris_rdma.atomic_add(
    result_ptr,      # Local buffer to store old value
    counter_ptr,     # Remote counter location (symmetric heap)
    increment,       # Value to add
    dst_rank,        # Which rank owns the counter
    device_ctx,      # Device context
    mask,            # Triton mask
)
```

- **Atomic**: Operation is indivisible - no race conditions
- **Fetch**: Returns the original value before the add
- **Symmetric Heap**: All ranks use same offset, automatically translated

### Expected Result

For N ranks, each rank i adds (i+1):
```
Final counter = 1 + 2 + 3 + ... + N = N × (N+1) / 2
```

For 2 ranks: 1 + 2 = 3  
For 4 ranks: 1 + 2 + 3 + 4 = 10  
For 8 ranks: 1 + 2 + 3 + ... + 8 = 36

## Running the Example

### With 2 ranks:
```bash
torchrun --nproc_per_node=2 examples/24_rdma_atomic_add/rdma_atomic_add.py
```

### With 4 ranks:
```bash
torchrun --nproc_per_node=4 examples/24_rdma_atomic_add/rdma_atomic_add.py
```

### With debug logging:
```bash
IRIS_LOG_LEVEL=DEBUG torchrun --nproc_per_node=2 examples/24_rdma_atomic_add/rdma_atomic_add.py
```

## Expected Output

```
[Rank 0/2] Initialized on cuda:0
[Rank 1/2] Initialized on cuda:1
[Rank 0] Iris RDMA initialized
[Rank 1] Iris RDMA initialized

[Rank 0] === Testing Atomic Add ===
[Rank 0] Initial counter value: 0
[Rank 0] Waiting for other ranks to increment...

[Rank 0] Atomically adding 1 to rank 0's counter...
[Rank 1] Atomically adding 2 to rank 0's counter...
[Rank 0] Atomic add completed. Old value was: 0
[Rank 1] Atomic add completed. Old value was: 1

[Rank 0] === Verification ===
[Rank 0] Final counter value: 3
[Rank 0] Expected value: 3
[Rank 0] Each rank added: [1, 2]

============================================================
[Rank 0] SUCCESS! Atomic operations worked correctly!
============================================================
```

## How It Works

1. **Initialization**: All ranks initialize Iris RDMA with symmetric heaps
2. **Buffer Allocation**: Each rank allocates counter/result buffers at same offset
3. **Atomic Operations**: 
   - Ranks launch Triton kernels that call `iris_rdma.atomic_add()`
   - Triton kernel enqueues atomic operation to device queue
   - CPU proxy thread dequeues and executes RDMA atomic via InfiniBand
   - Original value is returned to result buffer
4. **Verification**: Rank 0 checks that sum equals expected value

## Key Features Demonstrated

- ✅ **RDMA Atomics**: Hardware-level atomic operations over InfiniBand
- ✅ **Symmetric Heap**: Automatic address translation between ranks
- ✅ **Fetch-and-Add**: Returns old value atomically
- ✅ **GPU-initiated**: Triton kernel directly initiates RDMA operations
- ✅ **Zero-copy**: No intermediate buffers or CPU involvement for data path

## Notes

- Atomic operations require **64-bit integers** (`torch.int64` or `torch.uint64`)
- 32-bit atomics are also supported by changing the size parameter
- Operations are **synchronous** - kernel waits for completion before returning
- All ranks must allocate buffers at the **same symmetric heap offset**

## Troubleshooting

**Counter value is wrong:**
- Check that all ranks successfully performed atomic operations
- Verify InfiniBand connection is working
- Enable debug logging to see RDMA operations

**Atomics not supported error:**
- Ensure your InfiniBand HCA supports atomic operations
- Most modern Mellanox/NVIDIA and Broadcom NICs support this

**Hang on barrier:**
- Check that all ranks are running
- Verify NCCL is properly configured

