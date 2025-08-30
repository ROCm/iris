# Basic Operations Tutorial

Learn the fundamental operations in Iris: loading and storing data between GPUs. This tutorial covers the core Remote Memory Access (RMA) operations that form the foundation of multi-GPU programming with Iris.

## Overview

In this tutorial, you'll learn:
- How to initialize Iris and create symmetric heaps
- Basic load and store operations between GPUs
- Memory allocation and management
- Synchronization using barriers
- Best practices for basic operations

## Prerequisites

- ✅ Iris installed and working (see [Installation Guide](../getting-started/installation.md))
- ✅ Access to at least 2 AMD GPUs
- ✅ Understanding of basic Triton concepts

## Tutorial: Basic Load/Store Operations

### Step 1: Setup and Initialization

Create a new file `basic_operations.py`:

```python
import torch
import triton
import triton.language as tl
import iris

def basic_load_store_example():
    # Initialize Iris with a 1GB symmetric heap
    heap_size = 2**30
    iris_ctx = iris.iris(heap_size)
    rank = iris_ctx.get_rank()
    
    print(f"Rank {rank}: Iris initialized with {heap_size / 2**30:.1f}GB heap")
    
    # Allocate buffers on all ranks
    buffer_size = 1024
    local_buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)
    
    # Initialize local buffer with rank-specific values
    local_buffer.fill_(rank + 1.0)
    
    print(f"Rank {rank}: Local buffer initialized with value {rank + 1.0}")
    
    return iris_ctx, local_buffer, buffer_size
```

### Step 2: Store Operation

Add the store operation to your file:

```python
@triton.jit
def store_kernel(buffer, buffer_size: tl.constexpr, block_size: tl.constexpr, 
                 heap_bases_ptr, source_rank: tl.constexpr, target_rank: tl.constexpr):
    # Get block information
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    
    # Guard against out-of-bounds access
    mask = offsets < buffer_size
    
    # Store data from source_rank to target_rank's buffer
    iris.store(buffer + offsets, 
               source_rank + 100.0,  # Store rank + 100 as the value
               source_rank, target_rank,
               heap_bases_ptr, mask=mask)

def perform_store_operations(iris_ctx, local_buffer, buffer_size):
    rank = iris_ctx.get_rank()
    
    # Only rank 0 performs store operations
    if rank == 0:
        print("Rank 0: Performing store operations...")
        
        block_size = 256
        grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)
        
        # Store to rank 1's buffer
        store_kernel[grid](
            local_buffer,
            buffer_size,
            block_size,
            iris_ctx.get_heap_bases(),
            0, 1  # source_rank=0, target_rank=1
        )
        
        print("Rank 0: Store operations completed")
    
    # Synchronize all ranks
    iris_ctx.barrier()
```

### Step 3: Load Operation

Add the load operation:

```python
def perform_load_operations(iris_ctx, local_buffer, buffer_size):
    rank = iris_ctx.get_rank()
    
    # Rank 1 loads data from rank 0's buffer
    if rank == 1:
        print("Rank 1: Loading data from rank 0...")
        
        # Load data that was stored by rank 0
        received_data = iris.load(local_buffer, 
                                 source_rank=0, target_rank=1,
                                 heap_bases_ptr=iris_ctx.get_heap_bases())
        
        print(f"Rank 1: Received data: {received_data[:5]}...")  # Show first 5 elements
        print(f"Rank 1: Data shape: {received_data.shape}")
        print(f"Rank 1: Data type: {received_data.dtype}")
    
    iris_ctx.barrier()
```

### Step 4: Complete Example

Combine everything into a complete program:

```python
def main():
    print("=== Iris Basic Operations Tutorial ===\n")
    
    # Step 1: Initialize
    iris_ctx, local_buffer, buffer_size = basic_load_store_example()
    
    # Step 2: Store operations
    perform_store_operations(iris_ctx, local_buffer, buffer_size)
    
    # Step 3: Load operations
    perform_load_operations(iris_ctx, local_buffer, buffer_size)
    
    # Step 4: Verification
    rank = iris_ctx.get_rank()
    if rank == 1:
        # Verify the received data
        received_data = iris.load(local_buffer, 
                                 source_rank=0, target_rank=1,
                                 heap_bases_ptr=iris_ctx.get_heap_bases())
        
        expected_value = 100.0  # rank 0 + 100
        if torch.allclose(received_data, expected_value):
            print("✅ SUCCESS: Data transfer verified!")
        else:
            print("❌ ERROR: Data transfer failed!")
    
    iris_ctx.barrier()
    print(f"Rank {rank}: Tutorial completed!")

if __name__ == "__main__":
    main()
```

### Step 5: Run the Tutorial

Execute your program:

```bash
mpirun -np 2 python basic_operations.py
```

## Understanding the Output

You should see output similar to:

```
=== Iris Basic Operations Tutorial ===

Rank 0: Iris initialized with 1.0GB heap
Rank 1: Iris initialized with 1.0GB heap
Rank 0: Local buffer initialized with value 1.0
Rank 1: Local buffer initialized with value 2.0
Rank 0: Performing store operations...
Rank 0: Store operations completed
Rank 1: Loading data from rank 0...
Rank 1: Received data: tensor([100., 100., 100., 100., 100.])...
Rank 1: Data shape: torch.Size([1024])
Rank 1: Data type: torch.float32
✅ SUCCESS: Data transfer verified!
Rank 0: Tutorial completed!
Rank 1: Tutorial completed!
```

## Key Concepts Explained

### 1. Symmetric Heap

The symmetric heap is a shared memory space accessible from all GPUs:
- **Same size**: All ranks allocate the same heap size
- **Same addresses**: Memory addresses are consistent across ranks
- **Shared access**: Any rank can access any other rank's memory

### 2. Load and Store Operations

- **`iris.store()`**: Writes data from source rank to target rank's memory
- **`iris.load()`**: Reads data from source rank's memory to target rank
- **Remote operations**: Operations happen between different GPUs

### 3. Synchronization

- **`iris_ctx.barrier()`**: Ensures all ranks complete operations before proceeding
- **Critical for correctness**: Prevents race conditions and ensures data consistency

### 4. Memory Management

- **`iris_ctx.zeros()`**: Allocates zero-initialized tensors in the symmetric heap
- **Automatic cleanup**: Memory is managed by the Iris context

## Advanced Patterns

### Bidirectional Communication

```python
# Both ranks can store to each other
if rank == 0:
    # Store to rank 1
    iris.store(buffer, data, 0, 1, heap_bases_ptr)
elif rank == 1:
    # Store to rank 0
    iris.store(buffer, data, 1, 0, heap_bases_ptr)

iris_ctx.barrier()
```

### Multi-rank Operations

```python
# Store to multiple ranks
for target_rank in range(1, num_ranks):
    iris.store(buffer, data, rank, target_rank, heap_bases_ptr)

iris_ctx.barrier()
```

## Best Practices

1. **Always use barriers**: Ensure operations complete before proceeding
2. **Check buffer sizes**: Prevent out-of-bounds memory access
3. **Use meaningful variable names**: Make your code readable and maintainable
4. **Handle errors gracefully**: Check return values and handle failures
5. **Profile performance**: Monitor memory usage and operation timing

## Common Pitfalls

1. **Missing barriers**: Can cause race conditions and incorrect results
2. **Buffer size mismatches**: Ensure all ranks allocate the same buffer sizes
3. **Rank confusion**: Double-check source and target rank parameters
4. **Memory leaks**: Don't forget to clean up large allocations

## Next Steps

Now that you understand basic operations:

1. **Try variations**: Modify buffer sizes, data types, and values
2. **Add more ranks**: Experiment with 4 or 8 GPUs
3. **Explore atomics**: Learn about [Atomic Operations](atomic-operations.md)
4. **Study patterns**: Check out the [Examples](../reference/examples.md) directory

## Troubleshooting

- **"CUDA out of memory"**: Reduce heap size or buffer size
- **"Rank out of bounds"**: Ensure MPI rank count matches GPU count
- **"Import errors"**: Verify Iris installation and dependencies

## Need Help?

- Check the [Troubleshooting](../how-to/debug-common-issues.md) guide
- Start a discussion in [GitHub Discussions](https://github.com/ROCm/iris/discussions)
- Open an issue for bugs or problems

---

**Great job! You've mastered basic operations in Iris. Ready for more advanced patterns? Continue to [Atomic Operations](atomic-operations.md)!**
