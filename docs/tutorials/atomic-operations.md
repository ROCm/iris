# Atomic Operations Tutorial

Learn how to use atomic operations in Iris for safe, concurrent access to shared memory across multiple GPUs. This tutorial covers the fundamental atomic operations and their practical applications.

## Overview

In this tutorial, you'll learn:
- What atomic operations are and why they're important
- How to use `iris.atomic_add()` and `iris.atomic_xchg()`
- Common patterns for atomic operations
- Best practices for performance and correctness
- Real-world examples and use cases

## Prerequisites

- ✅ Iris installed and working (see [Installation Guide](../getting-started/installation.md))
- ✅ Understanding of [Basic Operations](basic-operations.md)
- ✅ Access to at least 2 AMD GPUs

## What Are Atomic Operations?

Atomic operations are operations that complete entirely or not at all, without interruption from other threads or processes. In multi-GPU programming, they're essential for:

- **Race condition prevention**: Ensuring data consistency when multiple GPUs access the same memory
- **Lock-free programming**: Building concurrent data structures without traditional locks
- **Performance optimization**: Avoiding expensive synchronization barriers

## Core Atomic Operations

### 1. Atomic Add (`iris.atomic_add`)

Atomically adds a value to a memory location and returns the previous value.

```python
@triton.jit
def atomic_add_example(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomically add 1 to each element
    old_value = iris.atomic_add(
        buffer + offsets,           # Target memory location
        1,                          # Value to add
        0,                          # Source rank
        1,                          # Target rank
        heap_bases_ptr,             # Heap bases pointer
        mask=mask                   # Conditional mask
    )
    
    # old_value contains the previous value before addition
    return old_value
```

### 2. Atomic Exchange (`iris.atomic_xchg`)

Atomically exchanges a value with the current value at a memory location.

```python
@triton.jit
def atomic_xchg_example(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomically exchange with new value
    old_value = iris.atomic_xchg(
        buffer + offsets,           # Target memory location
        42.0,                       # New value to store
        0,                          # Source rank
        1,                          # Target rank
        heap_bases_ptr,             # Heap bases pointer
        mask=mask                   # Conditional mask
    )
    
    # old_value contains the previous value
    return old_value
```

## Tutorial: Building a Counter

Let's build a practical example that demonstrates atomic operations by creating a distributed counter.

### Step 1: Setup and Initialization

Create a new file `atomic_counter_tutorial.py`:

```python
import torch
import triton
import triton.language as tl
import iris

def setup_atomic_counter():
    """Initialize Iris and create a counter buffer."""
    # Initialize Iris with 1GB heap
    heap_size = 2**30
    iris_ctx = iris.iris(heap_size)
    rank = iris_ctx.get_rank()
    
    print(f"Rank {rank}: Iris initialized with {heap_size / 2**30:.1f}GB heap")
    
    # Create a single-element counter buffer
    counter = iris_ctx.zeros(1, dtype=torch.int32)
    
    # Initialize counter to 0 on all ranks
    counter.fill_(0)
    
    print(f"Rank {rank}: Counter initialized to {counter.item()}")
    
    return iris_ctx, counter
```

### Step 2: Atomic Counter Kernel

Add the atomic counter kernel:

```python
@triton.jit
def atomic_counter_kernel(counter, heap_bases_ptr, increment_value: tl.constexpr):
    """Kernel that atomically increments a counter."""
    # Get the counter pointer
    counter_ptr = counter
    
    # Atomically add the increment value
    old_value = iris.atomic_add(
        counter_ptr,
        increment_value,
        0,  # source_rank
        0,  # target_rank (same rank for this example)
        heap_bases_ptr
    )
    
    # old_value contains the previous counter value
    # The new value is old_value + increment_value
```

### Step 3: Multiple Ranks Incrementing

Create a function where multiple ranks increment the counter:

```python
def multiple_ranks_increment(iris_ctx, counter):
    """Demonstrate multiple ranks incrementing the same counter."""
    rank = iris_ctx.get_rank()
    
    # Each rank increments the counter by its rank + 1
    increment_value = rank + 1
    
    print(f"Rank {rank}: Will increment counter by {increment_value}")
    
    # Launch kernel to increment counter
    grid = lambda meta: (1,)  # Single block
    
    atomic_counter_kernel[grid](
        counter,
        iris_ctx.get_heap_bases(),
        increment_value
    )
    
    print(f"Rank {rank}: Kernel launched")
    
    # Synchronize all ranks
    iris_ctx.barrier()
    
    # Check final counter value
    final_value = counter.item()
    print(f"Rank {rank}: Final counter value: {final_value}")
    
    return final_value
```

### Step 4: Complete Example

Combine everything into a complete program:

```python
def main():
    print("=== Iris Atomic Operations Tutorial ===\n")
    
    # Step 1: Initialize
    iris_ctx, counter = setup_atomic_counter()
    
    # Step 2: Multiple ranks increment counter
    final_values = multiple_ranks_increment(iris_ctx, counter)
    
    # Step 3: Verification
    rank = iris_ctx.get_rank()
    
    # Expected: sum of all increments (0+1 + 1+1 + 2+1 + 3+1 = 8 for 4 ranks)
    expected_value = sum(range(iris_ctx.num_ranks)) + iris_ctx.num_ranks
    
    if rank == 0:
        actual_value = counter.item()
        if actual_value == expected_value:
            print(f"✅ SUCCESS: Counter value {actual_value} matches expected {expected_value}")
        else:
            print(f"❌ ERROR: Counter value {actual_value} doesn't match expected {expected_value}")
    
    iris_ctx.barrier()
    print(f"Rank {rank}: Tutorial completed!")

if __name__ == "__main__":
    main()
```

### Step 5: Run the Tutorial

Execute your program:

```bash
mpirun -np 4 python atomic_counter_tutorial.py
```

## Understanding the Output

You should see output similar to:

```
=== Iris Atomic Operations Tutorial ===

Rank 0: Iris initialized with 1.0GB heap
Rank 1: Iris initialized with 1.0GB heap
Rank 2: Iris initialized with 1.0GB heap
Rank 3: Iris initialized with 1.0GB heap
Rank 0: Counter initialized to 0
Rank 1: Counter initialized to 0
Rank 2: Counter initialized to 0
Rank 3: Counter initialized to 0
Rank 0: Will increment counter by 1
Rank 1: Will increment counter by 2
Rank 2: Will increment counter by 3
Rank 3: Will increment counter by 4
Rank 0: Kernel launched
Rank 1: Kernel launched
Rank 2: Kernel launched
Rank 3: Kernel launched
Rank 0: Final counter value: 10
Rank 1: Final counter value: 10
Rank 2: Final counter value: 10
Rank 3: Final counter value: 10
✅ SUCCESS: Counter value 10 matches expected 10
Rank 0: Tutorial completed!
Rank 1: Tutorial completed!
Rank 2: Tutorial completed!
Rank 3: Tutorial completed!
```

## Key Concepts Explained

### 1. **Atomicity**

Atomic operations ensure that the entire operation completes without interruption:
- **No partial updates**: The operation either fully succeeds or fully fails
- **Consistent state**: Memory is always in a valid state
- **Race condition free**: Multiple GPUs can safely access the same location

### 2. **Return Values**

Atomic operations return the **previous value** before the operation:
- **`atomic_add`**: Returns the value before addition
- **`atomic_xchg`**: Returns the value before exchange
- **Useful for**: Building lock-free data structures, implementing counters

### 3. **Synchronization**

Atomic operations provide implicit synchronization:
- **No barriers needed**: The operation itself ensures consistency
- **Efficient**: Avoids expensive global synchronization
- **Scalable**: Performance doesn't degrade with more GPUs

## Advanced Patterns

### 1. **Distributed Counter with Multiple Targets**

```python
@triton.jit
def distributed_counter_kernel(counter, heap_bases_ptr, target_rank: tl.constexpr):
    """Increment counter on a specific target rank."""
    counter_ptr = counter
    
    # Atomically increment counter on target_rank
    old_value = iris.atomic_add(
        counter_ptr,
        1,
        tl.program_id(0),  # Current workgroup
        target_rank,        # Target rank
        heap_bases_ptr
    )
    
    return old_value
```

### 2. **Atomic Compare-and-Swap Pattern**

```python
@triton.jit
def atomic_cas_pattern(buffer, heap_bases_ptr, expected: tl.constexpr, new_value: tl.constexpr):
    """Implement compare-and-swap using atomic operations."""
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Read current value
    current = iris.load(buffer + offsets, 0, 1, heap_bases_ptr, mask=mask)
    
    # Check if current value matches expected
    should_swap = current == expected
    
    # Only swap if condition is met
    if should_swap:
        old_value = iris.atomic_xchg(
            buffer + offsets,
            new_value,
            0, 1, heap_bases_ptr, mask=mask
        )
        return old_value
    else:
        return current
```

### 3. **Atomic Array Operations**

```python
@triton.jit
def atomic_array_operations(buffer, heap_bases_ptr):
    """Perform atomic operations on array elements."""
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomic add to even indices
    even_mask = offsets % 2 == 0
    iris.atomic_add(
        buffer + offsets,
        10,
        0, 1, heap_bases_ptr,
        mask=mask & even_mask
    )
    
    # Atomic exchange on odd indices
    odd_mask = offsets % 2 == 1
    iris.atomic_xchg(
        buffer + offsets,
        42,
        0, 1, heap_bases_ptr,
        mask=mask & odd_mask
    )
```

## Real-World Use Cases

### 1. **Histogram Construction**

```python
@triton.jit
def histogram_kernel(data, histogram, heap_bases_ptr):
    """Build histogram using atomic operations."""
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Get data values
    values = iris.load(data + offsets, 0, 1, heap_bases_ptr, mask=mask)
    
    # Convert to histogram bin indices
    bin_indices = tl.cast(values * 255, tl.int32)
    bin_indices = tl.clamp(bin_indices, 0, 255)
    
    # Atomically increment histogram bins
    iris.atomic_add(
        histogram + bin_indices,
        1,
        0, 1, heap_bases_ptr,
        mask=mask
    )
```

### 2. **Reduction Operations**

```python
@triton.jit
def atomic_reduction_kernel(data, result, heap_bases_ptr):
    """Perform reduction using atomic operations."""
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Load data
    values = iris.load(data + offsets, 0, 1, heap_bases_ptr, mask=mask)
    
    # Atomically add to result
    iris.atomic_add(
        result,
        values,
        0, 1, heap_bases_ptr,
        mask=mask
    )
```

### 3. **Lock-Free Queue**

```python
@triton.jit
def lock_free_queue_push(queue, item, heap_bases_ptr):
    """Push item to lock-free queue using atomic operations."""
    # Get current tail pointer
    tail = iris.load(queue + 0, 0, 1, heap_bases_ptr)  # tail pointer
    
    # Atomically increment tail
    new_tail = iris.atomic_add(
        queue + 0,
        1,
        0, 1, heap_bases_ptr
    )
    
    # Store item at new tail position
    iris.store(
        queue + new_tail + 1,  # +1 to skip tail pointer
        item,
        0, 1, heap_bases_ptr
    )
```

## Performance Considerations

### 1. **Memory Access Patterns**

- **Coalesced access**: Use contiguous memory access for better performance
- **Cache locality**: Group atomic operations on nearby memory locations
- **Bank conflicts**: Avoid multiple threads accessing the same memory bank

### 2. **Operation Batching**

```python
@triton.jit
def batched_atomic_operations(buffer, heap_bases_ptr):
    """Batch multiple atomic operations for better performance."""
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Batch multiple operations
    for i in range(4):
        batch_offsets = offsets + i * 256
        iris.atomic_add(
            buffer + batch_offsets,
            i + 1,
            0, 1, heap_bases_ptr,
            mask=mask
        )
```

### 3. **Conditional Operations**

```python
@triton.jit
def conditional_atomic_operations(buffer, heap_bases_ptr, threshold: tl.constexpr):
    """Use masks for conditional atomic operations."""
    offsets = tl.arange(0, 1024)
    
    # Only perform operations on values above threshold
    values = iris.load(buffer + offsets, 0, 1, heap_bases_ptr)
    mask = values > threshold
    
    # Conditional atomic operation
    iris.atomic_add(
        buffer + offsets,
        1,
        0, 1, heap_bases_ptr,
        mask=mask
    )
```

## Best Practices

### 1. **Always Use Masks**

```python
# Good: Use masks for bounds checking
offsets = tl.arange(0, 1024)
mask = offsets < buffer_size
iris.atomic_add(buffer + offsets, 1, 0, 1, heap_bases_ptr, mask=mask)

# Bad: No bounds checking
iris.atomic_add(buffer + offsets, 1, 0, 1, heap_bases_ptr)
```

### 2. **Handle Return Values**

```python
# Good: Use return value for further computation
old_value = iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)
new_value = old_value + 1

# Bad: Ignore return value
iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)
```

### 3. **Minimize Atomic Operations**

```python
# Good: Batch operations
for i in range(10):
    iris.atomic_add(buffer + i, 1, 0, 1, heap_bases_ptr)

# Better: Single operation with larger increment
iris.atomic_add(buffer, 10, 0, 1, heap_bases_ptr)
```

## Common Pitfalls

### 1. **Missing Synchronization**

```python
# Bad: Race condition
iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)
result = iris.load(buffer, 1, 0, heap_bases_ptr)  # May read old value

# Good: Proper synchronization
iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)
iris_ctx.barrier()  # Ensure operation completes
result = iris.load(buffer, 1, 0, heap_bases_ptr)
```

### 2. **Incorrect Rank Parameters**

```python
# Bad: Wrong rank parameters
iris.atomic_add(buffer, 1, 1, 0, heap_bases_ptr)  # source_rank > target_rank

# Good: Correct rank parameters
iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)  # source_rank < target_rank
```

### 3. **Memory Alignment Issues**

```python
# Bad: Unaligned access
iris.atomic_add(buffer + 1, 1, 0, 1, heap_bases_ptr)  # +1 may cause alignment issues

# Good: Aligned access
iris.atomic_add(buffer, 1, 0, 1, heap_bases_ptr)  # Start from beginning
```

## Testing Atomic Operations

### 1. **Correctness Testing**

```python
def test_atomic_correctness():
    """Test that atomic operations produce correct results."""
    iris_ctx = iris.iris(heap_size=2**30)
    
    # Create test buffer
    buffer = iris_ctx.zeros(1, dtype=torch.int32)
    buffer.fill_(0)
    
    # Launch multiple kernels that increment
    grid = lambda meta: (4,)  # 4 workgroups
    
    for _ in range(10):
        atomic_counter_kernel[grid](buffer, iris_ctx.get_heap_bases(), 1)
    
    iris_ctx.barrier()
    
    # Verify result
    final_value = buffer.item()
    expected_value = 40  # 4 workgroups × 10 iterations
    
    assert final_value == expected_value, f"Expected {expected_value}, got {final_value}"
    print("✅ Atomic correctness test passed!")
```

### 2. **Performance Testing**

```python
import time

def test_atomic_performance():
    """Test performance of atomic operations."""
    iris_ctx = iris.iris(heap_size=2**30)
    
    buffer = iris_ctx.zeros(1024, dtype=torch.int32)
    buffer.fill_(0)
    
    # Warm up
    grid = lambda meta: (16,)
    atomic_counter_kernel[grid](buffer, iris_ctx.get_heap_bases(), 1)
    iris_ctx.barrier()
    
    # Profile
    start_time = time.time()
    
    for _ in range(100):
        atomic_counter_kernel[grid](buffer, iris_ctx.get_heap_bases(), 1)
    
    iris_ctx.barrier()
    end_time = time.time()
    
    total_time = end_time - start_time
    operations_per_second = 100 * 1024 / total_time
    
    print(f"Atomic operations: {operations_per_second:.0f} ops/sec")
```

## Next Steps

Now that you understand atomic operations:

1. **Practice**: Modify the counter example with different increment values
2. **Explore**: Try building more complex atomic data structures
3. **Optimize**: Profile and optimize your atomic operation patterns
4. **Study**: Learn about [Message Passing](message-passing.md) for more communication patterns

## Troubleshooting

### Common Issues

- **"Atomic operation failed"**: Check rank parameters and memory alignment
- **"Incorrect results"**: Verify synchronization and barrier placement
- **"Performance issues"**: Profile and optimize memory access patterns

### Debugging Tips

1. **Use logging**: Enable debug logging to track operation execution
2. **Validate results**: Add correctness checks to verify atomic behavior
3. **Profile performance**: Measure operation timing to identify bottlenecks

## Need Help?

- Check the [Debugging Guide](../how-to/debug-common-issues.md)
- Start a discussion in [GitHub Discussions](https://github.com/ROCm/iris/discussions)
- Open an issue for bugs or problems

---

**Great job! You've mastered atomic operations in Iris. Ready for more advanced communication patterns? Continue to [Message Passing](message-passing.md)!**
