# Programming Model

Iris is an open-source Triton-based framework for Remote Memory Access (RMA) operations written in only a few hundred lines of code. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming.

![Iris Model](../../images/iris-model.png)

## Core Design Principles

### 1. **Designed by Experts, Built for Scale**
- Written from scratch by GPU and distributed computing experts
- Minimal dependencies: only Triton, PyTorch, HIP runtime and mpi4py (for initialization)
- No external frameworks or heavyweight runtimes beyond core stack

### 2. **Clean Abstractions**
- Full Symmetric Heap implementation in Python
- Pythonic PyTorch-like host APIs for tensor allocation and construction
- Pythonic Triton-style device APIs for load, store, and atomic operations

### 3. **Communication + Computation**
- Device-side collective operations: broadcast, scatter, reduce, etc.
- Lock variants for communication and computation overlap
- Fine-grained GEMM + communication overlap via workgroup specialization

### 4. **Scalable by Design**
- Full scale-up (multi-GPU node) support
- Scale-out (multi-node) in progress

> **Note**: Remote Direct Memory Access (RDMA) is work-in-progress.

## Core APIs

### Simple `load` & `store` Operations

Iris provides intuitive APIs for remote memory access that feel natural to Triton developers:

#### `iris.load()` API

```python
@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
    """
    Loads a value from the specified rank's memory location.

    This function performs a memory read operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and loading
    data from the target memory location. If the from_rank and to_rank are the same,
    this function performs a local load operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType):
            Pointer in the from_rank's address space that will be translated to
            the to_rank's address space. Must be the current rank where the
            pointer is local.
        to_rank (int): The rank ID to which the pointer will be translated.
            Must be the current rank where the pointer is local.
        from_rank (int): The rank ID from which to read the data.
        heap_bases (triton.PointerType): Array containing the heap base
            addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not
            load the data at address pointer[idx]. Defaults to None.

    Returns:
        Block: The loaded value from the target memory location.
    """
```

#### `iris.store()` API

```python
@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
    """
    Writes data to the specified rank's memory location.

    This function performs a memory write operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and storing
    the provided data to the target memory location. If the from_rank and to_rank
    are the same, this function performs a local store operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType):
            Pointer in the from_rank's address space that will be translated to
            the to_rank's address space. Must be the current rank where the
            pointer is local.
        value (Block): The tensor of elements to be stored.
        from_rank (int): The rank ID from which the pointer originates.
            Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base
            addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not
            store the data at address pointer[idx]. Defaults to None.

    Returns:
        None
    """
```

## Symmetric Heap Architecture

![Iris Symmetric Heap](../../images/heap.png)

The Symmetric Heap is a Partitioned Global Address Space (PGAS) abstraction that enables seamless remote memory access.

### Key Concepts

**Symmetric Heap**: A shared memory space where all GPUs can access each other's memory using consistent addressing.

**Address Translation**: The key insight is that you can know the remote address of any symmetric variable with two offsets:
1. **Heap Base Offset**: Offset of target process' heap base in its virtual address space
2. **Variable Offset**: Offset of the variable within the symmetric heap

### Memory Layout

```
Rank 0: [Heap Base 0] + [Variable Offset] = Remote Address
Rank 1: [Heap Base 1] + [Variable Offset] = Remote Address
Rank 2: [Heap Base 2] + [Variable Offset] = Remote Address
...
```

### Benefits

- **Consistent Addressing**: Same variable has the same offset across all ranks
- **Efficient Access**: Direct memory access without message passing overhead
- **Scalable**: Works with any number of GPUs
- **Simple**: No complex routing or addressing schemes

## Programming Patterns

### 1. **Basic Remote Memory Access**

```python
@triton.jit
def basic_remote_access(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Store data from rank 0 to rank 1
    iris.store(buffer + offsets, 42.0, 0, 1, heap_bases_ptr)

    # Load data from rank 1 to rank 0
    data = iris.load(buffer + offsets, 0, 1, heap_bases_ptr)
```

### 2. **Collective Operations**

```python
@triton.jit
def all_gather_pattern(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Each rank stores its data to all other ranks
    for target_rank in range(4):  # Assuming 4 ranks
        iris.store(buffer + offsets, local_data, rank, target_rank, heap_bases_ptr)
```

### 3. **Communication Overlap**

```python
@triton.jit
def overlap_computation_communication(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Start communication
    iris.store(buffer + offsets, data, rank, (rank + 1) % 4, heap_bases_ptr)

    # Overlap with computation
    result = tl.where(offsets < 512,
                      buffer[offsets] * 2,
                      buffer[offsets] + 1)

    # Continue with more communication
    iris.store(buffer + offsets + 512, result, rank, (rank + 2) % 4, heap_bases_ptr)
```

## Memory Management

### Tensor Allocation

Iris provides PyTorch-like tensor creation APIs that automatically allocate memory in the symmetric heap:

```python
# Create tensors in symmetric heap
iris_ctx = iris.iris(heap_size=2**30)  # 1GB heap

# Allocate tensors
zeros_tensor = iris_ctx.zeros(1024, 1024, dtype=torch.float32)
ones_tensor = iris_ctx.ones(512, 512, dtype=torch.float16)
random_tensor = iris_ctx.randn(256, 256, dtype=torch.float32)
```

### Memory Layout

```
Symmetric Heap Layout:
[Rank 0 Data] [Rank 1 Data] [Rank 2 Data] ... [Rank N Data]
     ↑              ↑              ↑              ↑
  Base 0        Base 1        Base 2        Base N
```

### Memory Alignment

Iris automatically handles memory alignment for optimal performance:
- **Default alignment**: 1024 bytes
- **Automatic padding**: Ensures efficient memory access patterns
- **Configurable**: Can be adjusted based on hardware requirements

## Synchronization

### Barriers

```python
# Synchronize all ranks
iris_ctx.barrier()

# Ensure all operations complete before proceeding
if rank == 0:
    iris.store(buffer, data, 0, 1, heap_bases_ptr)

iris_ctx.barrier()  # Wait for store to complete

if rank == 1:
    result = iris.load(buffer, 1, 0, heap_bases_ptr)
```

### Locks and Atomic Operations

```python
@triton.jit
def atomic_counter(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Atomically increment counter
    old_value = iris.atomic_add(buffer + offsets, 1, rank, 0, heap_bases_ptr)

    # Use old_value for further computation
    result = old_value * 2
```

## Performance Considerations

### 1. **Memory Access Patterns**

- **Coalesced Access**: Use contiguous memory access patterns
- **Block Size**: Choose appropriate block sizes for your GPU
- **Grid Size**: Optimize grid dimensions for maximum occupancy

### 2. **Communication Optimization**

- **Batch Operations**: Group related operations to minimize overhead
- **Overlap**: Overlap computation and communication when possible
- **Reduce Barriers**: Minimize synchronization points

### 3. **Memory Efficiency**

- **Heap Size**: Choose appropriate heap size for your workload
- **Buffer Reuse**: Reuse buffers when possible
- **Data Types**: Use appropriate data types for precision vs. memory trade-offs

## Advanced Patterns

### 1. **Workgroup Specialization**

```python
@triton.jit
def specialized_workgroup(buffer, heap_bases_ptr, workgroup_id):
    if workgroup_id == 0:
        # Communication workgroup
        iris.store(buffer, data, rank, (rank + 1) % 4, heap_bases_ptr)
    elif workgroup_id == 1:
        # Computation workgroup
        result = buffer * 2 + 1
    else:
        # Mixed workgroup
        iris.store(buffer, result, rank, (rank + 2) % 4, heap_bases_ptr)
```

### 2. **Producer-Consumer Pattern**

```python
@triton.jit
def producer_consumer(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Producer: generate data
    data = offsets * 2 + 1

    # Store to consumer ranks
    for consumer_rank in range(1, 4):
        iris.store(buffer + offsets, data, rank, consumer_rank, heap_bases_ptr)
```

### 3. **Bulk Synchronous Pattern**

```python
@triton.jit
def bulk_synchronous(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)

    # Phase 1: Communication
    iris.store(buffer + offsets, data, rank, (rank + 1) % 4, heap_bases_ptr)

    # Phase 2: Computation (after barrier)
    result = buffer * 2

    # Phase 3: More communication
    iris.store(buffer + offsets, result, rank, (rank + 2) % 4, heap_bases_ptr)
```

## Best Practices

### 1. **Always Use Barriers Appropriately**

```python
# Good: Proper synchronization
iris.store(buffer, data, 0, 1, heap_bases_ptr)
iris_ctx.barrier()  # Wait for store to complete
result = iris.load(buffer, 1, 0, heap_bases_ptr)

# Bad: Race condition
iris.store(buffer, data, 0, 1, heap_bases_ptr)
result = iris.load(buffer, 1, 0, heap_bases_ptr)  # May read old data
```

### 2. **Use Masks for Conditional Operations**

```python
@triton.jit
def conditional_operations(buffer, heap_bases_ptr, size):
    offsets = tl.arange(0, 1024)
    mask = offsets < size  # Only operate on valid indices

    iris.store(buffer + offsets, data, rank, target_rank, heap_bases_ptr, mask=mask)
```

### 3. **Profile and Optimize**

```python
import time

# Time your operations
start_time = time.time()
kernel[grid](buffer, heap_bases_ptr)
iris_ctx.barrier()
end_time = time.time()

print(f"Operation took {end_time - start_time:.4f} seconds")
```

## Next Steps

Now that you understand the programming model:

1. **Try the examples**: Run the [Examples](../reference/examples.md) to see these patterns in action
2. **Explore tutorials**: Follow the [Tutorials](../tutorials/basic-operations.md) to learn step-by-step
3. **Build your own**: Start with simple patterns and gradually increase complexity
4. **Join the community**: Ask questions in GitHub Discussions

---

**Ready to start coding? Check out the [Quick Start Guide](../getting-started/quick-start.md) to run your first Iris program!**
