# API Reference

This document provides a comprehensive reference for all Iris APIs, including the main Iris class, memory management functions, and device-side operations.

## Core Classes

### Iris

The main Iris class that manages multi-GPU communication and memory allocation.

```python
class iris.iris(heap_size=1 << 30)
```

**Parameters:**
- `heap_size` (int): Size of the symmetric heap in bytes. Default: 1GB (2^30 bytes)

**Attributes:**
- `comm`: MPI communicator
- `num_ranks` (int): Total number of MPI ranks
- `cur_rank` (int): Current rank ID
- `gpu_id` (int): GPU device ID for this rank
- `heap_size` (int): Size of the symmetric heap
- `device` (str): CUDA device string (e.g., "cuda:0")

**Example:**
```python
import iris

# Initialize Iris with 2GB heap
iris_ctx = iris.iris(heap_size=2**31)
print(f"Rank {iris_ctx.cur_rank} of {iris_ctx.num_ranks}")
```

## Memory Management

### Tensor Creation

#### `zeros(*size, dtype=torch.int, device=None, requires_grad=False, **kwargs)`

Creates a tensor filled with zeros in the symmetric heap.

**Parameters:**
- `*size` (int...): Variable number of integers defining tensor shape
- `dtype` (torch.dtype): Data type of the tensor. Default: `torch.int`
- `device` (str): Device for the tensor. Default: Iris device
- `requires_grad` (bool): Whether to track gradients. Default: `False`

**Returns:**
- `torch.Tensor`: Zero-initialized tensor in symmetric heap

**Example:**
```python
# Create 1024x1024 float32 tensor
tensor = iris_ctx.zeros(1024, 1024, dtype=torch.float32)

# Create tensor with specific shape
tensor = iris_ctx.zeros((512, 256), dtype=torch.float16)
```

#### `zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

Creates a tensor with the same size as input, filled with zeros.

**Parameters:**
- `input` (torch.Tensor): Input tensor whose size determines output size
- `dtype` (torch.dtype): Data type. Default: same as input
- `device` (str): Device. Default: Iris device
- `requires_grad` (bool): Whether to track gradients. Default: `False`
- `memory_format` (torch.memory_format): Memory format. Default: `torch.preserve_format`

**Returns:**
- `torch.Tensor`: Zero-initialized tensor with same size as input

**Example:**
```python
reference = torch.randn(100, 200, device="cuda")
zeros_tensor = iris_ctx.zeros_like(reference, dtype=torch.float32)
```

#### `ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`

Creates a tensor filled with ones.

**Parameters:**
- `*size` (int...): Variable number of integers defining tensor shape
- `out` (torch.Tensor, optional): Output tensor
- `dtype` (torch.dtype): Data type. Default: global default dtype
- `device` (str): Device. Default: Iris device
- `requires_grad` (bool): Whether to track gradients. Default: `False`

**Returns:**
- `torch.Tensor`: Tensor filled with ones

**Example:**
```python
# Create 3x3 tensor of ones
ones_tensor = iris_ctx.ones(3, 3, dtype=torch.float32)
```

#### `full(size, fill_value, dtype=torch.int)`

Creates a tensor filled with a specific value.

**Parameters:**
- `size` (tuple): Tensor shape
- `fill_value` (scalar): Value to fill the tensor with
- `dtype` (torch.dtype): Data type. Default: `torch.int`

**Returns:**
- `torch.Tensor`: Tensor filled with specified value

**Example:**
```python
# Create 5x5 tensor filled with 3.14
pi_tensor = iris_ctx.full((5, 5), 3.14, dtype=torch.float32)
```

#### `empty(size, dtype=torch.float)`

Creates an uninitialized tensor.

**Parameters:**
- `size` (tuple): Tensor shape
- `dtype` (torch.dtype): Data type. Default: `torch.float`

**Returns:**
- `torch.Tensor`: Uninitialized tensor

**Example:**
```python
# Create empty tensor for later initialization
empty_tensor = iris_ctx.empty((100, 100), dtype=torch.float32)
```

### Random Tensor Generation

#### `randn(*size, generator=None, dtype=torch.float, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)`

Creates a tensor with random values from normal distribution.

**Parameters:**
- `*size` (int...): Variable number of integers defining tensor shape
- `generator` (torch.Generator, optional): Random number generator
- `dtype` (torch.dtype): Data type. Default: `torch.float`
- `device` (str): Device. Default: Iris device
- `requires_grad` (bool): Whether to track gradients. Default: `False`

**Returns:**
- `torch.Tensor`: Random tensor with normal distribution

**Example:**
```python
# Create 1000x1000 random tensor
random_tensor = iris_ctx.randn(1000, 1000, dtype=torch.float32)
```

#### `uniform(size, low=0.0, high=1.0, dtype=torch.float)`

Creates a tensor with random values from uniform distribution.

**Parameters:**
- `size` (tuple): Tensor shape
- `low` (float): Lower bound. Default: `0.0`
- `high` (float): Upper bound. Default: `1.0`
- `dtype` (torch.dtype): Data type. Default: `torch.float`

**Returns:**
- `torch.Tensor`: Random tensor with uniform distribution

**Example:**
```python
# Create random tensor between -1 and 1
uniform_tensor = iris_ctx.uniform((500, 500), low=-1.0, high=1.0)
```

#### `randint(size, low, high, dtype=torch.int)`

Creates a tensor with random integer values.

**Parameters:**
- `size` (tuple): Tensor shape
- `low` (int): Lower bound (inclusive)
- `high` (int): Upper bound (exclusive)
- `dtype` (torch.dtype): Data type. Default: `torch.int`

**Returns:**
- `torch.Tensor`: Random integer tensor

**Example:**
```python
# Create random integers between 0 and 100
int_tensor = iris_ctx.randint((100, 100), 0, 100)
```

### Specialized Tensor Creation

#### `linspace(start, end, steps, dtype=torch.float)`

Creates a 1D tensor with evenly spaced values.

**Parameters:**
- `start` (float): Start value
- `end` (float): End value
- `steps` (int): Number of steps
- `dtype` (torch.dtype): Data type. Default: `torch.float`

**Returns:**
- `torch.Tensor`: 1D tensor with evenly spaced values

**Example:**
```python
# Create tensor with 100 values from 0 to 2π
angles = iris_ctx.linspace(0, 2*3.14159, 100)
```

## Communication Operations

### Device-side Operations

#### `iris.store(ptr, value, source_rank, target_rank, heap_bases_ptr, mask=None)`

Stores a value from source rank to target rank's memory.

**Parameters:**
- `ptr` (tl.pointer): Pointer to target memory location
- `value` (tl.tensor): Value to store
- `source_rank` (tl.constexpr): Source rank ID
- `target_rank` (tl.constexpr): Target rank ID
- `heap_bases_ptr` (tl.pointer): Pointer to heap bases array
- `mask` (tl.tensor, optional): Boolean mask for conditional storage

**Example:**
```python
@triton.jit
def store_kernel(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Store value 42 from rank 0 to rank 1
    iris.store(buffer + offsets, 42.0, 0, 1, heap_bases_ptr, mask=mask)
```

#### `iris.load(ptr, source_rank, target_rank, heap_bases_ptr, mask=None)`

Loads a value from source rank's memory to target rank.

**Parameters:**
- `ptr` (tl.pointer): Pointer to source memory location
- `source_rank` (tl.constexpr): Source rank ID
- `target_rank` (tl.constexpr): Target rank ID
- `heap_bases_ptr` (tl.pointer): Pointer to heap bases array
- `mask` (tl.tensor, optional): Boolean mask for conditional loading

**Returns:**
- `tl.tensor`: Loaded value

**Example:**
```python
@triton.jit
def load_kernel(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Load value from rank 0 to rank 1
    value = iris.load(buffer + offsets, 0, 1, heap_bases_ptr, mask=mask)
```

### Atomic Operations

#### `iris.atomic_add(ptr, value, source_rank, target_rank, heap_bases_ptr, mask=None)`

Atomically adds a value to target rank's memory.

**Parameters:**
- `ptr` (tl.pointer): Pointer to target memory location
- `value` (tl.tensor): Value to add
- `source_rank` (tl.constexpr): Source rank ID
- `target_rank` (tl.constexpr): Target rank ID
- `heap_bases_ptr` (tl.pointer): Pointer to heap bases array
- `mask` (tl.tensor, optional): Boolean mask for conditional operation

**Returns:**
- `tl.tensor`: Previous value at the location

**Example:**
```python
@triton.jit
def atomic_add_kernel(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomically add 1 to rank 1's buffer
    iris.atomic_add(buffer + offsets, 1, 0, 1, heap_bases_ptr, mask=mask)
```

#### `iris.atomic_xchg(ptr, value, source_rank, target_rank, heap_bases_ptr, mask=None)`

Atomically exchanges a value with target rank's memory.

**Parameters:**
- `ptr` (tl.pointer): Pointer to target memory location
- `value` (tl.tensor): New value to store
- `source_rank` (tl.constexpr): Source rank ID
- `target_rank` (tl.constexpr): Target rank ID
- `heap_bases_ptr` (tl.pointer): Pointer to heap bases array
- `mask` (tl.tensor, optional): Boolean mask for conditional operation

**Returns:**
- `tl.tensor`: Previous value at the location

**Example:**
```python
@triton.jit
def atomic_xchg_kernel(buffer, heap_bases_ptr):
    offsets = tl.arange(0, 1024)
    mask = offsets < 1024
    
    # Atomically exchange value with rank 1's buffer
    old_value = iris.atomic_xchg(buffer + offsets, 100, 0, 1, heap_bases_ptr, mask=mask)
```

## Utility Methods

### Information and Control

#### `get_rank()`

Returns the current rank ID.

**Returns:**
- `int`: Current rank ID

**Example:**
```python
rank = iris_ctx.get_rank()
print(f"Current rank: {rank}")
```

#### `get_heap_bases()`

Returns the heap bases array for device-side operations.

**Returns:**
- `torch.Tensor`: Tensor containing heap base addresses for all ranks

**Example:**
```python
heap_bases = iris_ctx.get_heap_bases()
# Pass to device-side kernels
kernel[grid](buffer, heap_bases)
```

#### `barrier()`

Synchronizes all ranks.

**Example:**
```python
# Ensure all ranks complete operations
iris_ctx.barrier()
```

#### `broadcast(value, source_rank)`

Broadcasts a value from source rank to all other ranks.

**Parameters:**
- `value`: Value to broadcast
- `source_rank` (int): Source rank ID

**Returns:**
- Broadcasted value

**Example:**
```python
# Broadcast value from rank 0
if iris_ctx.get_rank() == 0:
    data = torch.tensor([1, 2, 3, 4])
else:
    data = None

# All ranks receive the broadcasted data
data = iris_ctx.broadcast(data, source_rank=0)
```

### Logging

#### `debug(message)`

Logs a debug message with rank information.

**Parameters:**
- `message` (str): Debug message

**Example:**
```python
iris_ctx.debug("Starting computation phase")
```

#### `info(message)`

Logs an info message with rank information.

**Parameters:**
- `message` (str): Info message

**Example:**
```python
iris_ctx.info("Computation completed successfully")
```

#### `warning(message)`

Logs a warning message with rank information.

**Parameters:**
- `message` (str): Warning message

**Example:**
```python
iris_ctx.warning("Memory usage is high")
```

#### `error(message)`

Logs an error message with rank information.

**Parameters:**
- `message` (str): Error message

**Example:**
```python
iris_ctx.error("Failed to allocate memory")
```

## Error Handling

### Common Exceptions

#### `MemoryError`

Raised when the symmetric heap runs out of memory.

**Example:**
```python
try:
    large_tensor = iris_ctx.zeros(2**30, dtype=torch.float32)
except MemoryError:
    print("Heap out of memory. Reduce heap size or tensor size.")
```

#### `RuntimeError`

Raised for device mismatches or unsupported operations.

**Example:**
```python
try:
    # This will raise RuntimeError if device doesn't match
    tensor = iris_ctx.zeros(100, device="cpu")
except RuntimeError as e:
    print(f"Device error: {e}")
```

## Best Practices

### Memory Management

1. **Plan heap size**: Estimate memory requirements before initialization
2. **Use appropriate dtypes**: Choose the smallest dtype that meets precision requirements
3. **Monitor heap usage**: Use logging to track memory allocation patterns

### Communication Patterns

1. **Use barriers appropriately**: Ensure synchronization at critical points
2. **Batch operations**: Group related operations to minimize communication overhead
3. **Check return values**: Always verify operation success

### Performance Optimization

1. **Use masks efficiently**: Apply masks only when necessary
2. **Minimize atomic operations**: Use regular operations when possible
3. **Profile communication**: Monitor timing of remote operations

## Examples

See the [Examples](examples.md) section for complete working examples of these APIs.

---

**Need help with specific APIs? Check the [Tutorials](../tutorials/basic-operations.md) or start a discussion in GitHub Discussions!**
