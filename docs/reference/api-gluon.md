# Gluon APIs (Experimental)

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

## Requirements

The Gluon backend requires:
- **ROCm 7.0** or later
- **Triton commit `aafec417bded34db6308f5b3d6023daefae43905`** or later

These specific versions are necessary to access the experimental Gluon features and `@aggregate` decorator support.

## Overview

The Gluon API provides a Triton Gluon-based implementation of Iris that uses the `@aggregate` decorator with `@gluon.jit` to encapsulate the Iris backend state, eliminating the need to pass `heap_bases` around manually in kernels.

## Key Differences from Standard Iris

- Uses Triton's experimental `@gluon.jit` decorator for device-side methods
- Encapsulates heap_bases and rank info in an `IrisDeviceCtx` aggregate
- Provides the same functionality as standard Iris with improved ergonomics
- Better integration with Triton's Gluon programming model

## Usage Example

```python
import iris.experimental.iris_gluon as iris_gl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Host-side: Initialize Iris Gluon context
ctx = iris_gl.iris(heap_size=2**30)  # 1GB heap
context_tensor = ctx.get_device_context()

# Device-side: Use in Gluon kernels
@gluon.jit
def kernel(IrisDeviceCtx: gl.constexpr, context_tensor, buffer):
    # Initialize device context from tensor
    ctx = IrisDeviceCtx.initialize(context_tensor)
    
    # Perform remote memory operations
    data = ctx.load(buffer, from_rank=1)
    ctx.store(buffer, data, to_rank=0)
```

## Factory Function

```{eval-rst}
.. autofunction:: iris.experimental.iris_gluon.iris
```

## Host-Side IrisGluon Class

The `IrisGluon` class provides host-side methods for managing the multi-GPU context and symmetric heap.

### Initialization & Context

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_device_context
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_backend
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_heap_bases
```

### Rank Information

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_rank
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_num_ranks
```

### Device & Compute Units

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_device
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_cu_count
```

### Synchronization

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.barrier
.. automethod:: iris.experimental.iris_gluon.IrisGluon.broadcast
```

### Tensor Creation

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.zeros
.. automethod:: iris.experimental.iris_gluon.IrisGluon.ones
.. automethod:: iris.experimental.iris_gluon.IrisGluon.full
.. automethod:: iris.experimental.iris_gluon.IrisGluon.zeros_like
```

### Logging

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.debug
.. automethod:: iris.experimental.iris_gluon.IrisGluon.info
.. automethod:: iris.experimental.iris_gluon.IrisGluon.warning
.. automethod:: iris.experimental.iris_gluon.IrisGluon.error
```

## Device-Side IrisDeviceCtx Aggregate

The `IrisDeviceCtx` aggregate is used within Gluon kernels to perform remote memory operations. It encapsulates the symmetric heap state and provides device-side APIs.

### Initialization

#### `initialize(context_tensor)`

Initialize IrisDeviceCtx from the encoded tensor.

The context tensor has the format: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

**Arguments:**
- `context_tensor`: Pointer to encoded context data

**Returns:**
- `IrisDeviceCtx`: Initialized device context

**Example:**
```python
@gluon.jit
def kernel(IrisDeviceCtx: gl.constexpr, context_tensor, buffer):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    # Use ctx for remote memory operations
```

### Memory Operations

#### `load(self, pointer, from_rank, mask=None)`

Loads a value from the specified rank's memory location to the current rank.

**Arguments:**
- `pointer`: Pointer in the from_rank's address space
- `from_rank`: The rank ID from which to read the data
- `mask`: Optional mask for conditional loading

**Returns:**
- The loaded value from the target memory location

**Example:**
```python
# Load from rank 1 to current rank
data = ctx.load(buffer + offsets, 1, mask=mask)
```

#### `store(self, pointer, value, to_rank, mask=None)`

Writes data from the current rank to the specified rank's memory location.

**Arguments:**
- `pointer`: Pointer in the current rank's address space
- `value`: The value to store
- `to_rank`: The rank ID to which the data will be written
- `mask`: Optional mask for conditional storing

**Example:**
```python
# Store from current rank to rank 1
ctx.store(buffer + offsets, values, 1, mask=mask)
```

#### `get(self, from_ptr, to_ptr, from_rank, mask=None)`

Copies data from the specified rank's memory to the current rank's local memory.

**Arguments:**
- `from_ptr`: Pointer to remote memory in from_rank's address space
- `to_ptr`: Pointer to local memory in current rank
- `from_rank`: The rank ID from which to read the data
- `mask`: Optional mask for conditional operations

**Example:**
```python
# Copy from rank 1 to current rank's local memory
ctx.get(remote_ptr + offsets, local_ptr + offsets, 1, mask=mask)
```

#### `put(self, from_ptr, to_ptr, to_rank, mask=None)`

Copies data from the current rank's local memory to the specified rank's memory.

**Arguments:**
- `from_ptr`: Pointer to local memory in current rank
- `to_ptr`: Pointer to remote memory in to_rank's address space
- `to_rank`: The rank ID to which the data will be written
- `mask`: Optional mask for conditional operations

**Example:**
```python
# Copy from current rank's local memory to rank 1
ctx.put(local_ptr + offsets, remote_ptr + offsets, 1, mask=mask)
```

#### `copy(self, src_ptr, dst_ptr, from_rank, to_rank, mask=None)`

Copies data from the specified rank's memory into the destination rank's memory.

This function performs the transfer by translating src_ptr from the from_rank's address space to the to_rank's address space, performing a masked load from the translated source, and storing the loaded data to dst_ptr in the to_rank memory location. If from_rank and to_rank are the same, this function performs a local copy operation. It is undefined behaviour if neither from_rank nor to_rank is the cur_rank.

**Arguments:**
- `src_ptr`: Pointer in the from_rank's local memory from which to read data
- `dst_ptr`: Pointer in the to_rank's local memory where the data will be written
- `from_rank`: The rank ID that owns src_ptr (source rank)
- `to_rank`: The rank ID that will receive the data (destination rank)
- `mask`: Optional mask for conditional operations

**Example:**
```python
# Copy from rank 1 to rank 0 (current rank must be either 1 or 0)
ctx.copy(remote_ptr + offsets, local_ptr + offsets, 1, 0, mask=mask)
```

### Atomic Operations

#### `atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic add at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to add
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically add to rank 1's memory
old_val = ctx.atomic_add(buffer, 5, 1)
```

#### `atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Atomically subtracts data from the specified rank's memory location.

**Arguments:**
- `pointer`: Pointer in the current rank's address space
- `val`: The value to subtract
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically subtract from rank 1's memory
old_val = ctx.atomic_sub(buffer, 3, 1)
```

#### `atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None)`

Atomically compares and exchanges the specified rank's memory location.

**Arguments:**
- `pointer`: Pointer in the current rank's address space
- `cmp`: The expected value to compare
- `val`: The new value to write if comparison succeeds
- `to_rank`: The rank ID to which the atomic operation will be performed
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Compare-and-swap on rank 1's memory
old_val = ctx.atomic_cas(flag + pid, 0, 1, 1, sem="release", scope="sys")
```

#### `atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic exchange at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to exchange
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Exchange value with rank 1's memory
old_val = ctx.atomic_xchg(buffer, 99, 1)
```

#### `atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic xor at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to xor
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically XOR with rank 1's memory
old_val = ctx.atomic_xor(buffer, 0xFF, 1)
```

#### `atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic and at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to and
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically AND with rank 1's memory
old_val = ctx.atomic_and(buffer, 0x0F, 1)
```

#### `atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic or at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to or
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically OR with rank 1's memory
old_val = ctx.atomic_or(buffer, 0xF0, 1)
```

#### `atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic min at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to compare and potentially store
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically compute minimum with rank 1's memory
old_val = ctx.atomic_min(buffer, 10, 1)
```

#### `atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None)`

Performs an atomic max at the specified rank's memory location.

**Arguments:**
- `pointer`: The memory location in the current rank's address space
- `val`: The value to compare and potentially store
- `to_rank`: The rank ID to which the atomic operation will be performed
- `mask`: Optional mask for conditional operations
- `sem`: Memory semantics (acquire, release, acq_rel, relaxed)
- `scope`: Scope of synchronization (gpu, cta, sys)

**Returns:**
- The value at the memory location before the atomic operation

**Example:**
```python
# Atomically compute maximum with rank 1's memory
old_val = ctx.atomic_max(buffer, 100, 1)
```

## Complete Example: Producer-Consumer Pattern

Here's a complete example demonstrating the use of Gluon APIs for a producer-consumer pattern:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl

@gluon.jit
def producer_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    source_buffer,
    target_buffer,
    flag,
    buffer_size,
    producer_rank: gl.constexpr,
    consumer_rank: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < buffer_size
    
    # Load from producer's buffer
    values = ctx.load(source_buffer + offsets, producer_rank, mask=mask)
    
    # Store to consumer's buffer
    ctx.store(target_buffer + offsets, values, consumer_rank, mask=mask)
    
    # Signal completion
    ctx.atomic_cas(flag + pid, 0, 1, consumer_rank, sem="release", scope="sys")

@gluon.jit
def consumer_kernel(
    IrisDeviceCtx: gl.constexpr,
    context_tensor,
    buffer,
    flag,
    buffer_size,
    consumer_rank: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    pid = gl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < buffer_size
    
    # Wait for producer
    done = 0
    while done == 0:
        done = ctx.atomic_cas(flag + pid, 1, 0, consumer_rank, sem="acquire", scope="sys")
    
    # Read from buffer
    values = ctx.load(buffer + offsets, consumer_rank, mask=mask)
    
    # Process values...
    values = values * 2
    
    # Store back
    ctx.store(buffer + offsets, values, consumer_rank, mask=mask)

def worker(rank, world_size):
    # Initialize distributed
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )
    
    # Initialize Iris Gluon
    ctx = iris_gl.iris(heap_size=2**30)
    context_tensor = ctx.get_device_context()
    
    # Allocate buffers
    buffer_size = 1024
    source = ctx.zeros(buffer_size, dtype=torch.float32)
    target = ctx.zeros(buffer_size, dtype=torch.float32)
    flag = ctx.zeros(triton.cdiv(buffer_size, 256), dtype=torch.int32)
    
    # Launch kernels based on rank...
    # (see examples/06_message_passing/message_passing_gluon.py for full code)
    
    ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
```

For more complete examples, see:
- `examples/06_message_passing/message_passing_gluon.py`
- Unit tests in `tests/unittests/test_*_gluon.py`
