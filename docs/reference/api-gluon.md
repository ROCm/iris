# Gluon APIs (Experimental)

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

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

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.initialize
```

### Memory Operations

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.load
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.store
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.get
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.put
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.copy
```

### Atomic Operations

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_add
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_sub
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_cas
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_xchg
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_xor
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_and
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_or
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_min
.. automethod:: iris.experimental.iris_gluon.IrisDeviceCtx.atomic_max
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
