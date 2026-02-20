# Gluon (Experimental)

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
- Encapsulates `heap_bases` and rank info in an `IrisDeviceCtx` aggregate
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

## API Reference

Explore the API by section:

- [Iris Class](class.md)
- [Tensor Creation](tensor-creation.md)
- [Device Functions](device-functions.md)
- [Collective Communication (CCL)](ccl.md)

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
    block_size = 256
    source = ctx.zeros(buffer_size, dtype=torch.float32)
    target = ctx.zeros(buffer_size, dtype=torch.float32)
    num_blocks = triton.cdiv(buffer_size, block_size)
    flag = ctx.zeros(num_blocks, dtype=torch.int32)
    
    # Initialize source data on producer
    producer_rank = 0
    consumer_rank = 1
    if rank == producer_rank:
        source.fill_(42.0)
    
    # Launch kernels based on rank
    grid = (num_blocks,)
    if rank == producer_rank:
        ctx.info(f"Rank {rank} producing data...")
        producer_kernel[grid](
            iris_gl.IrisDeviceCtx,
            context_tensor,
            source,
            target,
            flag,
            buffer_size,
            producer_rank,
            consumer_rank,
            block_size,
            num_warps=1,
        )
    else:
        ctx.info(f"Rank {rank} consuming data...")
        consumer_kernel[grid](
            iris_gl.IrisDeviceCtx,
            context_tensor,
            target,
            flag,
            buffer_size,
            consumer_rank,
            block_size,
            num_warps=1,
        )
    
    ctx.barrier()
    
    # Validate on consumer
    if rank == consumer_rank:
        expected = source * 2  # Consumer doubles the values
        if torch.allclose(target, expected, atol=1):
            ctx.info("Validation successful!")
        else:
            ctx.error("Validation failed!")
    
    ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
```
