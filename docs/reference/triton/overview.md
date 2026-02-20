# Triton

The standard Iris API uses Triton's `@triton.jit` decorator for device-side remote memory operations.

## Overview

Iris provides a Triton-based API for multi-GPU communication and symmetric heap management. Device-side operations are written using `@triton.jit` decorated functions that can perform remote memory access across GPUs.

## Usage Example

```python
import iris
import triton
import triton.language as tl

# Host-side: Initialize Iris context
ctx = iris.iris(heap_size=2**30)  # 1GB heap

# Device-side: Use in Triton kernels
@triton.jit
def kernel(ptr, heap_bases, cur_rank, remote_rank, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Perform remote memory operations
    data = iris.load(ptr + offsets, cur_rank, remote_rank, heap_bases)
    iris.store(ptr + offsets, data, cur_rank, remote_rank, heap_bases)
```

## API Reference

Explore the API by section:

- [Iris Class](class.md)
- [Tensor Creation](tensor-creation.md)
- [Device Functions](device-functions.md)
- [Collective Communication (CCL)](ccl.md)
- [Fused GEMM + CCL Operations](ops.md)

