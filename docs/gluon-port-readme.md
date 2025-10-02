# Iris Gluon Port

This directory contains the Gluon-based implementation of Iris, which uses Triton's `@aggregate` decorator to encapsulate the Iris backend state.

## Overview

The Gluon port provides the same functionality as the original Iris but with a cleaner API that eliminates the need to pass `heap_bases` as a separate parameter to device-side functions.

## Key Components

### 1. IrisBackend Aggregate (`iris/iris_gluon.py`)

The `IrisBackend` is a Triton aggregate (similar to a struct) that encapsulates:
- `heap_bases`: Pointer to array of heap base addresses for all ranks
- `cur_rank`: Current rank ID
- `num_ranks`: Total number of ranks

It provides device-side methods for:
- Memory operations: `load()`, `store()`, `get()`, `put()`
- Atomic operations: `atomic_add()`, `atomic_sub()`, `atomic_cas()`, `atomic_xchg()`, `atomic_xor()`, `atomic_and()`, `atomic_or()`, `atomic_min()`, `atomic_max()`

### 2. IrisGluon Class

The host-side class that manages:
- Symmetric heap allocation
- Memory management
- Distributed coordination
- Logging with rank information

## Usage Example

### Host Code

```python
import iris.iris_gluon as iris_gl

# Initialize Iris with 1GB heap
ctx = iris_gl.iris(heap_size=2**30)

# Get the backend aggregate
backend = ctx.get_backend()

# Allocate tensors on symmetric heap
buffer = ctx.zeros(1024, device="cuda", dtype=torch.float32)
```

### Device Code

```python
import triton
import triton.language as tl
import iris.iris_gluon as iris_gl

@triton.jit
def my_kernel(buffer, backend: iris_gl.IrisBackend):
    cur_rank = 0
    remote_rank = 1
    
    # Load from remote rank using backend
    data = backend.load(buffer, remote_rank)
    
    # Store to remote rank using backend
    backend.store(buffer, data * 2, remote_rank)
    
    # Atomic operations using backend
    old_val = backend.atomic_add(buffer, 1, remote_rank)
```

## Comparison with Original Iris

### Original Iris (Triton-based)

```python
@triton.jit
def kernel(buffer, heap_bases):
    cur_rank = 0
    remote_rank = 1
    
    # Need to pass heap_bases to every function
    data = iris.load(buffer, cur_rank, remote_rank, heap_bases)
    iris.store(buffer, data * 2, cur_rank, remote_rank, heap_bases)
    iris.atomic_add(buffer, 1, cur_rank, remote_rank, heap_bases)
```

### Gluon-based Iris

```python
@triton.jit
def kernel(buffer, backend: iris_gl.IrisBackend):
    cur_rank = 0
    remote_rank = 1
    
    # Backend encapsulates heap_bases and cur_rank
    data = backend.load(buffer, remote_rank)
    backend.store(buffer, data * 2, remote_rank)
    backend.atomic_add(buffer, 1, remote_rank)
```

## Benefits

1. **Cleaner API**: No need to pass `heap_bases` to every device function
2. **Better Encapsulation**: Backend state is bundled together in an aggregate
3. **Type Safety**: The backend aggregate provides a clear contract for device code
4. **Consistency**: All Iris operations go through the backend object

## Examples

See `examples/06_message_passing/message_passing_gluon.py` for a complete producer-consumer example using the Gluon port.

## Implementation Notes

- The `@aggregate` decorator is from Triton's language core, not Gluon specifically
- Device-side methods in `IrisBackend` use Triton language (`tl.*`) primitives
- The implementation maintains full compatibility with the original Iris API
- All atomic operations support the same semantics (`sem`) and scope (`scope`) parameters

## Future Work

- Port additional examples to use the Gluon-based API
- Add performance benchmarks comparing Gluon vs original implementation
- Explore additional Gluon-specific optimizations
