# Iris API Comparison: Original vs Gluon

This document provides a side-by-side comparison of the Original Iris API and the Gluon-based API.

## Simple Load/Store Example

### Original API

```python
import torch
import triton
import triton.language as tl
import iris

# Host code
ctx = iris.iris(heap_size=2**30)
buffer = ctx.zeros(1024, dtype=torch.float32)
heap_bases = ctx.get_heap_bases()

@triton.jit
def kernel(buffer, heap_bases):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Load from rank 1
    data = iris.load(buffer + offsets, 0, 1, heap_bases)
    
    # Store to rank 1
    iris.store(buffer + offsets, data * 2, 0, 1, heap_bases)

# Launch
kernel[grid](buffer, heap_bases)
```

### Gluon API

```python
import torch
import triton
import triton.language as tl
import iris.experimental.iris_gluon as iris_gl

# Host code
ctx = iris_gl.iris(heap_size=2**30)
buffer = ctx.zeros(1024, dtype=torch.float32)
backend = ctx.get_backend()  # Get aggregate instead of heap_bases

@triton.jit
def kernel(buffer, backend: iris_gl.IrisBackend):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Load from rank 1
    data = backend.load(buffer + offsets, 1)
    
    # Store to rank 1
    backend.store(buffer + offsets, data * 2, 1)

# Launch
kernel[grid](buffer, backend)
```

**Key Differences:**
- ✅ No need to pass `heap_bases` separately
- ✅ Backend methods are called on the object: `backend.load()` vs `iris.load()`
- ✅ One fewer parameter to track

---

## Producer-Consumer Pattern

### Original API

```python
import iris

@triton.jit
def producer_kernel(source, target, flag, producer_rank: tl.constexpr, 
                   consumer_rank: tl.constexpr, heap_bases):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Load from local memory
    values = iris.load(source + offsets, producer_rank, producer_rank, heap_bases)
    
    # Store to remote memory
    iris.store(target + offsets, values, producer_rank, consumer_rank, heap_bases)
    
    # Signal completion
    iris.atomic_cas(flag + pid, 0, 1, producer_rank, consumer_rank, 
                   heap_bases, sem="release", scope="sys")

@triton.jit
def consumer_kernel(buffer, flag, consumer_rank: tl.constexpr, heap_bases):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Wait for data
    done = 0
    while done == 0:
        done = iris.atomic_cas(flag + pid, 1, 0, consumer_rank, consumer_rank,
                              heap_bases, sem="acquire", scope="sys")
    
    # Read data
    values = iris.load(buffer + offsets, consumer_rank, consumer_rank, heap_bases)
    
    # Process
    values = values * 2
    iris.store(buffer + offsets, values, consumer_rank, consumer_rank, heap_bases)

# Launch on rank 0
producer_kernel[grid](source, target, flag, 0, 1, heap_bases)

# Launch on rank 1
consumer_kernel[grid](buffer, flag, 1, heap_bases)
```

### Gluon API

```python
import iris.experimental.iris_gluon as iris_gl

@triton.jit
def producer_kernel(source, target, flag, producer_rank: tl.constexpr, 
                   consumer_rank: tl.constexpr, backend: iris_gl.IrisBackend):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Load from local memory
    values = backend.load(source + offsets, producer_rank)
    
    # Store to remote memory
    backend.store(target + offsets, values, consumer_rank)
    
    # Signal completion
    backend.atomic_cas(flag + pid, 0, 1, consumer_rank,
                      sem="release", scope="sys")

@triton.jit
def consumer_kernel(buffer, flag, consumer_rank: tl.constexpr, 
                   backend: iris_gl.IrisBackend):
    pid = tl.program_id(0)
    offsets = pid * 64 + tl.arange(0, 64)
    
    # Wait for data
    done = 0
    while done == 0:
        done = backend.atomic_cas(flag + pid, 1, 0, consumer_rank,
                                 sem="acquire", scope="sys")
    
    # Read data
    values = backend.load(buffer + offsets, consumer_rank)
    
    # Process
    values = values * 2
    backend.store(buffer + offsets, values, consumer_rank)

# Launch on rank 0
producer_kernel[grid](source, target, flag, 0, 1, backend)

# Launch on rank 1
consumer_kernel[grid](buffer, flag, 1, backend)
```

**Key Differences:**
- ✅ Cleaner kernel signatures (one parameter instead of many)
- ✅ All operations go through backend object
- ✅ Less visual clutter in the code

---

## Atomic Operations

### Original API

```python
@triton.jit
def atomic_kernel(counter, heap_bases):
    # Atomic add
    old = iris.atomic_add(counter, 1, 0, 1, heap_bases)
    
    # Atomic CAS
    old = iris.atomic_cas(counter, 0, 42, 0, 1, heap_bases)
    
    # Atomic exchange
    old = iris.atomic_xchg(counter, 99, 0, 1, heap_bases)
    
    # Atomic min/max
    old = iris.atomic_min(counter, 10, 0, 1, heap_bases)
    old = iris.atomic_max(counter, 100, 0, 1, heap_bases)
```

### Gluon API

```python
@triton.jit
def atomic_kernel(counter, backend: iris_gl.IrisBackend):
    # Atomic add
    old = backend.atomic_add(counter, 1, 1)
    
    # Atomic CAS
    old = backend.atomic_cas(counter, 0, 42, 1)
    
    # Atomic exchange
    old = backend.atomic_xchg(counter, 99, 1)
    
    # Atomic min/max
    old = backend.atomic_min(counter, 10, 1)
    old = backend.atomic_max(counter, 100, 1)
```

**Key Differences:**
- ✅ Shorter function calls (no heap_bases parameter)
- ✅ More readable with consistent method call syntax

---

## Get/Put Operations

### Original API

```python
@triton.jit
def transfer_kernel(remote_ptr, local_ptr, heap_bases):
    offsets = tl.arange(0, 64)
    
    # Get: copy from remote to local
    iris.get(remote_ptr + offsets, local_ptr + offsets, 1, 0, heap_bases)
    
    # Put: copy from local to remote
    iris.put(local_ptr + offsets, remote_ptr + offsets, 0, 1, heap_bases)
```

### Gluon API

```python
@triton.jit
def transfer_kernel(remote_ptr, local_ptr, backend: iris_gl.IrisBackend):
    offsets = tl.arange(0, 64)
    
    # Get: copy from remote to local
    backend.get(remote_ptr + offsets, local_ptr + offsets, 1)
    
    # Put: copy from local to remote
    backend.put(local_ptr + offsets, remote_ptr + offsets, 1)
```

**Key Differences:**
- ✅ Consistent object-oriented style
- ✅ Less parameter passing

---

## Copy Between Ranks

The `copy` function enables direct copying between any two ranks (where current rank must be either source or destination).

### Original API

```python
@triton.jit
def copy_kernel(src_ptr, dst_ptr, cur_rank, heap_bases):
    offsets = tl.arange(0, 64)
    
    # Copy from rank 1 to rank 2 (when cur_rank is either 1 or 2)
    iris.copy(src_ptr + offsets, dst_ptr + offsets, 1, 2, cur_rank, heap_bases)
```

### Gluon API

```python
@triton.jit
def copy_kernel(src_ptr, dst_ptr, backend: iris_gl.IrisBackend):
    offsets = tl.arange(0, 64)
    
    # Copy from rank 1 to rank 2 (cur_rank automatically from backend)
    backend.copy(src_ptr + offsets, dst_ptr + offsets, 1, 2)
```

**Key Differences:**
- ✅ No need to pass `cur_rank` explicitly - it's in the backend
- ✅ More flexible than get/put for rank-to-rank copies

---

## Memory Semantics and Scope

Both APIs support the same memory semantics and scope parameters:

### Original API

```python
iris.atomic_add(ptr, 1, 0, 1, heap_bases, sem="acquire", scope="sys")
iris.store(ptr, value, 0, 1, heap_bases, mask=mask)
```

### Gluon API

```python
backend.atomic_add(ptr, 1, 1, sem="acquire", scope="sys")
backend.store(ptr, value, 1, mask=mask)
```

**Supported Values:**
- `sem`: "acquire", "release", "acq_rel", "relaxed"
- `scope`: "gpu", "cta", "sys"
- `mask`: Optional boolean mask for conditional operations

---

## Complete Host-Side Comparison

### Original API

```python
import iris

# Initialize
ctx = iris.iris(heap_size=2**30)

# Get info
rank = ctx.get_rank()
num_ranks = ctx.get_num_ranks()
device = ctx.get_device()

# Allocate memory
tensor = ctx.zeros(1024, dtype=torch.float32)

# Synchronization
ctx.barrier()

# Logging
ctx.info("Starting computation")

# Get heap bases for kernel
heap_bases = ctx.get_heap_bases()
```

### Gluon API

```python
import iris.experimental.iris_gluon as iris_gl

# Initialize
ctx = iris_gl.iris(heap_size=2**30)

# Get info (same)
rank = ctx.get_rank()
num_ranks = ctx.get_num_ranks()
device = ctx.get_device()

# Allocate memory (same)
tensor = ctx.zeros(1024, dtype=torch.float32)

# Synchronization (same)
ctx.barrier()

# Logging (same)
ctx.info("Starting computation")

# Get backend aggregate for kernel
backend = ctx.get_backend()
```

**Key Differences:**
- Host-side API is nearly identical
- Only difference: `get_backend()` instead of `get_heap_bases()`

---

## Summary

| Aspect | Original API | Gluon API |
|--------|-------------|-----------|
| **Parameter passing** | Must pass `heap_bases` to every function | Pass `backend` aggregate once |
| **Function calls** | Module-level functions: `iris.load()` | Object methods: `backend.load()` |
| **Code clarity** | More verbose | More concise |
| **Type safety** | `heap_bases` type unclear | `backend: IrisBackend` is explicit |
| **Encapsulation** | State passed separately | State bundled in aggregate |
| **Backward compatibility** | N/A - original API | ✅ Fully compatible |
| **Performance** | Baseline | Expected to be equivalent |

## Migration Guide

To migrate from Original API to Gluon API:

1. **Change import:**
   ```python
   # Before
   import iris
   
   # After
   import iris.experimental.iris_gluon as iris_gl
   ```

2. **Update initialization:**
   ```python
   # Before
   heap_bases = ctx.get_heap_bases()
   
   # After
   backend = ctx.get_backend()
   ```

3. **Update kernel signatures:**
   ```python
   # Before
   @triton.jit
   def kernel(..., heap_bases):
   
   # After
   @triton.jit
   def kernel(..., backend: iris_gl.IrisBackend):
   ```

4. **Update function calls:**
   ```python
   # Before
   iris.load(ptr, 0, 1, heap_bases)
   
   # After
   backend.load(ptr, 1)  # Only need remote rank
   ```

5. **Update kernel launches:**
   ```python
   # Before
   kernel[grid](..., heap_bases)
   
   # After
   kernel[grid](..., backend)
   ```

That's it! The rest of the code remains the same.
