# Device Context

Device-side methods provided by Iris for remote memory operations, atomics and
collectives. These methods are part of the `DeviceContext` aggregate used inside
`@triton.jit` kernels, and are reached through the context tensor passed in from
the host:

```python
from iris import DeviceContext

@triton.jit
def producer(context_tensor, target_buffer, ..., rank: tl.constexpr, world_size: tl.constexpr):
    ctx = DeviceContext.initialize(context_tensor, rank, world_size)
    ...
    ctx.store(target_buffer + offsets, values, to_rank=consumer_rank, mask=mask)
    ctx.atomic_cas(flag + pid, 0, 1, to_rank=consumer_rank, sem="release", scope="sys")
```

Taken from `examples/06_message_passing/message_passing_device_context.py`.

```{note}
The collectives below are **tile-level** -- they run inside your kernel, on the
tile a program is already holding, and fuse with whatever compute surrounds
them. They are a different surface from the host collectives in
[Collective Communication Operations](../host/ccl.md), which launch their own
kernels and are called from Python.
```

Every operation here is also available as a module-level function -- see
[Device Functions](device-functions.md). The module-level form takes the heap
bases and both ranks explicitly; the context already knows the current rank and
holds the heap bases, so its methods take fewer arguments:

```python
iris.load(pointer, to_rank, from_rank, heap_bases)   # module-level function
ctx.load(pointer, from_rank)                         # method on the context
```

## Initialization

### initialize
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.initialize
```

## Memory transfer operations

### load
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.load
```

### store
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.store
```

### get
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.get
```

### put
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.put
```

### copy
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.copy
```

## Atomic operations

### atomic_add
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_add
```

### atomic_sub
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_sub
```

### atomic_cas
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_cas
```

### atomic_xchg
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_xchg
```

### atomic_xor
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_xor
```

### atomic_and
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_and
```

### atomic_or
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_or
```

### atomic_min
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_min
```

### atomic_max
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_max
```

## Collective operations

### all_reduce_atomic
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_atomic
```

### all_reduce_spinlock
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_spinlock
```

### all_reduce_one_shot
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_one_shot
```

### all_reduce_ring
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_ring
```

### all_reduce_two_shot
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_two_shot
```

### all_gather
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_gather
```

### gather
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.gather
```

### all_to_all
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_to_all
```

### reduce_scatter
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.reduce_scatter
```
