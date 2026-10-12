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

```{note}
Every operation here also exists as a free function -- see
[Device Functions](device-functions.md). The free form is explicit about both
ends of the transfer and the heap: `load(pointer, to_rank, from_rank, heap_bases)`.
The bound form below knows the current rank and carries the heap bases itself, so
the same call is `ctx.load(pointer, from_rank)`.
```

## Initialization

### initialize
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.initialize
   :noindex:
```

## Memory transfer operations

### load
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.load
   :noindex:
```

### store
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.store
   :noindex:
```

### get
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.get
   :noindex:
```

### put
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.put
   :noindex:
```

### copy
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.copy
   :noindex:
```

## Atomic operations

### atomic_add
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_add
   :noindex:
```

### atomic_sub
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_sub
   :noindex:
```

### atomic_cas
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_cas
   :noindex:
```

### atomic_xchg
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_xchg
   :noindex:
```

### atomic_xor
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_xor
   :noindex:
```

### atomic_and
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_and
   :noindex:
```

### atomic_or
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_or
   :noindex:
```

### atomic_min
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_min
   :noindex:
```

### atomic_max
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.atomic_max
   :noindex:
```

## Collective operations

### all_reduce_atomic
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_atomic
   :noindex:
```

### all_reduce_spinlock
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_spinlock
   :noindex:
```

### all_reduce_one_shot
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_one_shot
   :noindex:
```

### all_reduce_ring
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_ring
   :noindex:
```

### all_reduce_two_shot
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_reduce_two_shot
   :noindex:
```

### all_gather
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_gather
   :noindex:
```

### gather
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.gather
   :noindex:
```

### all_to_all
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.all_to_all
   :noindex:
```

### reduce_scatter
```{eval-rst}
.. automethod:: iris.mem.triton.context.Context.reduce_scatter
   :noindex:
```
