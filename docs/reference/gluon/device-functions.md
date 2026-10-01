# Device Functions

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

Device-side functions provided by Iris Gluon for remote memory operations and atomics. These methods are part of the `IrisDeviceCtx` aggregate used within Gluon kernels.

## Initialization

### initialize
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.initialize
   :noindex:
```

## Memory transfer operations

### load
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.load
   :noindex:
```

### store
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.store
   :noindex:
```

### copy
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.copy
   :noindex:
```

### get
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.get
   :noindex:
```

### put
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.put
   :noindex:
```

## Atomic operations

### atomic_add
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_add
   :noindex:
```

### atomic_sub
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_sub
   :noindex:
```

### atomic_cas
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_cas
   :noindex:
```

### atomic_xchg
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_xchg
   :noindex:
```

### atomic_xor
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_xor
   :noindex:
```

### atomic_and
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_and
   :noindex:
```

### atomic_or
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_or
   :noindex:
```

### atomic_min
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_min
   :noindex:
```

### atomic_max
```{eval-rst}
.. automethod:: iris.mem.gluon.context.IrisDeviceCtx.atomic_max
   :noindex:
```

