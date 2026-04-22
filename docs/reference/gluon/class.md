# Iris Class

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

## Requirements

The Gluon backend requires:
- **ROCm 7.0** or later
- **Triton commit `aafec417bded34db6308f5b3d6023daefae43905`** or later

## Factory Function

Prefer using the convenience factory over calling the constructor directly:

```{eval-rst}
.. autofunction:: iris.host.iris.iris
```

## Core Methods

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_device_context
.. automethod:: iris.host.iris.Iris.get_backend
.. automethod:: iris.host.iris.Iris.get_heap_bases
.. automethod:: iris.host.iris.Iris.barrier
.. automethod:: iris.host.iris.Iris.get_device
.. automethod:: iris.host.iris.Iris.get_cu_count
.. automethod:: iris.host.iris.Iris.get_rank
.. automethod:: iris.host.iris.Iris.get_num_ranks
```

## Logging Helpers

Use Iris-aware logging that automatically annotates each message with the current rank and world size. This is helpful when debugging multi-rank programs.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.debug
.. automethod:: iris.host.iris.Iris.info
.. automethod:: iris.host.iris.Iris.warning
.. automethod:: iris.host.iris.Iris.error
```

## Broadcast Helper

Broadcast data from a source rank to all ranks. This method automatically detects whether the value is a tensor/array or a scalar and uses the appropriate broadcast mechanism.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.broadcast
```



