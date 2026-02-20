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
.. autofunction:: iris.experimental.iris_gluon.iris
```

## Core Methods

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_device_context
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_backend
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_heap_bases
.. automethod:: iris.experimental.iris_gluon.IrisGluon.barrier
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_device
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_cu_count
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_rank
.. automethod:: iris.experimental.iris_gluon.IrisGluon.get_num_ranks
```

## Logging Helpers

Use Iris-aware logging that automatically annotates each message with the current rank and world size. This is helpful when debugging multi-rank programs.

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.debug
.. automethod:: iris.experimental.iris_gluon.IrisGluon.info
.. automethod:: iris.experimental.iris_gluon.IrisGluon.warning
.. automethod:: iris.experimental.iris_gluon.IrisGluon.error
```

## Broadcast Helper

Broadcast data from a source rank to all ranks. This method automatically detects whether the value is a tensor/array or a scalar and uses the appropriate broadcast mechanism.

```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.broadcast
```



