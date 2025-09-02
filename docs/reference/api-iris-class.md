# Iris Class API

## Factory Function

Prefer using the convenience factory over calling the constructor directly:

```{eval-rst}
.. autofunction:: iris.iris.iris
```

## Core Methods

```{eval-rst}
.. automethod:: iris.iris.Iris.get_heap_bases
.. automethod:: iris.iris.Iris.barrier
.. automethod:: iris.iris.Iris.get_device
.. automethod:: iris.iris.Iris.get_cu_count
.. automethod:: iris.iris.Iris.get_rank
.. automethod:: iris.iris.Iris.get_num_ranks
```

## Logging Helpers

Use Iris-aware logging that automatically annotates each message with the current rank and world size. This is helpful when debugging multi-rank programs.

```{eval-rst}
.. automethod:: iris.iris.Iris.debug
.. automethod:: iris.iris.Iris.info
.. automethod:: iris.iris.Iris.warning
.. automethod:: iris.iris.Iris.error
```


## Broadcast Helper

Broadcast a Python scalar or small object from a source rank to all ranks. This is a convenience wrapper over the internal MPI helper.

```{eval-rst}
.. automethod:: iris.iris.Iris.broadcast
```




