# Iris Class

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
.. autofunction:: iris.logging.set_logger_level
.. automethod:: iris.iris.Iris.debug
.. automethod:: iris.iris.Iris.info
.. automethod:: iris.iris.Iris.warning
.. automethod:: iris.iris.Iris.error
```


## Utility Functions

```{eval-rst}
.. autofunction:: iris.util.do_bench
```

## Broadcast Helper

Broadcast data from a source rank to all ranks. This method automatically detects whether the value is a tensor/array or a scalar and uses the appropriate broadcast mechanism.

```{eval-rst}
.. automethod:: iris.iris.Iris.broadcast
```




