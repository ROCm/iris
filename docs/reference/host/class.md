# Iris Class

## Factory Function

Prefer using the convenience factory over calling the constructor directly:

```{eval-rst}
.. autofunction:: iris.host.iris.iris
```

## Device Placement

By default each rank runs on GPU `rank % torch.cuda.device_count()`, and Iris
makes it the current device. If your launcher puts ranks on other GPUs,
for example ranks 0-1 on GPUs 2-3, pass the GPU explicitly. It must be the
device the process group was initialized with, because Iris runs its
collectives on that device:

```python
torch.cuda.set_device(gpu)
dist.init_process_group("nccl", device_id=torch.device("cuda", gpu))
ctx = iris.iris(heap_size=1 << 30, device=gpu)  # or device="cuda" for the current device
```

`device` accepts a GPU index, `"cuda"`, `"cuda:N"`, or a `torch.device`. `"cuda"`
means the current device, so set it with `torch.cuda.set_device` first. If the
process group was initialized with `device_id`, Iris raises `ValueError` when
the GPU it would use, whether from `device` or the default, is a different one.

Ranks find each other's GPUs by PCI address, so processes may see the GPUs in a
different order (different `HIP_VISIBLE_DEVICES`). The `vmem` allocator and the
SDMA copy engine map peers' memory onto GPUs on this host, so each process must
be able to see every rank's GPU. Both are single-host: in a job that spans hosts,
`engines=("sdma",)` raises `ValueError` on every rank during initialization.

## Optional SDMA Copy Engine

The `engines` parameter defaults to `()`. In this mode, Iris skips SDMA
initialization and queue creation, and `get_copy_engine_ctx()` returns `None`.
Regular Triton and Gluon memory operations and collectives work without SDMA.

Pass `engines=("sdma",)` to initialize both host- and device-initiated SDMA queues.
This needs the `sdma` extra ([Installation](../../getting-started/installation.md)):

```python
ctx = iris.iris(heap_size=1 << 30, engines=("sdma",))
copy_engine_ctx = ctx.get_copy_engine_ctx()
```

This is required for the host methods `put`, `put_tile`, `put_tiles`, and `quiet`,
and for device operations using `use_copy_engine=True`. Host SDMA methods raise
`RuntimeError` if the copy engine is disabled. Pass the returned context to
device operations through their `copy_engine_ctx` argument.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_copy_engine_ctx
```

## Core Methods

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_heap_bases
.. automethod:: iris.host.iris.Iris.get_device_context
.. automethod:: iris.host.iris.Iris.barrier
.. automethod:: iris.host.iris.Iris.get_device
.. automethod:: iris.host.iris.Iris.get_cu_count
.. automethod:: iris.host.iris.Iris.get_rank
.. automethod:: iris.host.iris.Iris.get_num_ranks
```

## Logging Helpers

Use Iris-aware logging that automatically annotates each message with the current rank and world size. This is helpful when debugging multi-rank programs.

```{eval-rst}
.. autofunction:: iris.host.logging.logging.set_logger_level
.. automethod:: iris.host.iris.Iris.debug
.. automethod:: iris.host.iris.Iris.info
.. automethod:: iris.host.iris.Iris.warning
.. automethod:: iris.host.iris.Iris.error
```

## Utility Functions

```{eval-rst}
.. autofunction:: iris.host.platform.utils.do_bench
```

## Broadcast Helper

Broadcast data from a source rank to all ranks. This method automatically detects whether the value is a tensor/array or a scalar and uses the appropriate broadcast mechanism.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.broadcast
```

## allocate_symmetric

```{eval-rst}
.. automethod:: iris.host.iris.Iris.allocate_symmetric
```

## device_barrier

```{eval-rst}
.. automethod:: iris.host.iris.Iris.device_barrier
```

## get_device_id

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_device_id
```

## ops

```{eval-rst}
.. autoproperty:: iris.host.iris.Iris.ops
```
