# Fused Communication and Compute Operations

Operations that fuse communication with compute in a single kernel, accessible
via the `ops` property on the `Iris` instance (e.g. `ctx.ops.matmul_all_reduce(...)`).
The compute side is currently GEMM, supplied by tritonBLAS; the communication side
is an Iris collective. Fusing them means the collective overlaps the matmul instead
of running as a separate kernel after it.

## matmul_all_reduce
```{eval-rst}
.. automethod:: iris.ops.OpsNamespace.matmul_all_reduce
```

### Reusing an all-reduce workspace

This fused operation supports `atomic`, `spinlock`, `one_shot` and `two_shot`.
The shared `FusedConfig` also accepts `ring`, but this operation does not implement
it and raises `ValueError` collectively instead of launching a kernel with no
matching reduction branch.

Keep the returned workspace and pass it to subsequent collective calls:

```python
workspace = None
for A, B in inputs:
    workspace = ctx.ops.matmul_all_reduce(C, A, B, workspace=workspace, config=config)
```

All ranks must call the operation in the same order with matching dimensions,
dtype, variant and tile configuration. Allocation decisions are agreed across
the default process group: if one rank needs replacement buffers, every rank
participates. The workspace belongs to its original Iris context. A different
context, changed shape/dtype/variant/tile mapping, or exhausted generation counter
requires collective reallocation.

`one_shot` and `two_shot` overwrite their output and use versioned readiness and
completion flags. Reusing a matching workspace does not clear these flags or the
auxiliary buffer. `atomic` and `spinlock` clear the accumulation state and
synchronize the current stream across ranks before launch. Do not manually set
`workspace.prepared` or clear buffers between calls; preparation is managed by
the operation. Preallocate locks using the smallest tile sizes you plan to use;
an undersized preallocated lock array raises `ValueError` collectively at launch.

`async_op=True` omits the final barrier but still performs host-side workspace
agreement. It is not a promise of a fully nonblocking or CUDA-graph-capturable
call. Call `ctx.barrier()` before host access, releasing buffers, or handing the
result to work that is not ordered on the current stream. Keep workspace calls
ordered on one stream; concurrent reuse from multiple streams is not supported.

## all_gather_matmul
```{eval-rst}
.. automethod:: iris.ops.OpsNamespace.all_gather_matmul
```

## matmul_all_gather
```{eval-rst}
.. automethod:: iris.ops.OpsNamespace.matmul_all_gather
```

## matmul_reduce_scatter
```{eval-rst}
.. automethod:: iris.ops.OpsNamespace.matmul_reduce_scatter
```
