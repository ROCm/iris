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
