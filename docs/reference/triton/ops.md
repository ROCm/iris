# Fused GEMM + CCL Operations

Fused matrix multiplication and collective communication operations accessible via the `ops` property on the `Iris` instance (e.g. `ctx.ops.matmul_all_reduce(...)`).

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
