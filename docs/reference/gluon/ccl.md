# Collective Communication Operations

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

Collective communication operations accessible via the `ccl` attribute on the `IrisGluon` instance (e.g. `ctx.ccl.all_to_all(...)`).

## all_to_all
```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.CCL.all_to_all
```

## all_gather
```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.CCL.all_gather
```

## reduce_scatter
```{eval-rst}
.. automethod:: iris.experimental.iris_gluon.IrisGluon.CCL.reduce_scatter
```
