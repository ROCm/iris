# Collective Communication Operations

```{warning}
The Gluon API is **experimental** and may undergo breaking changes in future releases.
```

Collective communication operations accessible via the `ccl` attribute on the `Iris` instance (e.g. `ctx.ccl.all_to_all(...)`).

## all_to_all
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_to_all
```

## all_gather
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_gather
```

## reduce_scatter
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.reduce_scatter
```
