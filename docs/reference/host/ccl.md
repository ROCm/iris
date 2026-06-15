# Collective Communication Operations

Collective communication operations accessible via the `ccl` attribute on the `Iris` instance (e.g. `ctx.ccl.all_reduce(...)`).

## all_to_all
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_to_all
```

## all_gather
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_gather
```

## all_reduce_preamble

```{note}
`all_reduce_preamble` is currently only available when using the Triton backend.
```

```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_reduce_preamble
```

## all_reduce

```{note}
`all_reduce` is currently only available when using the Triton backend.
```

```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_reduce
```

## reduce_scatter
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.reduce_scatter
```
