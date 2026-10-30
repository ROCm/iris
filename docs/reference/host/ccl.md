# Collective Communication Operations

Collective communication operations accessible via the `ccl` attribute on the `Iris` instance (e.g. `ctx.ccl.all_reduce(...)`).

## all_to_all
```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_to_all
```

## all_gather

`Config(engine="sdma")` moves the data with the SDMA copy engines instead of compute units. The
context must be created with `engines=("sdma",)`; that only sets the copy engines up, and each call
still picks its engine (the default is `"cu"`).

```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.all_gather
```

## prepare_all_gather

```{note}
Experimental: returns a [triggered plan](plan.md), which may change between releases without deprecation.
```

```{eval-rst}
.. automethod:: iris.host.iris.Iris.CCL.prepare_all_gather
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
