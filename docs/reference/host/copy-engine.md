# Copy Engine (SDMA)

Host-initiated transfers that run on the GPU's SDMA copy engines rather than on
compute units. The host enqueues a descriptor and the DMA engine moves the
bytes, so no kernel launch is involved and no CUs are consumed.

```{note}
These operations require the copy engine to be initialised. See
[`iris.iris()`](class.md) for how to enable it, and the
[Installation guide](../../getting-started/installation.md) for the optional
`rocm-xio` dependency they rely on.
```

Unlike the device-side operations, there is no engine choice to make here: a CU
store is an instruction a wave executes, so the host cannot issue one. Anything
host-initiated goes through the copy engine.

## put

```{eval-rst}
.. automethod:: iris.host.iris.Iris.put
```

## put_tile

```{eval-rst}
.. automethod:: iris.host.iris.Iris.put_tile
```

## put_tiles

```{eval-rst}
.. automethod:: iris.host.iris.Iris.put_tiles
```

## quiet

```{eval-rst}
.. automethod:: iris.host.iris.Iris.quiet
```

## get_copy_engine_ctx

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_copy_engine_ctx
   :no-index:
```
