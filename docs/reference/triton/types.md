# Tile Types

The tile abstractions the device APIs operate on. All of these are exported from
the top level, so `iris.Tile`, `iris.TensorView` and so on are the import paths.

A `TensorView` wraps a pointer plus the tensor's shape and strides, and hands out
tile pointers and masks. A `TileView` carries a tile's coordinates; a `Tile`
carries its coordinates *and* the data. `AllReduceConfig` is the device-side
configuration consumed by the all-reduce methods on
[Device Context](device-context.md).

## TensorView

```{eval-rst}
.. autoclass:: iris.mem.triton.types.TensorView
   :noindex:
```

## TileView

```{eval-rst}
.. autoclass:: iris.mem.triton.types.TileView
   :noindex:
```

## Tile

```{eval-rst}
.. autoclass:: iris.mem.triton.types.Tile
   :noindex:
```

## AllReduceConfig

```{eval-rst}
.. autoclass:: iris.mem.triton.types.AllReduceConfig
   :noindex:
```

## make_tensor_view

```{eval-rst}
.. autofunction:: iris.mem.triton.types.make_tensor_view
   :noindex:
```
