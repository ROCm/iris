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

### TensorView.tile_ptr
```{eval-rst}
.. automethod:: iris.mem.triton.types.TensorView.tile_ptr
   :noindex:
```

### TensorView.tile_ptr_from_indices
```{eval-rst}
.. automethod:: iris.mem.triton.types.TensorView.tile_ptr_from_indices
   :noindex:
```

### TensorView.offset_tile_ptr
```{eval-rst}
.. automethod:: iris.mem.triton.types.TensorView.offset_tile_ptr
   :noindex:
```

### TensorView.offset
```{eval-rst}
.. automethod:: iris.mem.triton.types.TensorView.offset
   :noindex:
```

### TensorView.number_of_tiles
```{eval-rst}
.. automethod:: iris.mem.triton.types.TensorView.number_of_tiles
   :noindex:
```

## TileView

```{eval-rst}
.. autoclass:: iris.mem.triton.types.TileView
   :noindex:
```

### TileView.layout
```{eval-rst}
.. automethod:: iris.mem.triton.types.TileView.layout
   :noindex:
```

### TileView.indices
```{eval-rst}
.. automethod:: iris.mem.triton.types.TileView.indices
   :noindex:
```

## Tile

```{eval-rst}
.. autoclass:: iris.mem.triton.types.Tile
   :noindex:
```

### Tile.layout
```{eval-rst}
.. automethod:: iris.mem.triton.types.Tile.layout
   :noindex:
```

### Tile.indices
```{eval-rst}
.. automethod:: iris.mem.triton.types.Tile.indices
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
