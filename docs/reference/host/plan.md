# Triggered Plans

Experimental (`iris.experimental.plan`): may change between releases without deprecation.

Copy-engine transfers prepared once and released by kernels, every round. See
[Triggered Plans](../../conceptual/triggered-plans.md) for the model and its ordering rules.

## TriggeredPlan
```{eval-rst}
.. autoclass:: iris.experimental.plan.TriggeredPlan
   :members: start, wait, next_round, run, close, chunk_id
```

## Schedules
```{eval-rst}
.. autoclass:: iris.experimental.plan.Region
.. autoclass:: iris.experimental.plan.Chunk
.. autofunction:: iris.experimental.plan.whole
.. autofunction:: iris.experimental.plan.chunks
.. autofunction:: iris.experimental.plan.tile_chunks
```

## TriggeredView (device)
```{eval-rst}
.. autoclass:: iris.experimental.plan.TriggeredView
   :members: initialize, mark_ready, arrived, wait, owns
   :exclude-members: __init__, view, gates, round, rank
```

## TriggeredView (Gluon device)
```{eval-rst}
.. autoclass:: iris.experimental.plan.gluon.TriggeredView
   :members: initialize, mark_ready, arrived, wait, owns
   :exclude-members: __init__, view, gates, round, rank
```
