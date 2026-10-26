# Collective Plans

Experimental (`iris.experimental.plan`): may change between releases without deprecation.

Copy-engine transfers prepared once and released by kernels, every round. See
[Collective Plans](../../conceptual/collective-plans.md) for the model and its ordering rules.

## CollectivePlan
```{eval-rst}
.. autoclass:: iris.experimental.plan.CollectivePlan
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

## CollectiveView (device)
```{eval-rst}
.. autoclass:: iris.experimental.plan.CollectiveView
   :members: initialize, mark_ready, arrived, wait, owns
```
