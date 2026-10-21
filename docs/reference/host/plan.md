# Collective Plans

Copy-engine transfers prepared once and released by kernels, every round. See
[Collective Plans](../../conceptual/collective-plans.md) for the model and its ordering rules.

## CollectivePlan
```{eval-rst}
.. autoclass:: iris.plan.CollectivePlan
   :members: start, wait, next_round, run, close, chunk_id
```

## Schedules
```{eval-rst}
.. autoclass:: iris.plan.Region
.. autoclass:: iris.plan.Chunk
.. autofunction:: iris.plan.whole
.. autofunction:: iris.plan.chunks
.. autofunction:: iris.plan.tile_chunks
```

## CollectiveView (device)
```{eval-rst}
.. autoclass:: iris.plan.CollectiveView
   :members: initialize, mark_ready, arrived, wait, owns
```

## CollectiveView (Gluon device)
```{eval-rst}
.. autoclass:: iris.gluon.CollectiveView
   :members: initialize, mark_ready, arrived, wait, owns
```
