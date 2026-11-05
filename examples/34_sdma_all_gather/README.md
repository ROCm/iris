<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# All-gather on the SDMA copy engines

Each rank owns `M` rows of an `(world_size * M, N)` output and gathers every other rank's rows, with
the SDMA copy engines moving the data instead of compute units. The context is created with
`engines=("sdma",)`, which sets the copy engines up; each call still chooses its engine.

1. **`ctx.ccl.all_gather(out, inp, config=Config(engine="sdma"))`**: the same collective as the
   default compute-unit all-gather (`engine="cu"`). The example runs both and checks them against
   `torch.distributed.all_gather_into_tensor`.
2. **`ctx.ccl.prepare_all_gather(out, schedule)`** (experimental): a persistent tiled kernel computes
   its rows of `out` (`x * round`) and marks each wave of tiles ready with `TriggeredView.mark_ready`.
   The copy engines send a wave to every peer as soon as all programs have marked it, while the
   kernel computes the next wave. The plan is prepared once and reused for every round. See
   [Triggered Plans](../../docs/conceptual/triggered-plans.md).
3. **The same plan, stream-ordered** (`prepare_ahead=True`): nothing in the loop waits for the GPU.
   The host prepares the next round with `prepare_next()` while the GPU runs this one,
   `wait(blocking=False)` queues the wait, a consumer kernel adds each round's `out` into a sum,
   and `next_round()` queues the round fence. The sum is checked once, after the loop, and rank 0
   prints the host time per round (excluding round 1, which compiles the kernels).

## Run

From the repository root, start one process per GPU (needs `pip install iris[sdma]`):

```terminal
torchrun --standalone --nproc_per_node=2 examples/34_sdma_all_gather/example.py
```

Options: `--m` rows per rank (default 1000), `--n` columns (1024), `--block` producer tile shape
(64 64), `--num_programs` producer programs (32), `--rounds` (5), `--heap_size` (1 GiB). Every result
is checked; the script raises on a mismatch.
