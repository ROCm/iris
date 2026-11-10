<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Scatter, gather, broadcast and all-to-all on the SDMA copy engines

`ctx.ccl.prepare_scatter`, `prepare_gather`, `prepare_broadcast` and `prepare_all_to_all`
(experimental) return [triggered plans](../../docs/conceptual/triggered-plans.md): the copy-engine
chains are prepared once, and a kernel releases them chunk by chunk with
`TriggeredView.mark_ready`. Each part of the tensor lands at the same offset on the ranks it goes to.

| collective | writes | lands |
|---|---|---|
| scatter | the root, every part | rank `r`, part `r` |
| gather | every rank, its own part | the root, every part |
| broadcast | the root, the whole tensor | every rank |
| all-to-all | rank `r`, blocks `(r, j)` | rank `j`, blocks `(r, j)` |

The kernel does not need to know these roles: one program per chunk marks the chunk if this rank
sends it (`owns`), then waits for it (`wait` returns at once for a chunk not sent to this rank).
Each part is cut into chunks of `--tile_rows` rows with `tile_chunks`. Every round checks that the
parts that should land hold their data and the others are untouched.

## Run

From the repository root, start one process per GPU (needs `pip install iris[sdma]`):

```terminal
torchrun --standalone --nproc_per_node=4 examples/35_sdma_collectives/example.py --collective scatter --root 1
torchrun --standalone --nproc_per_node=4 examples/35_sdma_collectives/example.py --collective gather --root 2
torchrun --standalone --nproc_per_node=4 examples/35_sdma_collectives/example.py --collective broadcast
torchrun --standalone --nproc_per_node=4 examples/35_sdma_collectives/example.py --collective all_to_all
```

Across hosts, one rank per host with `mlx5` NICs, the copy engine still triggers and RDMA carries
the data ([triggered plans over RDMA](../../docs/conceptual/rdma-gda.md#triggered-plans-over-rdma)):

```terminal
torchrun --nnodes=2 --nproc_per_node=1 ... examples/35_sdma_collectives/example.py --collective scatter --rdma_device mlx5_0
```

Options: `--m` rows per part (default 64), `--n` columns (128), `--tile_rows` rows per chunk (8),
`--rounds` (3), `--heap_size` (256 MiB). The script raises on a mismatch.
