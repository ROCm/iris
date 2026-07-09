# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Barrier collective — CCL API entry point.

This module provides the CCL-facing ``launch()`` wrapper for the barrier
operation. It manages per-group flag allocation on the symmetric heap and
delegates to ``distributed_device_barrier()`` in
``iris.host.distributed.helpers``, which launches the pre-existing
``_device_barrier_kernel`` Triton kernel.

No new kernel is defined here. The underlying kernel uses a 2-phase
arrive-wait pattern with atomic epoch counters (matching RCCL's
device-side LSA barrier design):
  1. Each rank atomically increments its own flag (release semantics)
  2. Each rank polls every remote rank's flag until it reaches the
     target epoch (acquire semantics)

See ``iris.host.distributed.helpers._device_barrier_kernel`` for the
kernel implementation and ``iris.host.distributed.helpers.distributed_device_barrier``
for the launcher.
"""

import torch

from iris.host.distributed.helpers import distributed_device_barrier


def launch(ctx, group=None):
    """
    Launch the barrier kernel.

    Allocates (or reuses) a per-group flags tensor on the symmetric heap,
    then delegates to ``distributed_device_barrier()`` which launches the
    single-CTA Triton barrier kernel (``_device_barrier_kernel``).

    Args:
        ctx: Iris instance
        group: ProcessGroup or None
    """
    # Reuse the CCL barrier state from ctx (initialized in Iris.__init__)
    # — same flags tensor pattern as ctx.device_barrier() but keyed
    # separately for the CCL interface

    if group not in ctx._ccl_barrier_state:
        ctx._ccl_barrier_state[group] = ctx.zeros((ctx.num_ranks,), dtype=torch.int32)

    flags = ctx._ccl_barrier_state[group]
    heap_bases = ctx.get_heap_bases()

    distributed_device_barrier(flags, group, ctx.get_rank(), ctx.num_ranks, heap_bases)
