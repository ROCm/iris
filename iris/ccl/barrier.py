# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Barrier collective operation — public CCL API.

Provides ``barrier()`` as a CCL entry point that delegates to the
pre-existing ``distributed_device_barrier()`` kernel in
``iris.host.distributed.helpers``. The underlying kernel uses
device-side atomic epoch counters on the symmetric heap, following
an arrive-wait pattern with device-side atomic epoch counters.
"""


def barrier(ctx, group=None, async_op=False, config=None):
    """
    Barrier synchronization across all ranks in the group.

    Delegates to the pre-existing device-side barrier kernel
    (``_device_barrier_kernel`` in ``iris.host.distributed.helpers``)
    via the CCL launch wrapper. The kernel uses an epoch-counter approach:
      1. Each rank atomically increments its own flag (release semantics)
      2. Each rank polls all other ranks' flags until they reach the target
         epoch value (acquire semantics)

    This is CUDA graph capturable since all synchronization happens on-device.

    Args:
        ctx: Iris instance
        group: ProcessGroup or None. If None, uses all ranks.
        async_op: If True, skip trailing host barrier. Default: False.
        config: Config instance (unused for zero-byte barrier, used if
                extended to allreduce-based barrier with data).
    """
    from iris.ccl.triton.barrier import launch

    launch(ctx, group=group)
