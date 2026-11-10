# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Which host queue a chunk's chains run on.

A **lane** is one independent ordered queue to a peer; ``Chunk.lane`` picks it. rocm-xio calls the
index of a queue within a GPU pair a "channel" (``channel_idx``), and that is what the ``channel=``
argument of Iris's host ``put`` and ``quiet`` takes. A lane is neither a route nor a share of a
collective's tiles.

``queue_index`` is the one place that turns a chunk's lane into the queue index rocm-xio is given, so
the plan and the ledger speak in queues and the mapping can change without touching ``Chunk.lane``.
"""


def queue_index(lane):
    """The host queue index (rocm-xio's ``channel_idx``) that ``lane`` runs on: the lane itself."""
    return lane
