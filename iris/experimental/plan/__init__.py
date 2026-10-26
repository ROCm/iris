# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Collective plans: copy engines send data to every peer as a kernel marks it ready.

Host side: ``CollectivePlan`` and the schedule builders in ``iris.experimental.plan.schedule``.
Device side: ``CollectiveView`` (``mark_ready``, ``arrived``, ``wait``, ``owns``).
"""

from iris.experimental.plan.triton import CollectiveView
from iris.experimental.plan.collective import CollectivePlan, State
from iris.experimental.plan.schedule import Chunk, Region, chunks, tile_chunks, whole

__all__ = ["CollectivePlan", "CollectiveView", "State", "Chunk", "Region", "whole", "chunks", "tile_chunks"]
