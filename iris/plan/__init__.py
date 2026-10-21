# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Collective plans: copy engines send data to every peer as a kernel marks it ready.

Host side: ``CollectivePlan`` and the schedule builders in ``iris.plan.schedule``.
Device side: ``CollectiveView`` (``mark_ready``, ``arrived``, ``wait``, ``owns``).
"""

from iris.mem.triton.plan import CollectiveView
from iris.plan.collective import CollectivePlan, State
from iris.plan.schedule import Chunk, Region, chunks, tile_chunks, whole

__all__ = ["CollectivePlan", "CollectiveView", "State", "Chunk", "Region", "whole", "chunks", "tile_chunks"]
