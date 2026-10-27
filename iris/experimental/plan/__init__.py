# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered plans: copy engines send data to every peer as a kernel marks it ready.

Host side: ``TriggeredPlan`` and the schedule builders in ``iris.experimental.plan.schedule``.
Device side: ``TriggeredView`` (``mark_ready``, ``arrived``, ``wait``, ``owns``).
"""

from iris.experimental.plan.triton import TriggeredView
from iris.experimental.plan.triggered import TriggeredPlan, State
from iris.experimental.plan.schedule import Chunk, Region, chunks, tile_chunks, whole

__all__ = ["TriggeredPlan", "TriggeredView", "State", "Chunk", "Region", "whole", "chunks", "tile_chunks"]
