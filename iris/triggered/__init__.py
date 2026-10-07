# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered SDMA: the host pre-arms copy-engine chains, a kernel releases them per batch.

Host side: ``TriggeredPlan`` and the schedule builders in ``iris.triggered.schedule``.
Device side: ``TriggeredView`` (``publish``, ``arrived``, ``wait``).
"""

from iris.mem.triton.triggered import TriggeredView
from iris.triggered.plan import State, TriggeredPlan
from iris.triggered.schedule import Batch, Segment, chunks, waves, whole

__all__ = ["TriggeredPlan", "TriggeredView", "State", "Batch", "Segment", "whole", "chunks", "waves"]
