# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
:mod:`iris.concurrent` -- concurrent independent compute + collective primitives.

These run an independent GEMM alongside a collective on a single device via
work-stealing scheduling. The operands are unrelated (not producer/consumer);
they contend only for GPU resources.

Namespaces are keyed by the compute operation. Today:

* :mod:`iris.concurrent.gemm` -- GEMM concurrent with a collective, e.g.
  :func:`iris.concurrent.gemm.all_gather`.

Each op supports two overlap models via ``mode``:

* ``"fused"``      -- one persistent kernel, two work-stealing queues.
* ``"concurrent"`` -- two independent persistent work-stealing kernels on
  separate streams.

Pass ``tune=True`` to any op to autotune the CU split / GEMM tile for the given
shape via :mod:`iris.concurrent.autotune` (a top-k, predictor-seeded tuner with
a persistent JSON cache).
"""

from . import autotune, gemm

__all__ = ["autotune", "gemm"]
