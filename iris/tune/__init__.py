# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.tune — Autotuning Framework for Collective Operations

A declarative autotuning framework that finds optimal ``iris.ccl.Config``
parameters for a given kernel, problem size, and GPU topology.

Execution Model
---------------

Every tunable function has the signature ``fn(state, ctx)`` — identical to
``iris.bench``. The difference is that instead of sweeping *problem sizes*,
``iris.tune`` sweeps *kernel configurations* for a **fixed** problem size.

1. **Define the search space** — use ``@tune.search_space`` to declare which
   ``Config`` fields to sweep and their candidate values.

2. **Register the tunable** — use ``@tune.register`` (outermost decorator).

3. **Run** — ``tune.main()`` launches workers, benchmarks every config in
   the search space, picks the best, and optionally caches results to disk.

Search Space
~~~~~~~~~~~~

The search space is the Cartesian product of all ``@tune.search_space``
values. Use ``@tune.prune`` to filter invalid combinations before
benchmarking (e.g., ``block_size_m * block_size_n < threads_per_warp *
num_warps`` for Gluon).

Caching
~~~~~~~

Results are cached to ``~/.iris/tune_cache/`` as JSON files, keyed by:
kernel name, problem parameters, GPU architecture, and world size.
Set ``--no_cache`` to skip cache lookup.

Example
-------

::

    import torch
    import iris.tune as tune
    from iris.ccl import Config

    @tune.register
    @tune.search_space("block_size_m", [16, 32, 64, 128])
    @tune.search_space("block_size_n", [32, 64, 128, 256])
    @tune.search_space("comm_sms", [32, 64, 108])
    @tune.search_space("num_warps", [2, 4, 8])
    @tune.search_space("swizzle_size", [2, 4, 8])
    @tune.param("M", 4096)
    @tune.param("N", 4096)
    @tune.param("dtype", torch.float16)
    @tune.prune(lambda cfg: cfg["block_size_m"] * cfg["block_size_n"]
                >= cfg["num_warps"] * 64)
    def all_gather(state, ctx):
        M, N, dtype = state["M"], state["N"], state["dtype"]
        world_size = ctx.get_num_ranks()

        inp = ctx.zeros((M, N), dtype=dtype)
        out = ctx.zeros((world_size * M, N), dtype=dtype)

        config = Config(**state.config_kwargs())
        state.set_bytes((world_size - 1) * M * N * inp.element_size())
        state.exec(
            lambda: ctx.ccl.all_gather(out, inp, config=config),
            preamble_fn=lambda: out.zero_(),
        )

    if __name__ == "__main__":
        tune.main()

Run::

    python tune_all_gather.py --axis_num_ranks=8
    python tune_all_gather.py --top_k=10 --benchmark_format=json
"""

from ._core import (
    TuneDef,
    TuneResult,
    TuneState,
    param,
    prune,
    register,
    search_space,
)
from ._runner import main

__all__ = [
    "TuneDef",
    "TuneResult",
    "TuneState",
    "main",
    "param",
    "prune",
    "register",
    "search_space",
]
