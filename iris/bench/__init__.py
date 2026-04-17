# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
iris.bench — GPU Benchmarking Framework

A declarative benchmarking framework for iris that eliminates boilerplate.
Write ~25 lines instead of ~350 to benchmark a GPU kernel.

Execution Model
---------------

Every benchmark function has the signature ``fn(state, ctx)`` where *state*
is a :class:`State` object and *ctx* is an :class:`~iris.Iris` context. The
framework calls each function once per parameter combination. Inside the
function you do three things:

1. **Setup** — allocate tensors, build configs, fill data.  This code runs
   **once** per parameter combination and is **not timed**.

2. **Declare metrics** — call ``state.set_bytes(n)`` and/or
   ``state.set_flops(n)`` so the framework can compute bandwidth / TFLOPS.

3. **Register the kernel** — call ``state.exec(fn)`` with the callable to
   time.  ``exec`` does **not** run the callable; it stores it. After your
   function returns, the framework passes it to ``iris.do_bench()`` which
   handles warmup, cache clearing, barrier synchronization, and CUDA-event
   timing.

The callable registered via ``state.exec()`` is invoked
``1 + n_warmup + n_repeat`` times total.  Only the last ``n_repeat``
invocations are timed.

Per-Iteration Reset (``preamble_fn``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to reset state before **each** invocation (zero output buffers,
reset locks, reinitialize a workspace), pass a ``preamble_fn``::

    state.exec(
        lambda: ctx.ccl.all_gather(out, inp, config=config),
        preamble_fn=lambda: out.zero_(),
    )

``preamble_fn`` runs before every invocation (warmup and timed) but is
**not timed** — it executes before the CUDA start event is recorded. It can
be as heavyweight as needed without affecting measured results.

The ``num_ranks`` Axis
~~~~~~~~~~~~~~~~~~~~~~

``num_ranks`` is a special axis. It controls how many GPU processes are
spawned.  The framework collects all unique ``num_ranks`` values across
registered benchmarks, then launches a separate worker group via
``torch.distributed.launcher.api.elastic_launch`` for each value.
Other axes are iterated inside the worker processes.

If no ``num_ranks`` axis is declared, the benchmark runs with 8 ranks.

Axes & Parameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~~

Stack multiple ``@bench.axis`` decorators to define a sweep. The framework
generates the Cartesian product of all axes. The outermost ``@axis``
decorator is the slowest-varying axis in the output table.

CLI Overrides
~~~~~~~~~~~~~

Any axis can be overridden or filtered from the command line:

- ``--axis_M=1024,2048`` — replace the M axis with these values.
- ``--axis_M=pow2:8:12`` — replace with ``[256, 512, 1024, 2048, 4096]``.
- ``--axis_dtype=fp16`` — run only float16.
- ``--skip_num_ranks=1,2`` — exclude 1- and 2-rank runs.
- ``--benchmark_filter=all_gather`` — regex filter on benchmark name.

Example
-------

::

    import torch
    import iris.bench as bench
    from iris.ccl import Config

    @bench.register
    @bench.axis("num_ranks", [2, 4, 8])
    @bench.axis("M", bench.power_of_two(8, 13))
    @bench.axis("N", [256, 512, 1024])
    @bench.axis("dtype", [torch.float16, torch.float32])
    def all_gather(state, ctx):
        M, N, dtype = state["M"], state["N"], state["dtype"]
        world_size = ctx.get_num_ranks()

        inp = ctx.zeros((M, N), dtype=dtype)
        out = ctx.zeros((world_size * M, N), dtype=dtype)
        inp.fill_(float(ctx.get_rank() + 1))

        state.set_bytes((world_size - 1) * M * N * inp.element_size())

        config = Config(use_gluon=False)
        state.exec(lambda: ctx.ccl.all_gather(out, inp, config=config))

    if __name__ == "__main__":
        bench.main()

Run::

    python bench_all_gather.py
    python bench_all_gather.py --skip_num_ranks=2
    python bench_all_gather.py --axis_M=1024 --benchmark_format=json
"""

from ._core import (
    AxisDef,
    BenchmarkDef,
    Result,
    State,
    axis,
    linear_range,
    power_of_two,
    register,
)
from ._runner import main

__all__ = [
    "AxisDef",
    "BenchmarkDef",
    "Result",
    "State",
    "axis",
    "linear_range",
    "main",
    "power_of_two",
    "register",
]
