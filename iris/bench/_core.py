# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Core types, decorators, and range helpers for iris.bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# Dataclasses
@dataclass
class AxisDef:
    """A single sweep axis (name + list of values)."""

    name: str
    values: list[Any]


@dataclass
class BenchmarkDef:
    """A registered benchmark: function + axes."""

    name: str
    fn: Callable
    axes: list[AxisDef]


@dataclass
class Result:
    """Stores results for one (benchmark x parameter-combination) run."""

    benchmark_name: str
    params: dict[str, Any]
    gpu_time_ms: float
    all_times_ms: list[float]
    bandwidth_gbps: float | None = None
    tflops: float | None = None
    counters: dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""
    world_size: int = 1


# Registry
_registry: list[BenchmarkDef] = []


# Skip sentinel
class _SkipCombination(Exception):
    """Raised by :meth:`State.skip` to skip the current parameter combo."""

    def __init__(self, reason: str = ""):
        self.reason = reason


# State — passed into every benchmark function
class State:
    """Per-combination state object passed as the first argument to every
    benchmark function.

    The benchmark function body is the **setup phase** — it runs once per
    parameter combination and is not timed.  Use ``State`` to:

    - Read axis values: ``state["M"]``, ``state.get("dtype")``.
    - Declare metrics: :meth:`set_bytes`, :meth:`set_flops`, :meth:`add_counter`.
    - Register the callable to time: :meth:`exec`.
    - Conditionally skip: :meth:`skip`.
    - Override iteration counts: :meth:`set_warmup`, :meth:`set_repeat`.

    After the benchmark function returns, the framework calls
    ``iris.do_bench()`` with the callable registered via :meth:`exec`.
    """

    def __init__(self, params: dict[str, Any], n_warmup: int, n_repeat: int):
        self._params = params
        self._bytes: int | None = None
        self._flops: int | None = None
        self._counters: dict[str, float] = {}
        self._exec_fn: Callable | None = None
        self._preamble_fn: Callable = lambda: None
        self._n_warmup = n_warmup
        self._n_repeat = n_repeat

    # -- axis access --------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    # -- metric declarations ------------------------------------------------

    def set_bytes(self, n: int) -> None:
        """Declare bytes transferred so the framework can report bandwidth.

        The output table will include a **BW (GB/s)** column computed as
        ``n / 1e9 / (gpu_time_ms * 1e-3)``.
        """
        self._bytes = n

    def set_flops(self, n: int) -> None:
        """Declare FLOPs so the framework can report throughput.

        The output table will include a **TFLOPS** column computed as
        ``n / 1e12 / (gpu_time_ms * 1e-3)``.
        """
        self._flops = n

    def add_counter(self, name: str, value: float) -> None:
        """Add a custom metric column to the output table.

        Call multiple times with different names to add multiple columns.
        """
        self._counters[name] = value

    # -- timing control -----------------------------------------------------

    def set_warmup(self, n: int) -> None:
        """Override the number of warmup iterations (default: 25, or ``--n_warmup``)."""
        self._n_warmup = n

    def set_repeat(self, n: int) -> None:
        """Override the number of timed iterations (default: 100, or ``--n_repeat``)."""
        self._n_repeat = n

    def exec(self, fn: Callable, *, preamble_fn: Callable | None = None) -> None:
        """Register the callable to time.

        This does **not** call *fn* immediately.  After the benchmark
        function returns, the framework passes *fn* to ``iris.do_bench()``
        which runs it ``1 + n_warmup + n_repeat`` times (1 initial call,
        warmup iterations, then timed iterations).

        Parameters
        ----------
        fn:
            The kernel / operation to benchmark.  Only this callable is
            inside the timed region (between CUDA start/end events).
        preamble_fn:
            Optional callable executed before **every** invocation of *fn*
            (warmup and timed).  Runs **outside** the timed region — before
            the CUDA start event is recorded — so it can be arbitrarily
            expensive without affecting results.  Use it to reset output
            buffers, reinitialize locks, rebuild workspaces, etc.

        Example::

            # Zero the output buffer before each iteration
            state.exec(
                lambda: ctx.ccl.all_gather(out, inp, config=config),
                preamble_fn=lambda: out.zero_(),
            )
        """
        self._exec_fn = fn
        if preamble_fn is not None:
            self._preamble_fn = preamble_fn

    # -- skip ---------------------------------------------------------------

    def skip(self, reason: str = "") -> None:
        """Skip this parameter combination.

        Call this during setup to skip combinations that are invalid or
        uninteresting.  The combination appears as ``(skipped)`` in the
        output rather than being silently omitted.

        Example::

            if M < N:
                state.skip("M must be >= N")
        """
        raise _SkipCombination(reason)


# Range helpers
def power_of_two(start_exp: int, end_exp: int) -> list[int]:
    """Return ``[2**start_exp, ..., 2**end_exp]`` inclusive."""
    return [1 << e for e in range(start_exp, end_exp + 1)]


def linear_range(start: int, end: int, step: int) -> list[int]:
    """Return ``[start, start+step, ..., end]`` inclusive."""
    return list(range(start, end + 1, step))


# Decorators
def axis(name: str, values: list[Any]):
    """Define a sweep axis for a benchmark.

    Multiple ``@axis`` decorators stack; the framework generates the
    Cartesian product of all axes at runtime.  The outermost ``@axis``
    is the slowest-varying in the output.

    The axis named ``"num_ranks"`` is special: it controls how many GPU
    processes are spawned rather than being iterated inside a worker.

    Any axis can be overridden (``--axis_M=1024``) or filtered
    (``--skip_dtype=fp32``) from the command line.

    Parameters
    ----------
    name:
        Axis name.  Accessible in the benchmark via ``state["name"]``.
    values:
        List of values to sweep.  Use :func:`power_of_two` or
        :func:`linear_range` for common patterns.
    """

    def decorator(fn: Callable) -> Callable:
        if not hasattr(fn, "_bench_axes"):
            fn._bench_axes = []
        # Prepend so that declaration order matches iteration order
        # (outermost decorator = slowest-varying axis).
        fn._bench_axes.insert(0, AxisDef(name, list(values)))
        return fn

    return decorator


def register(fn: Callable) -> Callable:
    """Register *fn* as a benchmark.  Must be the **outermost** decorator.

    The function must have the signature ``fn(state, ctx)`` where *state*
    is a :class:`State` and *ctx* is an :class:`~iris.Iris` context
    (or ``iris_gluon`` context if ``--use_gluon`` is passed).
    """
    axes: list[AxisDef] = getattr(fn, "_bench_axes", [])
    _registry.append(BenchmarkDef(name=fn.__name__, fn=fn, axes=axes))
    return fn
