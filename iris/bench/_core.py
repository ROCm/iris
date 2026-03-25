# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Core types, decorators, and range helpers for iris.bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: list[BenchmarkDef] = []


# ---------------------------------------------------------------------------
# Skip sentinel
# ---------------------------------------------------------------------------


class _SkipCombination(Exception):
    """Raised by :meth:`State.skip` to skip the current parameter combo."""

    def __init__(self, reason: str = ""):
        self.reason = reason


# ---------------------------------------------------------------------------
# State — passed into every benchmark function
# ---------------------------------------------------------------------------


class State:
    """Per-combination state object passed as the first argument to every
    benchmark function.

    Provides access to current axis values and lets the user declare
    metrics (bytes transferred, FLOPs) and the callable to time.
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
        """Declare the number of bytes transferred (for bandwidth calc)."""
        self._bytes = n

    def set_flops(self, n: int) -> None:
        """Declare the number of floating-point operations (for TFLOPS calc)."""
        self._flops = n

    def add_counter(self, name: str, value: float) -> None:
        """Add a custom metric column."""
        self._counters[name] = value

    # -- timing control -----------------------------------------------------

    def set_warmup(self, n: int) -> None:
        """Override the default number of warmup iterations."""
        self._n_warmup = n

    def set_repeat(self, n: int) -> None:
        """Override the default number of timed iterations."""
        self._n_repeat = n

    def exec(self, fn: Callable, *, preamble_fn: Callable | None = None) -> None:
        """Register the callable to time.

        The framework calls ``iris.do_bench()`` with *fn* **after** the
        benchmark function returns — clean separation of setup and timing.

        Parameters
        ----------
        fn:
            The kernel / operation to benchmark.
        preamble_fn:
            Optional callable executed before each timed iteration
            (maps to ``do_bench``'s ``preamble_fn``).
        """
        self._exec_fn = fn
        if preamble_fn is not None:
            self._preamble_fn = preamble_fn

    # -- skip ---------------------------------------------------------------

    def skip(self, reason: str = "") -> None:
        """Skip this parameter combination."""
        raise _SkipCombination(reason)


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------


def power_of_two(start_exp: int, end_exp: int) -> list[int]:
    """Return ``[2**start_exp, ..., 2**end_exp]`` inclusive."""
    return [1 << e for e in range(start_exp, end_exp + 1)]


def linear_range(start: int, end: int, step: int) -> list[int]:
    """Return ``[start, start+step, ..., end]`` inclusive."""
    return list(range(start, end + 1, step))


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def axis(name: str, values: list[Any]):
    """Attach an :class:`AxisDef` to a benchmark function.

    Multiple ``@axis`` decorators stack; the framework generates the
    Cartesian product of all axes at runtime.
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
    """Register *fn* as a benchmark.  Must be the **outermost** decorator."""
    axes: list[AxisDef] = getattr(fn, "_bench_axes", [])
    _registry.append(BenchmarkDef(name=fn.__name__, fn=fn, axes=axes))
    return fn
