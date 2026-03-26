# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Core types, decorators, and search space helpers for iris.tune."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SearchAxis:
    """A single search axis (Config field name + candidate values)."""

    name: str
    values: list[Any]


@dataclass
class TuneDef:
    """A registered tunable: function + search space + fixed params + prune fns."""

    name: str
    fn: Callable
    search_axes: list[SearchAxis]
    fixed_params: dict[str, Any]
    prune_fns: list[Callable[[dict[str, Any]], bool]]


@dataclass
class TuneResult:
    """Result for one config trial."""

    tune_name: str
    config_kwargs: dict[str, Any]
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
_registry: list[TuneDef] = []


class TuneState:
    """Per-config-trial state object passed to every tunable function.

    Combines fixed problem parameters (from ``@tune.param``) with
    the current config candidate (from ``@tune.search_space``).
    """

    def __init__(
        self,
        params: dict[str, Any],
        config_trial: dict[str, Any],
        n_warmup: int,
        n_repeat: int,
    ):
        self._params = params
        self._config_trial = config_trial
        self._bytes: int | None = None
        self._flops: int | None = None
        self._counters: dict[str, float] = {}
        self._exec_fn: Callable | None = None
        self._preamble_fn: Callable = lambda: None
        self._n_warmup = n_warmup
        self._n_repeat = n_repeat

    # -- param access -------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Read a fixed param or config trial value."""
        if key in self._params:
            return self._params[key]
        if key in self._config_trial:
            return self._config_trial[key]
        raise KeyError(f"Unknown param or config key: {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._params:
            return self._params[key]
        return self._config_trial.get(key, default)

    def config_kwargs(self) -> dict[str, Any]:
        """Return the current config trial kwargs for ``Config(**state.config_kwargs())``."""
        return dict(self._config_trial)

    # -- metric declarations ------------------------------------------------

    def set_bytes(self, n: int) -> None:
        self._bytes = n

    def set_flops(self, n: int) -> None:
        self._flops = n

    def add_counter(self, name: str, value: float) -> None:
        self._counters[name] = value

    # -- timing control -----------------------------------------------------

    def set_warmup(self, n: int) -> None:
        self._n_warmup = n

    def set_repeat(self, n: int) -> None:
        self._n_repeat = n

    def exec(self, fn: Callable, *, preamble_fn: Callable | None = None) -> None:
        """Register the callable to time (same as ``bench.State.exec``)."""
        self._exec_fn = fn
        if preamble_fn is not None:
            self._preamble_fn = preamble_fn


# Decorators

def search_space(name: str, values: list[Any]):
    """Declare a Config field to sweep during tuning.

    Parameters
    ----------
    name:
        Config field name (e.g., ``"block_size_m"``, ``"num_warps"``).
    values:
        Candidate values to try.
    """

    def decorator(fn: Callable) -> Callable:
        if not hasattr(fn, "_tune_search_axes"):
            fn._tune_search_axes = []
        fn._tune_search_axes.insert(0, SearchAxis(name, list(values)))
        return fn

    return decorator


def param(name: str, value: Any):
    """Declare a fixed problem parameter (e.g., M, N, dtype).

    Parameters
    ----------
    name:
        Parameter name, accessible via ``state["name"]``.
    value:
        Fixed value for this tuning run.
    """

    def decorator(fn: Callable) -> Callable:
        if not hasattr(fn, "_tune_params"):
            fn._tune_params = {}
        fn._tune_params[name] = value
        return fn

    return decorator


def prune(fn: Callable[[dict[str, Any]], bool]):
    """Add a pruning predicate. Configs where ``fn(config_kwargs)`` is False
    are skipped before benchmarking.

    Parameters
    ----------
    fn:
        Takes a dict of config kwargs, returns True to keep, False to prune.
    """

    def decorator(target: Callable) -> Callable:
        if not hasattr(target, "_tune_prune_fns"):
            target._tune_prune_fns = []
        target._tune_prune_fns.append(fn)
        return target

    return decorator


def register(fn: Callable) -> Callable:
    """Register a tunable function. Must be the outermost decorator."""
    search_axes: list[SearchAxis] = getattr(fn, "_tune_search_axes", [])
    fixed_params: dict[str, Any] = getattr(fn, "_tune_params", {})
    prune_fns: list[Callable] = getattr(fn, "_tune_prune_fns", [])
    _registry.append(
        TuneDef(
            name=fn.__name__,
            fn=fn,
            search_axes=search_axes,
            fixed_params=fixed_params,
            prune_fns=prune_fns,
        )
    )
    return fn
