# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Transparent autotuning engine for iris-ccl collective operations.

When a Config field is set to AUTOTUNE (or config=None), the engine
benchmarks candidate configurations on first call and caches the winner.
Subsequent calls with the same shape/dtype/topology hit the cache with
zero overhead.

Environment variables:
    IRIS_AUTOTUNE:         Set to "0" to disable autotuning and use defaults.
    IRIS_AUTOTUNE_BUDGET:  Max candidate configs to benchmark (default: 50).
    IRIS_AUTOTUNE_VERBOSE: Set to "1" to print tuning progress.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys
import threading
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
import torch.distributed as dist

import iris
from .config import AUTOTUNE, Config

# ---------------------------------------------------------------------------
# Tunable fields per collective
# ---------------------------------------------------------------------------
# Fields that the autotuner will search over. Fields not listed here are never
# auto-tuned (e.g., use_gluon, threads_per_warp, num_xcds).
_TUNABLE_FIELDS: dict[str, list[str]] = {
    "all_gather": [
        "block_size_m",
        "block_size_n",
        "swizzle_size",
        "comm_sms",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "all_gather_variant",
    ],
    "all_reduce": [
        "block_size_m",
        "block_size_n",
        "swizzle_size",
        "comm_sms",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "all_reduce_variant",
        "all_reduce_distribution",
    ],
    "reduce_scatter": [
        "block_size_m",
        "block_size_n",
        "swizzle_size",
        "comm_sms",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "all_reduce_distribution",
    ],
    "all_to_all": [
        "block_size_m",
        "block_size_n",
        "swizzle_size",
        "comm_sms",
        "num_warps",
        "num_stages",
        "waves_per_eu",
    ],
}

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "all_gather": {
        "block_size_m": [8, 16, 32, 64],
        "block_size_n": [32, 64, 128, 256],
        "swizzle_size": [2, 4, 6, 8],
        "comm_sms": [32, 48, 64, 80, 96, 108],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [0, 1, 2],
        "all_gather_variant": ["persistent", "partitioned"],
    },
    "all_reduce": {
        "block_size_m": [8, 16, 32, 64],
        "block_size_n": [32, 64, 128, 256],
        "swizzle_size": [2, 4, 6, 8],
        "comm_sms": [32, 48, 64, 80, 96, 108],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [0, 1, 2],
        "all_reduce_variant": ["two_shot", "one_shot", "atomic"],
        "all_reduce_distribution": [0, 1],
    },
    "reduce_scatter": {
        "block_size_m": [8, 16, 32, 64],
        "block_size_n": [32, 64, 128, 256],
        "swizzle_size": [2, 4, 6, 8],
        "comm_sms": [32, 48, 64, 80, 96, 108],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [0, 1, 2],
        "all_reduce_distribution": [0, 1],
    },
    "all_to_all": {
        "block_size_m": [8, 16, 32, 64, 128],
        "block_size_n": [32, 64, 128, 256],
        "swizzle_size": [2, 4, 6, 8],
        "comm_sms": [32, 48, 64, 80, 96, 108],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [0, 1, 2],
    },
}

# Default configs used as fallback when autotuning is disabled.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "all_gather": {"block_size_m": 32, "block_size_n": 64},
    "all_reduce": {"block_size_m": 32, "block_size_n": 64, "all_reduce_distribution": 1},
    "reduce_scatter": {"block_size_m": 32, "block_size_n": 64, "all_reduce_distribution": 1},
    "all_to_all": {"block_size_m": 32, "block_size_n": 128},
}

# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

class AutotuneKey(NamedTuple):
    collective: str
    M: int
    N: int
    dtype: str
    world_size: int
    gpu_arch: str
    fixed_fields: tuple  # sorted tuple of (name, value) for user-fixed fields


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------
_cache: dict[AutotuneKey, Config] = {}
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
_CACHE_DIR = Path.home() / ".iris" / "autotune_cache"


def _cache_path(key: AutotuneKey) -> Path:
    """Deterministic file path for a cache key."""
    # Hash everything except collective (used as prefix for readability)
    h = hashlib.sha256()
    for part in key[1:]:  # skip collective
        h.update(repr(part).encode())
    return _CACHE_DIR / f"{key.collective}_{h.hexdigest()[:16]}.json"


def _save_to_disk(key: AutotuneKey, config: Config) -> None:
    """Persist a tuned config to disk (rank 0 only)."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "key": key._asdict(),
            "config": {f.name: getattr(config, f.name) for f in dataclass_fields(config)},
        }
        path = _cache_path(key)
        path.write_text(json.dumps(data, indent=2, default=str))
    except OSError:
        pass  # non-fatal


def _load_from_disk(key: AutotuneKey) -> Config | None:
    """Load a cached config from disk, or None on miss."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cfg_dict = data["config"]
        # num_xcds and chunk_size are auto-derived, set to None for re-derivation
        cfg_dict["num_xcds"] = None
        cfg_dict["chunk_size"] = None
        return Config(**cfg_dict)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Recursion guard
# ---------------------------------------------------------------------------
_tuning_active = threading.local()


def _is_tuning() -> bool:
    return getattr(_tuning_active, "active", False)


# ---------------------------------------------------------------------------
# Candidate pruning
# ---------------------------------------------------------------------------

def _is_valid_candidate(collective: str, values: dict[str, Any], world_size: int) -> bool:
    """Return False for configurations known to be invalid."""
    comm_sms = values.get("comm_sms")

    # Partitioned all-gather needs comm_sms divisible by world_size
    if collective == "all_gather":
        variant = values.get("all_gather_variant", "persistent")
        if variant == "partitioned" and comm_sms is not None and comm_sms % world_size != 0:
            return False

    # block_size_n must be a power of two for ring_slice_n derivation
    block_size_n = values.get("block_size_n")
    if block_size_n is not None and block_size_n & (block_size_n - 1) != 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Core autotuning
# ---------------------------------------------------------------------------

def _generate_candidates(
    collective: str,
    autotune_field_names: list[str],
    fixed_values: dict[str, Any],
    world_size: int,
    budget: int,
) -> list[dict[str, Any]]:
    """Generate up to `budget` candidate value dicts for the AUTOTUNE fields."""
    spaces = _SEARCH_SPACES.get(collective, {})
    ordered_fields = [f for f in autotune_field_names if f in spaces]

    if not ordered_fields:
        # Nothing to tune — return defaults
        return [{}]

    # Cartesian product over tunable fields
    value_lists = [spaces[f] for f in ordered_fields]
    all_combos = list(itertools.product(*value_lists))

    # Merge with fixed values for pruning
    valid = []
    for combo in all_combos:
        values = dict(zip(ordered_fields, combo))
        merged = {**fixed_values, **values}
        if _is_valid_candidate(collective, merged, world_size):
            valid.append(values)

    # Deterministic sampling so all ranks agree
    if len(valid) > budget:
        rng = random.Random(42)
        valid = rng.sample(valid, budget)

    return valid if valid else [{}]


def _autotune(
    collective: str,
    key: AutotuneKey,
    autotune_field_names: list[str],
    base_config: Config,
    collective_fn: Callable,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    shmem,
    extra_kwargs: dict,
) -> Config:
    """Run the autotuning benchmark loop and return the best Config."""
    budget = int(os.environ.get("IRIS_AUTOTUNE_BUDGET", "50"))
    verbose = os.environ.get("IRIS_AUTOTUNE_VERBOSE", "0") == "1"

    rank = shmem.get_rank()
    world_size = key.world_size

    # Build fixed values dict (non-AUTOTUNE fields from base_config)
    fixed_values = {}
    for f in dataclass_fields(base_config):
        val = getattr(base_config, f.name)
        if val is not AUTOTUNE:
            fixed_values[f.name] = val

    candidates = _generate_candidates(
        collective, autotune_field_names, fixed_values, world_size, budget
    )

    if verbose and rank == 0:
        print(
            f"[iris autotune] {collective}: tuning {len(candidates)} configs "
            f"for shape ({key.M}, {key.N}) dtype={key.dtype}",
            file=sys.stderr,
        )

    best_time = float("inf")
    best_values: dict[str, Any] = {}

    for i, candidate_values in enumerate(candidates):
        # Build a concrete config: start from defaults, overlay fixed, overlay candidate
        resolved = {**fixed_values, **candidate_values}
        # Ensure num_xcds and chunk_size are re-derived
        resolved["num_xcds"] = None
        resolved["chunk_size"] = None

        try:
            trial_config = Config(**resolved)
        except (ValueError, TypeError):
            continue

        # Benchmark this config
        try:
            def _run():
                collective_fn(
                    output_tensor,
                    input_tensor,
                    shmem,
                    config=trial_config,
                    async_op=True,
                    **extra_kwargs,
                )

            time_ms = iris.do_bench(
                _run,
                barrier_fn=shmem.barrier,
                n_warmup=3,
                n_repeat=10,
                return_mode="median",
            )
        except Exception:
            time_ms = float("inf")

        if verbose and rank == 0:
            print(
                f"[iris autotune]   [{i + 1}/{len(candidates)}] "
                f"{candidate_values} -> {time_ms:.3f} ms",
                file=sys.stderr,
            )

        if time_ms < best_time:
            best_time = time_ms
            best_values = candidate_values

    # Rank 0 broadcasts the winner
    if dist.is_initialized() and world_size > 1:
        result = [best_values] if rank == 0 else [None]
        dist.broadcast_object_list(result, src=0)
        best_values = result[0]

    # Build final config
    final_values = {**fixed_values, **best_values}
    final_values["num_xcds"] = None
    final_values["chunk_size"] = None
    best_config = Config(**final_values)

    if verbose and rank == 0:
        print(
            f"[iris autotune] {collective}: best config "
            f"({best_time:.3f} ms): {best_values}",
            file=sys.stderr,
        )

    # Cache
    with _cache_lock:
        _cache[key] = best_config
    if rank == 0:
        _save_to_disk(key, best_config)

    return best_config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_config(
    collective: str,
    config: Config | None,
    collective_fn: Callable,
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    shmem,
    **extra_kwargs,
) -> Config:
    """Resolve AUTOTUNE fields in a Config, using cache or benchmarking.

    This is the main entry point called from each collective's dispatch
    function.  It handles three cases:

    1. config is None  ->  create Config with all tunable fields = AUTOTUNE
    2. config has no AUTOTUNE fields  ->  return unchanged (zero overhead)
    3. config has AUTOTUNE fields  ->  cache lookup, or run autotune()

    Args:
        collective: One of "all_gather", "all_reduce", "reduce_scatter", "all_to_all".
        config: User-supplied Config, or None.
        collective_fn: The collective's dispatch function (e.g., all_gather).
        output_tensor: Output tensor for benchmarking.
        input_tensor: Input tensor for benchmarking.
        shmem: Iris context.
        **extra_kwargs: Extra keyword args forwarded to collective_fn during
            benchmarking (e.g., op=ReduceOp.SUM, group=group).

    Returns:
        A fully concrete Config (no AUTOTUNE fields).
    """
    # Env-var kill switch
    if os.environ.get("IRIS_AUTOTUNE", "1") == "0":
        if config is None:
            defaults = _DEFAULTS.get(collective, {})
            return Config(**defaults)
        autotune_fields = config.get_autotune_fields()
        if not autotune_fields:
            return config
        # Replace AUTOTUNE fields with defaults
        defaults = _DEFAULTS.get(collective, {})
        resolved = {}
        for f in dataclass_fields(config):
            val = getattr(config, f.name)
            if val is AUTOTUNE:
                resolved[f.name] = defaults.get(f.name, Config.__dataclass_fields__[f.name].default)
            else:
                resolved[f.name] = val
        resolved["num_xcds"] = None
        resolved["chunk_size"] = None
        return Config(**resolved)

    # Recursion guard: if we're inside an autotune loop, the collective is
    # being called with a fully concrete config, so just use defaults.
    if _is_tuning():
        if config is None:
            defaults = _DEFAULTS.get(collective, {})
            return Config(**defaults)
        return config

    # Case 1: config is None -> mark all tunable fields as AUTOTUNE
    if config is None:
        config = Config.autotune()

    # Case 2: no AUTOTUNE fields -> fast path, zero overhead
    autotune_fields = config.get_autotune_fields()
    if not autotune_fields:
        return config

    # Case 3: has AUTOTUNE fields -> cache lookup or autotune
    M, N = input_tensor.shape[:2]
    dtype_str = str(input_tensor.dtype)
    world_size = shmem.get_num_ranks()
    gpu_arch = iris.hip.get_arch_string()

    # Build fixed_fields for cache key (non-AUTOTUNE fields that differ from defaults)
    fixed_items = []
    for f in dataclass_fields(config):
        val = getattr(config, f.name)
        if val is not AUTOTUNE and f.name not in ("num_xcds", "chunk_size"):
            fixed_items.append((f.name, val))
    fixed_items.sort()

    key = AutotuneKey(
        collective=collective,
        M=M,
        N=N,
        dtype=dtype_str,
        world_size=world_size,
        gpu_arch=gpu_arch,
        fixed_fields=tuple(fixed_items),
    )

    # Check in-memory cache
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    # Check disk cache
    disk_config = _load_from_disk(key)
    if disk_config is not None:
        with _cache_lock:
            _cache[key] = disk_config
        return disk_config

    # Run autotuning
    _tuning_active.active = True
    try:
        result = _autotune(
            collective,
            key,
            autotune_fields,
            config,
            collective_fn,
            output_tensor,
            input_tensor,
            shmem,
            extra_kwargs,
        )
    finally:
        _tuning_active.active = False

    return result
