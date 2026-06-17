# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Auto-selection mechanism for fused AG+MM kernel configurations.

Given problem dimensions (M, N, K), transpose mode, world_size, and GPU
architecture, this module selects the best known configuration or returns
a sensible default. For world sizes where iris AG+MM is known to lose
against PyTorch (ws<8), the default disables iris and signals fallback.

Config files live under:
    configs/{arch}/{transpose}/ws{N}.json

Each config file contains:
  - Per-shape champion configs with all kernel parameters in a flat "params" dict
  - A "default_params" dict with architecture-appropriate defaults
  - Params include FusedConfig fields (block_size_m, etc.) and HBM buffer
    kernel params (k_per_flag, num_fetch_sms, num_warps, num_stages, etc.)

Transpose coverage:
    The iris AG+MM kernel (`_fused_all_gather_matmul_kernel`) uses stride-based
    addressing (`stride_am, stride_ak, stride_bk, stride_bn`), so transpose
    layouts are handled implicitly by tensor strides. Config files exist for
    all four layouts (NN, TN, NT, TT) under each architecture directory.
    Only NN has per-shape champion configs from benchmarking (3,489 trials).
    TN/NT/TT files contain heuristic defaults only (empty shapes dict) and are
    marked enabled at ws>=8 to allow heuristic fallback. All transposes at ws<8
    are disabled (NO-GO based on NN benchmarks).

Usage:
    >>> from auto_config import select_ag_mm_config
    >>> result = select_ag_mm_config(M=131072, N=16384, K=16384, world_size=8)
    >>> if result.enabled:
    ...     config = result.to_fused_config()
    ...     hbm_params = result.hbm_buffer_params  # k_per_flag, num_fetch_sms, etc.
    ...     shmem.ops.all_gather_matmul(output, A, B, config=config)
    ... else:
    ...     # Fallback to PyTorch all_gather + matmul
    ...     ...

    >>> # List all regression test sizes
    >>> from auto_config import load_regression_sizes
    >>> sizes = load_regression_sizes()
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from iris.ops.config import FusedConfig

# Config files live alongside this module
_CONFIGS_DIR = Path(__file__).parent / "configs"

# FusedConfig field names — everything else in "params" is an HBM buffer param
_FUSED_CONFIG_FIELDS = {f.name for f in FusedConfig.__dataclass_fields__.values()}

# HBM buffer param names (kernel launch params, not FusedConfig fields)
_HBM_BUFFER_FIELDS = {
    "k_per_flag",
    "num_fetch_sms",
    "num_fetch_stages",
    "first_stage_fetch_sms",
    "fetch_block_m",
    "fetch_block_k",
    "num_warps",
    "num_stages",
}

# In-memory cache: (arch, transpose, world_size) -> loaded JSON data
_config_cache: Dict[Tuple[str, str, int], dict] = {}

# Cached GPU architecture detection result
_detected_arch: Optional[str] = None

# Supported transpose modes. The AG+MM kernel only supports NN layout.
# TN/NT/TT would require kernel-level changes to permute strides.
SUPPORTED_TRANSPOSES = ("NN",)

# Supported GPU architectures with tuned configs
SUPPORTED_ARCHITECTURES = ("mi300x", "mi355x")

# Map gfx target IDs to architecture names used in config paths
_GFX_TO_ARCH = {
    "gfx942": "mi300x",  # MI300X, MI300A
    "gfx950": "mi355x",  # MI355X
}


def detect_gpu_arch() -> str:
    """Auto-detect GPU architecture from the current system.

    Detection order:
    1. IRIS_GPU_ARCH environment variable (override)
    2. rocm-smi --showproductname parsing
    3. rocminfo gfx target parsing
    4. Falls back to "mi300x" (most common deployment target)

    Returns:
        Architecture string (e.g., "mi300x") suitable for config lookup.
    """
    global _detected_arch
    if _detected_arch is not None:
        return _detected_arch

    # 1. Environment variable override
    env_arch = os.environ.get("IRIS_GPU_ARCH", "").strip().lower()
    if env_arch:
        _detected_arch = env_arch
        return _detected_arch

    # 2. Try rocminfo for gfx target
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line_stripped = line.strip().lower()
                if "name:" in line_stripped and "gfx" in line_stripped:
                    for gfx_id, arch_name in _GFX_TO_ARCH.items():
                        if gfx_id in line_stripped:
                            _detected_arch = arch_name
                            return _detected_arch
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 3. Fallback to MI300X (most common deployment target)
    _detected_arch = "mi300x"
    return _detected_arch


@dataclass
class AutoConfigResult:
    """Result of auto-config lookup.

    Attributes:
        enabled: If False, iris AG+MM should NOT be used; fallback to PyTorch.
        config_params: Dict of FusedConfig parameters (only valid if enabled=True).
        hbm_buffer_params: Dict of HBM buffer-specific kernel params
            (k_per_flag, num_fetch_sms, num_fetch_stages, first_stage_fetch_sms).
        source: Human-readable description of where this config came from.
        shape_key: The MxNxK key that matched (None if heuristic/default).
        expected_iris_ms: Expected kernel time in ms on target GPU (None if unknown).
    """

    enabled: bool = False
    config_params: Dict = field(default_factory=dict)
    hbm_buffer_params: Dict = field(default_factory=dict)
    source: str = "default"
    shape_key: Optional[str] = None
    expected_iris_ms: Optional[float] = None

    def to_fused_config(self) -> FusedConfig:
        """Convert to FusedConfig for use with iris.ops functions.

        Raises:
            RuntimeError: If this config is disabled (enabled=False).
        """
        if not self.enabled:
            raise RuntimeError(
                f"Cannot create FusedConfig: iris AG+MM is disabled for this "
                f"configuration. Reason: {self.source}. "
                f"Use PyTorch all_gather + matmul instead."
            )
        # Filter to only fields FusedConfig accepts
        valid_fields = {f.name for f in FusedConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in self.config_params.items() if k in valid_fields}
        return FusedConfig(**filtered)


def _split_params(params: Dict) -> Tuple[Dict, Dict]:
    """Split a flat params dict into (config_params, hbm_buffer_params).

    FusedConfig fields go into config_params.
    Everything else (num_warps, num_stages, k_per_flag, etc.) goes into hbm_buffer_params.
    """
    config_params = {}
    hbm_params = {}
    for k, v in params.items():
        if k in _FUSED_CONFIG_FIELDS:
            config_params[k] = v
        else:
            hbm_params[k] = v
    return config_params, hbm_params


def _extract_shape_params(shape_data: Dict) -> Tuple[Dict, Dict]:
    """Extract config_params and hbm_buffer_params from shape data.

    Supports both the new flat "params" format and the legacy split
    "config" + "hbm_buffer_params" format for backward compatibility.
    """
    if "params" in shape_data:
        return _split_params(shape_data["params"])
    return shape_data.get("config", {}), shape_data.get("hbm_buffer_params", {})


def _extract_default_params(data: Dict) -> Optional[Tuple[Dict, Dict]]:
    """Extract default config_params and hbm_buffer_params from file-level defaults.

    Supports both "default_params" (flat) and legacy "default_config" + "default_hbm_buffer_params".
    Returns None if no defaults are available.
    """
    if "default_params" in data and data["default_params"] is not None:
        return _split_params(data["default_params"])
    default_config = data.get("default_config")
    if default_config:
        return default_config, data.get("default_hbm_buffer_params", {})
    return None


def _load_config_file(arch: str, transpose: str, world_size: int) -> Optional[dict]:
    """Load and cache a config JSON file.

    Args:
        arch: GPU architecture identifier (e.g., "mi300x").
        transpose: Transpose mode (e.g., "NN", "NT", "TN", "TT").
        world_size: Number of ranks.

    Returns:
        Parsed JSON dict, or None if file doesn't exist.
    """
    cache_key = (arch, transpose, world_size)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    config_path = _CONFIGS_DIR / arch / transpose / f"ws{world_size}.json"
    if not config_path.exists():
        _config_cache[cache_key] = None
        return None

    with open(config_path, "r") as f:
        data = json.load(f)

    _config_cache[cache_key] = data
    return data


def _load_default_config() -> dict:
    """Load the global default config."""
    default_path = _CONFIGS_DIR / "default_config.json"
    if default_path.exists():
        with open(default_path, "r") as f:
            return json.load(f)
    return {}


def _find_nearest_shape(M: int, N: int, K: int, shapes: dict, tolerance: float = 0.15) -> Optional[str]:
    """Find the nearest matching shape in the config database.

    Uses log-space geometric distance to find shapes that are structurally
    similar (within `tolerance` ratio per dimension). This avoids falling
    back to heuristic when the user's problem is close to a champion shape.

    Args:
        M, N, K: Target dimensions.
        shapes: Dict of shape_key -> shape_data from the config file.
        tolerance: Max fractional distance per dimension (default 15%).

    Returns:
        The shape_key of the nearest match, or None if no shape is close enough.
    """
    import math

    best_key = None
    best_dist = float("inf")

    for shape_key, shape_data in shapes.items():
        sm, sn, sk = shape_data["M"], shape_data["N"], shape_data["K"]

        # Check per-dimension ratio tolerance
        if sm == 0 or sn == 0 or sk == 0:
            continue
        rm = abs(M - sm) / sm
        rn = abs(N - sn) / sn
        rk = abs(K - sk) / sk

        if rm > tolerance or rn > tolerance or rk > tolerance:
            continue

        # Geometric distance in log space
        dist = math.sqrt(
            math.log(max(M, 1) / max(sm, 1)) ** 2
            + math.log(max(N, 1) / max(sn, 1)) ** 2
            + math.log(max(K, 1) / max(sk, 1)) ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_key = shape_key

    return best_key


def _apply_heuristic(M: int, N: int, K: int, arch: str = "mi300x") -> Tuple[Dict, Dict]:
    """Apply heuristic rules to generate config + HBM buffer params.

    Based on optimization data:
    - MI300X: 3,489 measured trials
    - MI355X: Optuna TPE + broad sweep

    Args:
        M: Rows dimension.
        N: Columns dimension.
        K: Reduction dimension.
        arch: GPU architecture for arch-specific heuristics.

    Returns:
        Tuple of (config_params dict, hbm_buffer_params dict).
    """
    bk = 64
    num_k_blocks = K // bk

    if arch == "mi355x":
        bm = 256
        num_m_tiles = M // bm
        gm = 4 if M <= 32768 else 8
        config_params = {
            "block_size_m": bm,
            "block_size_n": 256,
            "block_size_k": bk,
            "group_size_m": gm,
            "num_warps": 8,
            "num_stages": 2,
            "num_xcds": 8,
            "allow_tf32": True,
        }
        kpf = 8 if num_k_blocks <= 512 else 16
        while num_k_blocks % kpf != 0 and kpf > 1:
            kpf //= 2
        hbm_params = {
            "k_per_flag": kpf,
            "num_fetch_sms": 16,
            "num_fetch_stages": 1,
            "first_stage_fetch_sms": 52,
        }
        return config_params, hbm_params

    # MI300X heuristics
    if M <= 16384:
        bm = 128
    else:
        bm = 256

    num_m_tiles = M // bm

    if M <= 8192:
        gm = 8
    elif M <= 16384:
        gm = 16
    else:
        gm = 24

    config_params = {
        "block_size_m": bm,
        "block_size_n": 256,
        "block_size_k": bk,
        "group_size_m": gm,
        "num_warps": 8,
        "num_stages": 2,
        "num_xcds": 8,
        "allow_tf32": True,
    }

    if num_k_blocks >= 512:
        kpf = 64
    elif num_k_blocks >= 128:
        kpf = 16
    elif num_k_blocks >= 64:
        kpf = 8
    else:
        kpf = 4
    while num_k_blocks % kpf != 0 and kpf > 1:
        kpf //= 2

    if num_m_tiles <= 8:
        fs = 4
    elif num_m_tiles <= 32:
        fs = 16
    elif num_m_tiles <= 128:
        fs = 32
    else:
        fs = 52

    if num_m_tiles >= 512:
        nfs = 4
    elif num_m_tiles >= 64:
        nfs = 2
    else:
        nfs = 1

    hbm_params = {
        "k_per_flag": kpf,
        "num_fetch_sms": fs,
        "num_fetch_stages": nfs,
        "first_stage_fetch_sms": 64,
    }

    return config_params, hbm_params


def select_ag_mm_config(
    M: int,
    N: int,
    K: int,
    world_size: int,
    transpose: str = "NN",
    arch: str = "auto",
) -> AutoConfigResult:
    """Select the best AG+MM config for the given problem.

    Lookup order:
    1. Exact shape match in benchmark/ops/all_gather_matmul/configs/{arch}/{transpose}/ws{world_size}.json
    2. Heuristic-based config from the same file's defaults
    3. Global default from benchmark/ops/all_gather_matmul/configs/default_config.json

    For world sizes where iris is known to lose (ws<8 on MI300X), returns
    a disabled result signaling fallback to PyTorch.

    Args:
        M: Number of rows (or M_local * world_size for AG+MM).
        N: Number of columns.
        K: Reduction dimension.
        world_size: Number of ranks in the communicator.
        transpose: Transpose mode ("NN", "NT", "TN", "TT"). Default "NN".
        arch: GPU architecture ("mi300x", etc.) or "auto" to auto-detect.
            Default "auto". Set IRIS_GPU_ARCH env var to override.

    Returns:
        AutoConfigResult with .enabled indicating whether to use iris,
        .to_fused_config() to get the FusedConfig if enabled, and
        .hbm_buffer_params with kernel-specific parameters.

    Example:
        >>> result = select_ag_mm_config(131072, 16384, 16384, world_size=8)
        >>> result.enabled
        True
        >>> config = result.to_fused_config()
        >>> result.hbm_buffer_params
        {'k_per_flag': 32, 'num_fetch_sms': 4, 'num_fetch_stages': 64, 'first_stage_fetch_sms': 52}

        >>> result = select_ag_mm_config(4096, 4096, 4096, world_size=2)
        >>> result.enabled
        False
    """
    transpose = transpose.upper()
    if arch == "auto":
        arch = detect_gpu_arch()
    else:
        arch = arch.lower()

    # Step 1: Try to load the specific config file
    data = _load_config_file(arch, transpose, world_size)

    if data is not None:
        # Check if this world_size is enabled
        if not data.get("enabled", True):
            return AutoConfigResult(
                enabled=False,
                source=f"Disabled by config: {arch}/{transpose}/ws{world_size}.json — {data.get('reason', 'no reason given')}",
            )

        # Look for exact shape match
        shape_key = f"{M}x{N}x{K}"
        shapes = data.get("shapes", {})
        if shape_key in shapes:
            shape_data = shapes[shape_key]
            cfg, hbm = _extract_shape_params(shape_data)
            return AutoConfigResult(
                enabled=True,
                config_params=cfg,
                hbm_buffer_params=hbm,
                source=f"Exact match: {arch}/{transpose}/ws{world_size}.json [{shape_data.get('label', shape_key)}]",
                shape_key=shape_key,
                expected_iris_ms=shape_data.get("expected_iris_ms"),
            )

        # No exact match — try nearest champion shape (within 15% per dim)
        nearest_key = _find_nearest_shape(M, N, K, shapes)
        if nearest_key is not None:
            nearest_data = shapes[nearest_key]
            cfg, hbm = _extract_shape_params(nearest_data)
            return AutoConfigResult(
                enabled=True,
                config_params=cfg,
                hbm_buffer_params=hbm,
                source=f"Nearest match: {arch}/{transpose}/ws{world_size}.json [{nearest_data.get('label', nearest_key)}] (target {M}x{N}x{K} ≈ {nearest_key})",
                shape_key=nearest_key,
                expected_iris_ms=nearest_data.get("expected_iris_ms"),
            )

        # No nearby match — use heuristic + file defaults
        defaults = _extract_default_params(data)
        if defaults is not None:
            file_default_config, file_default_hbm = defaults
            heuristic_config, heuristic_hbm = _apply_heuristic(M, N, K, arch=arch)
            merged_config = {**file_default_config, **heuristic_config}
            merged_hbm = {**file_default_hbm, **heuristic_hbm}
            return AutoConfigResult(
                enabled=True,
                config_params=merged_config,
                hbm_buffer_params=merged_hbm,
                source=f"Heuristic (no exact shape match in {arch}/{transpose}/ws{world_size}.json)",
            )

    # Step 2: No config file found — check global default
    default_data = _load_default_config()
    ws_gate = default_data.get("world_size_gate", {})
    min_ws = ws_gate.get("min_world_size", 8)

    if world_size < min_ws:
        return AutoConfigResult(
            enabled=False,
            source=f"world_size={world_size} < min_world_size={min_ws} (global default). {ws_gate.get('reason', '')}",
        )

    # World size OK but no specific config — apply heuristic
    heuristic_config, heuristic_hbm = _apply_heuristic(M, N, K, arch=arch)
    return AutoConfigResult(
        enabled=True,
        config_params=heuristic_config,
        hbm_buffer_params=heuristic_hbm,
        source=f"Heuristic fallback (no config file for {arch}/{transpose}/ws{world_size})",
    )


def list_known_shapes(
    world_size: int,
    transpose: str = "NN",
    arch: str = "mi300x",
) -> list:
    """List all known shape configurations for a given world_size/transpose/arch.

    Returns:
        List of dicts with keys: shape_key, label, M, N, K, expected_iris_ms.
    """
    data = _load_config_file(arch, transpose.upper(), world_size)
    if data is None or not data.get("enabled", True):
        return []

    result = []
    for shape_key, shape_data in data.get("shapes", {}).items():
        result.append(
            {
                "shape_key": shape_key,
                "label": shape_data.get("label", ""),
                "M": shape_data["M"],
                "N": shape_data["N"],
                "K": shape_data["K"],
                "expected_iris_ms": shape_data.get("expected_iris_ms"),
            }
        )

    result.sort(key=lambda x: x.get("expected_iris_ms") or float("inf"))
    return result


def load_regression_sizes() -> List[Dict]:
    """Load regression test sizes from the JSON config file.

    Returns:
        List of regression size dicts, each with: name, M, N, K, tier,
        description, world_sizes, expected, regression_threshold_pct.
    """
    reg_path = _CONFIGS_DIR / "regression_sizes.json"
    if not reg_path.exists():
        return []
    with open(reg_path, "r") as f:
        data = json.load(f)
    return data.get("sizes", [])


def clear_config_cache():
    """Clear the in-memory config cache. Useful after modifying config files."""
    _config_cache.clear()
