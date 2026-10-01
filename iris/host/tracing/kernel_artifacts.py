# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Kernel Artifacts Capture for Iris.

When IRIS_KERNEL_ARTIFACTS_DIR is set, captures Triton compilation artifacts
(TTIR, TTGIR, LLIR, AMDGCN assembly, metadata) for every iris kernel
specialization, organized by algorithm, kernel name, rank, and constexpr combo.

Usage:
    IRIS_KERNEL_ARTIFACTS_DIR=/tmp/iris_artifacts torchrun --nproc_per_node=8 my_script.py

Directory layout:
    $IRIS_KERNEL_ARTIFACTS_DIR/
    ├── all_reduce/
    │   └── persistent_all_reduce_atomic/
    │       └── rank_0/
    │           └── BM32_BN64_fp16_w4/
    │               └── a1b2c3d4e5f6/          # codegen hash (SHA-256 of AMDGCN, first 12 chars)
    │                   ├── metadata.json
    │                   ├── kernel.ttir
    │                   ├── kernel.ttgir
    │                   ├── kernel.llir
    │                   └── kernel.amdgcn
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from iris.host.logging.logging import _log_rank

_artifacts_dir: Optional[Path] = None
_enabled: bool = False


def _init():
    global _artifacts_dir, _enabled
    d = os.environ.get("IRIS_KERNEL_ARTIFACTS_DIR", "").strip()
    if d:
        _artifacts_dir = Path(d)
        _enabled = True


def is_enabled() -> bool:
    return _enabled


def iris_launch(kernel_fn, grid, *args, algorithm: str, rank: int, dtype=None, **kwargs):
    """Launch a Triton kernel and capture artifacts if enabled.

    Drop-in replacement for ``kernel_fn[grid](*args, **kwargs)``.
    When ``IRIS_KERNEL_ARTIFACTS_DIR`` is unset, the only overhead is a single
    boolean check.

    Args:
        kernel_fn: Triton JITFunction (the ``@triton.jit`` decorated function).
        grid: Launch grid tuple.
        *args: Positional arguments forwarded to the kernel.
        algorithm: Algorithm name for directory grouping (e.g. "all_reduce").
        rank: Current global rank.
        dtype: Optional tensor dtype for the spec directory name.
        **kwargs: Keyword arguments forwarded to the kernel (num_warps, etc.).
    """
    _log_rank(
        logging.DEBUG,
        "iris_launch: algorithm=%s kernel=%s grid=%s rank=%d",
        algorithm,
        _get_kernel_name(kernel_fn),
        grid,
        rank,
        rank=rank,
    )
    compiled = kernel_fn[grid](*args, **kwargs)
    if _enabled and compiled is not None:
        kernel_name = _get_kernel_name(kernel_fn)
        _save(compiled, algorithm, kernel_name, rank, dtype, grid)
    return compiled


def _get_kernel_name(kernel_fn) -> str:
    """Extract the kernel function name from a Triton JITFunction."""
    if hasattr(kernel_fn, "fn"):
        return kernel_fn.fn.__name__
    if hasattr(kernel_fn, "__name__"):
        return kernel_fn.__name__
    return str(kernel_fn)


def _save(compiled, algorithm: str, kernel_name: str, rank: int, dtype, grid):
    """Extract and write artifacts from a CompiledKernel."""
    spec_dirname = _build_spec_dirname(compiled, dtype)
    codegen_hash = _codegen_hash(compiled)
    output_dir = _artifacts_dir / algorithm / kernel_name / f"rank_{rank}" / spec_dirname / codegen_hash

    # Dedup: skip if this exact codegen already captured
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        return

    metadata = _extract_metadata(compiled, algorithm, kernel_name, rank, grid, dtype)
    metadata["codegen_hash"] = codegen_hash
    _write_artifacts(output_dir, compiled, metadata)


def _codegen_hash(compiled) -> str:
    """Hash codegen output to detect actual codegen changes.

    Prefers AMDGCN assembly text. Falls back to compiled.hash when
    assembly is unavailable (avoids empty-string hash collisions).
    """
    asm = getattr(compiled, "asm", {}) or {}
    blob = asm.get("amdgcn") or asm.get("hsaco")
    if blob:
        if isinstance(blob, bytes):
            return hashlib.sha256(blob).hexdigest()[:12]
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    # Fallback to Triton's compilation hash
    h = getattr(compiled, "hash", None) or "unknown"
    return hashlib.sha256(h.encode("utf-8")).hexdigest()[:12]


def _build_spec_dirname(compiled, dtype=None) -> str:
    """Build a readable directory name from constexpr values and compilation options.

    Format: BM{m}_BN{n}[_BK{k}]_{dtype}_w{warps}
    Falls back to the compilation hash if no block sizes are found.
    """
    parts = []
    constexprs = _extract_constexprs(compiled)

    # Block sizes
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        short = key.replace("BLOCK_SIZE_", "B")
        if key in constexprs:
            parts.append(f"{short}{constexprs[key]}")

    # dtype
    dtype_str = _dtype_short_name(dtype)
    if dtype_str:
        parts.append(dtype_str)

    # num_warps
    meta = compiled.metadata
    if hasattr(meta, "num_warps"):
        parts.append(f"w{meta.num_warps}")

    if not parts:
        # Fallback: use hash prefix
        h = getattr(compiled, "hash", "unknown")
        return h[:12] if h else "unknown"

    return "_".join(parts)


def _dtype_short_name(dtype) -> Optional[str]:
    """Convert a torch dtype to a short string for directory naming."""
    if dtype is None:
        return None
    s = str(dtype)
    # torch.float16 -> fp16, torch.bfloat16 -> bf16, torch.float32 -> fp32, etc.
    mapping = {
        "torch.float16": "fp16",
        "torch.bfloat16": "bf16",
        "torch.float32": "fp32",
        "torch.float64": "fp64",
        "torch.int8": "i8",
        "torch.int16": "i16",
        "torch.int32": "i32",
        "torch.int64": "i64",
        "torch.float8_e4m3fnuz": "fp8e4m3",
        "torch.float8_e5m2fnuz": "fp8e5m2",
        "torch.float8_e4m3fn": "fp8e4m3fn",
        "torch.float8_e5m2": "fp8e5m2",
    }
    return mapping.get(s, s.replace("torch.", ""))


def _extract_constexprs(compiled) -> Dict[str, Any]:
    """Extract constexpr parameter values from a CompiledKernel.

    Triton stores constexprs in src.constants as {(arg_index,): value}.
    We map them back to parameter names using src.fn.arg_names.
    """
    result = {}
    src = getattr(compiled, "src", None)
    if src is None:
        return result

    constants = getattr(src, "constants", {})
    fn = getattr(src, "fn", None)
    arg_names = getattr(fn, "arg_names", None) if fn else None

    for key, value in constants.items():
        if arg_names and isinstance(key, tuple) and len(key) == 1:
            idx = key[0]
            if 0 <= idx < len(arg_names):
                name = arg_names[idx]
                # Only include simple types in dir name
                if isinstance(value, (int, float, bool, str)):
                    result[name] = value
                elif hasattr(value, "value"):
                    # tl.constexpr wraps a value
                    result[name] = value.value
        elif isinstance(key, str):
            if isinstance(value, (int, float, bool, str)):
                result[key] = value

    return result


def _extract_metadata(compiled, algorithm: str, kernel_name: str, rank: int, grid, dtype) -> dict:
    """Build the metadata dict for a kernel specialization."""
    meta = compiled.metadata
    constexprs = _extract_constexprs(compiled)

    target_info = {}
    if hasattr(meta, "target"):
        t = meta.target
        target_info = {
            "backend": getattr(t, "backend", str(t)),
            "arch": getattr(t, "arch", ""),
            "warp_size": getattr(t, "warp_size", 64),
        }

    result = {
        "kernel_name": kernel_name,
        "algorithm": algorithm,
        "rank": rank,
        "hash": getattr(compiled, "hash", None),
        "target": target_info,
        "constexprs": constexprs,
        "num_warps": getattr(meta, "num_warps", None),
        "num_stages": getattr(meta, "num_stages", None),
        "shared_memory_bytes": getattr(meta, "shared", None),
        "grid": list(grid) if grid else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # n_regs / n_spills are lazily initialized after first run
    if hasattr(compiled, "n_regs") and compiled.n_regs is not None:
        result["num_registers"] = compiled.n_regs
    if hasattr(compiled, "n_spills") and compiled.n_spills is not None:
        result["num_spills"] = compiled.n_spills

    if dtype is not None:
        result["dtype"] = str(dtype)

    return result


def _write_artifacts(output_dir: Path, compiled, metadata: dict):
    """Write all artifact files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write IR artifacts
    asm = getattr(compiled, "asm", {})
    artifact_map = {
        "kernel.ttir": "ttir",
        "kernel.ttgir": "ttgir",
        "kernel.llir": "llir",
        "kernel.amdgcn": "amdgcn",
    }

    for filename, asm_key in artifact_map.items():
        content = asm.get(asm_key)
        if content is not None:
            filepath = output_dir / filename
            if isinstance(content, bytes):
                filepath.write_bytes(content)
            else:
                filepath.write_text(content, encoding="utf-8")

    # Write metadata
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


_init()
