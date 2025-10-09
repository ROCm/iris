# SPDX-License-Identifier: MIT

"""
Backend auto-detection for Iris.

Automatically detects and loads the appropriate GPU backend (CUDA or HIP)
based on what's available on the system. It tries CUDA first, then falls back to HIP.

The backend can be forced by setting the IRIS_BACKEND environment variable to 'cuda' or 'hip'.
"""

import ctypes
import os
from types import ModuleType
from typing import NamedTuple


class _Backend(NamedTuple):
    name: str
    module: ModuleType

    def get_name(self):  # for re-export
        return self.name


def _detect_and_load() -> _Backend:
    """Detect available GPU runtime and load corresponding backend."""

    def library_exists(lib_path):
        try:
            ctypes.cdll.LoadLibrary(lib_path)
            return True
        except OSError:
            return False

    def backend_allowed(name):
        """Backend is allowed based on IRIS_BACKEND env var."""
        return not forced or forced == name

    forced = os.getenv("IRIS_BACKEND", "").lower()
    if forced and forced not in ("cuda", "hip"):
        raise ValueError(f"Invalid IRIS_BACKEND='{forced}'. Must be 'cuda' or 'hip'.")

    if library_exists("libcudart.so") and backend_allowed("cuda"):
        from . import cuda

        return _Backend("cuda", cuda)

    if library_exists("libamdhip64.so") and backend_allowed("hip"):
        from . import hip

        return _Backend("hip", hip)

    forced_msg = f"IRIS_BACKEND={forced} but {forced.upper()} runtime not found. " if forced else ""
    raise RuntimeError(
        f"No GPU backend available. {forced_msg}"
        "Iris requires either CUDA or HIP runtime. "
        "Please install CUDA (NVIDIA) or ROCm (AMD) to use Iris."
    )


_backend = _detect_and_load()  # Load backend at import time
# Re-export backend funcs
backend_name = _backend.get_name
set_device = _backend.module.set_device
get_cu_count = _backend.module.get_cu_count
count_devices = _backend.module.count_devices
get_ipc_handle = _backend.module.get_ipc_handle
open_ipc_handle = _backend.module.open_ipc_handle
get_wall_clock_rate = _backend.module.get_wall_clock_rate
get_device_id = _backend.module.get_device_id
get_arch_string = _backend.module.get_arch_string
get_num_xcc = _backend.module.get_num_xcc
malloc_fine_grained = _backend.module.malloc_fine_grained
malloc = _backend.module.malloc
free = _backend.module.free
get_runtime_version = _backend.module.get_runtime_version


__all__ = [
    "backend_name",
    "set_device",
    "get_cu_count",
    "count_devices",
    "get_ipc_handle",
    "open_ipc_handle",
    "get_wall_clock_rate",
    "get_device_id",
    "get_arch_string",
    "get_num_xcc",
    "malloc_fine_grained",
    "malloc",
    "free",
    "get_runtime_version",
]
