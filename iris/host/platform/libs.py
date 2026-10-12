# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Load vendor runtime libraries, reusing the copies already mapped into the process."""

import ctypes
import glob
import logging
import os
from typing import Any, List, Optional, Set, Tuple

logger = logging.getLogger("iris.platform")


def _mapped_library_paths(stem: str) -> List[str]:
    """Distinct files named ``<stem>.so*`` mapped into this process, in /proc/self/maps order."""
    try:
        with open("/proc/self/maps") as maps:
            lines = maps.readlines()
    except OSError:
        return []

    paths: List[str] = []
    seen = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not os.path.basename(path).startswith(f"{stem}.so"):
            continue
        try:
            st = os.stat(path)
            key = (st.st_dev, st.st_ino)
        except OSError:
            key = path
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


_reported: Set[Tuple[str, Tuple[str, ...]]] = set()


def _sibling_library(stem: str, beside: str) -> Optional[str]:
    """``<stem>.so*`` in the directory of the first mapped copy of ``beside``."""
    anchors = _mapped_library_paths(beside)
    if not anchors:
        return None
    candidates = sorted(glob.glob(os.path.join(os.path.dirname(anchors[0]), f"{stem}.so*")))
    return candidates[0] if candidates else None


def _ld_library_path_copy(name: str) -> Optional[str]:
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        path = os.path.join(directory, name)
        if directory and os.path.exists(path):
            return path
    return None


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def load_vendor_library(stem: str, *fallbacks: Optional[str], beside: Optional[str] = None) -> Any:
    """
    Load ``<stem>.so``, preferring a copy that is already mapped, then one in the
    directory of the mapped ``beside`` library, then the first of ``fallbacks``
    that loads. Returns None if none does.

    torch's ROCm wheels bring their own HIP runtime and amd_smi. Resolving the
    library by name instead (LD_LIBRARY_PATH, ldconfig, /opt/rocm) can map a
    second copy from a different ROCm into the same process, which can abort
    during device initialization.
    """
    mapped = _mapped_library_paths(stem)
    preferred = mapped[:1] or ([_sibling_library(stem, beside)] if beside else [])
    for name in preferred + list(fallbacks):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        loaded = tuple(_mapped_library_paths(stem)) or (name,)
        if (stem, loaded) not in _reported:
            _reported.add((stem, loaded))
            if len(loaded) > 1:
                logger.warning(
                    "Multiple copies of %s are loaded in this process: %s. Point LD_LIBRARY_PATH at the ROCm "
                    "that torch uses.",
                    stem,
                    list(loaded),
                )
            else:
                shadowed = _ld_library_path_copy(f"{stem}.so")
                if shadowed and not _same_file(shadowed, loaded[0]):
                    logger.info("Using %s from %s, not %s from LD_LIBRARY_PATH", stem, loaded[0], shadowed)
                else:
                    logger.debug("Using %s from %s", stem, loaded[0])
        return lib
    return None
