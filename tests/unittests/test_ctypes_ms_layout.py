# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Packed ctypes structs must declare _layout_ = "ms".

Python 3.14 deprecates the implicit layout chosen by _pack_ (it becomes an
error in 3.19). "ms" is the only legal value alongside _pack_ and matches the
previous implicit behavior, so the declaration is a no-op on older Pythons,
which ignore the unknown class attribute.
"""

import ast
import ctypes
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_STRUCT_SOURCES = {
    _REPO_ROOT / "iris" / "drivers" / "fabric" / "amd.py": {"FABRIC_HANDLE_BYTES": 64},
    _REPO_ROOT / "iris" / "host" / "distributed" / "topology.py": {"_AMDSMI_PPOD_ID_BYTES": 16},
}


def _struct_classes(path, consts):
    tree = ast.parse(path.read_text())
    namespace = {"ctypes": ctypes, **consts}
    classes = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(ast.unparse(base) in ("ctypes.Structure", "ctypes.Union") for base in node.bases):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        classes[node.name] = namespace[node.name]
    return classes


@pytest.mark.parametrize("path", sorted(_STRUCT_SOURCES), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_packed_structs_declare_ms_layout(path):
    classes = _struct_classes(path, _STRUCT_SOURCES[path])
    packed = {name: cls for name, cls in classes.items() if hasattr(cls, "_pack_")}
    assert packed, f"no packed structs found in {path}"
    for name, cls in packed.items():
        assert getattr(cls, "_layout_", None) == "ms", f"{name} sets _pack_ without _layout_ = 'ms'"


@pytest.mark.parametrize(
    ("path", "name", "sizeof"),
    [
        (_REPO_ROOT / "iris" / "drivers" / "fabric" / "amd.py", "_amdsmi_bdf_bitfields", 8),
        (_REPO_ROOT / "iris" / "drivers" / "fabric" / "amd.py", "amdsmi_bdf_t", 8),
        (_REPO_ROOT / "iris" / "drivers" / "fabric" / "amd.py", "amdsmi_fabric_info_t", 320),
        (_REPO_ROOT / "iris" / "host" / "distributed" / "topology.py", "_AmdSmiBdfBitfields", 8),
        (_REPO_ROOT / "iris" / "host" / "distributed" / "topology.py", "_AmdSmiBdf", 8),
        (_REPO_ROOT / "iris" / "host" / "distributed" / "topology.py", "_AmdSmiFabricInfo", 320),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_packed_layout_sizes(path, name, sizeof):
    classes = _struct_classes(path, _STRUCT_SOURCES[path])
    assert ctypes.sizeof(classes[name]) == sizeof
