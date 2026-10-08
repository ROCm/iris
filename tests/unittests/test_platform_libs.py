# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Vendor library loading prefers copies already mapped into the process."""

import io
import logging
import os

import pytest
import torch

from iris.host.platform import libs


def _fake_maps(monkeypatch, paths):
    text = "".join(f"7f0000000000-7f0000001000 r-xp 00000000 08:01 1 {p}\n" for p in paths)
    text += "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0 [stack]\n"
    text += "7ffe00000000-7ffe00001000 rw-p 00000000 00:00 0\n"
    monkeypatch.setattr(libs, "open", lambda path: io.StringIO(text), raising=False)


class TestMappedLibraryPaths:
    def test_matches_versioned_names_and_dedupes_hardlinks(self, monkeypatch, tmp_path):
        core = tmp_path / "core" / "libamdhip64.so.7"
        core.parent.mkdir()
        core.write_bytes(b"hip")
        devel = tmp_path / "devel" / "libamdhip64.so"
        devel.parent.mkdir()
        os.link(core, devel)
        system = tmp_path / "system" / "libamdhip64.so.7.17.0"
        system.parent.mkdir()
        system.write_bytes(b"other hip")
        other = tmp_path / "core" / "libamdhip64_helper.so"
        other.write_bytes(b"not hip")

        _fake_maps(monkeypatch, [core, core, devel, other, system])

        assert libs._mapped_library_paths("libamdhip64") == [str(core), str(system)]

    def test_unreadable_maps(self, monkeypatch):
        def fail(path):
            raise OSError("no procfs")

        monkeypatch.setattr(libs, "open", fail, raising=False)
        assert libs._mapped_library_paths("libamdhip64") == []

    @pytest.mark.skipif(torch.version.hip is None, reason="needs a ROCm build of torch")
    def test_finds_torch_hip_runtime(self):
        assert libs._mapped_library_paths("libamdhip64")


class TestLoadVendorLibrary:
    @pytest.fixture(autouse=True)
    def _fresh_reports(self, monkeypatch):
        monkeypatch.setattr(libs, "_reported", set())

    def test_prefers_mapped_copy(self, monkeypatch):
        opened = []
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: ["/sdk/libamdhip64.so.7"])
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: opened.append(name) or name)

        assert libs.load_vendor_library("libamdhip64", "libamdhip64.so") == "/sdk/libamdhip64.so.7"
        assert opened == ["/sdk/libamdhip64.so.7"]

    def test_falls_back_in_order(self, monkeypatch):
        opened = []

        def cdll(name):
            opened.append(name)
            if name != "/opt/rocm/lib/libamdhip64.so":
                raise OSError(name)
            return name

        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: [])
        monkeypatch.setattr(libs.ctypes, "CDLL", cdll)

        result = libs.load_vendor_library("libamdhip64", "libamdhip64.so", None, "/opt/rocm/lib/libamdhip64.so")
        assert result == "/opt/rocm/lib/libamdhip64.so"
        assert opened == ["libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"]

    def test_returns_none_when_nothing_loads(self, monkeypatch):
        def cdll(name):
            raise OSError(name)

        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: [])
        monkeypatch.setattr(libs.ctypes, "CDLL", cdll)

        assert libs.load_vendor_library("libamdhip64", "libamdhip64.so") is None

    def test_warns_when_a_second_copy_gets_loaded(self, monkeypatch, caplog):
        calls = iter([[], ["/sdk/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so.7"]])
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: next(calls))
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: name)
        caplog.set_level(logging.WARNING, logger="iris.platform")

        libs.load_vendor_library("libamdhip64", "libamdhip64.so")

        assert any("Multiple copies of libamdhip64" in r.message for r in caplog.records)

    def test_info_once_when_torch_copy_overrides_ld_library_path(self, monkeypatch, caplog, tmp_path):
        sdk, system = tmp_path / "sdk", tmp_path / "system"
        sdk.mkdir()
        system.mkdir()
        (sdk / "libamdhip64.so.7").write_bytes(b"sdk hip")
        (system / "libamdhip64.so").write_bytes(b"system hip")
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{tmp_path / 'missing'}:{system}")
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: [str(sdk / "libamdhip64.so.7")])
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: name)
        caplog.set_level(logging.INFO, logger="iris.platform")

        libs.load_vendor_library("libamdhip64", "libamdhip64.so")
        libs.load_vendor_library("libamdhip64", "libamdhip64.so")

        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert infos == [
            f"Using libamdhip64 from {sdk / 'libamdhip64.so.7'}, not {system / 'libamdhip64.so'} from LD_LIBRARY_PATH"
        ]

    def test_debug_only_when_ld_library_path_has_the_same_copy(self, monkeypatch, caplog, tmp_path):
        lib = tmp_path / "libamdhip64.so.7"
        lib.write_bytes(b"hip")
        os.symlink(lib, tmp_path / "libamdhip64.so")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: [str(lib)])
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: name)
        caplog.set_level(logging.INFO, logger="iris.platform")

        libs.load_vendor_library("libamdhip64", "libamdhip64.so")

        assert not caplog.records

    def test_unmapped_library_comes_from_beside_directory(self, monkeypatch, tmp_path):
        sdk = tmp_path / "sdk"
        sdk.mkdir()
        (sdk / "libamd_smi.so.27").write_bytes(b"smi")
        mapped = {"libamdhip64": [str(sdk / "libamdhip64.so.7")], "libamd_smi": []}
        opened = []
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: mapped[stem])
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: opened.append(name) or name)

        libs.load_vendor_library("libamd_smi", "libamd_smi.so", beside="libamdhip64")

        assert opened == [str(sdk / "libamd_smi.so.27")]

    def test_beside_falls_back_when_directory_has_no_copy(self, monkeypatch, tmp_path):
        mapped = {"libamdhip64": [str(tmp_path / "libamdhip64.so.7")], "libamd_smi": []}
        opened = []
        monkeypatch.setattr(libs, "_mapped_library_paths", lambda stem: mapped[stem])
        monkeypatch.setattr(libs.ctypes, "CDLL", lambda name: opened.append(name) or name)

        libs.load_vendor_library("libamd_smi", "libamd_smi.so", beside="libamdhip64")

        assert opened == ["libamd_smi.so"]
