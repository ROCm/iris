# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from unittest import mock

from iris.host.platform import hip


def test_get_rocm_version_runs_hipconfig_once(tmp_path):
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("7.1.0-12345\n")
    completed = mock.Mock(stdout=f"{tmp_path}\n")

    hip.get_rocm_version.cache_clear()
    try:
        with mock.patch.object(hip, "_is_amd_backend", True), mock.patch.object(
            hip.subprocess, "run", return_value=completed
        ) as run:
            results = {hip.get_rocm_version() for _ in range(5)}
            assert results == {(7, 1)}
            assert run.call_count == 1
    finally:
        hip.get_rocm_version.cache_clear()
