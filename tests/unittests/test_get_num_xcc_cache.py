# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from unittest import mock

from iris.host.platform import hip


def _mocked_runtime():
    runtime = mock.MagicMock()
    runtime.hipDeviceGetAttribute.return_value = 0
    return runtime


def test_get_num_xcc_runs_hip_device_get_attribute_once_per_device(tmp_path):
    hip._get_num_xcc_cached.cache_clear()
    runtime = _mocked_runtime()
    try:
        with mock.patch.object(hip, "_is_amd_backend", True), mock.patch.object(
            hip, "get_rocm_version", return_value=(7, 1)
        ), mock.patch.object(hip, "gpu_runtime", runtime), mock.patch.object(hip, "gpu_try"):
            results = {hip.get_num_xcc(device_id=0) for _ in range(5)}
            assert results == {0}
            assert runtime.hipDeviceGetAttribute.call_count == 1
    finally:
        hip._get_num_xcc_cached.cache_clear()


def test_get_num_xcc_caches_the_default_device_resolution(tmp_path):
    hip._get_num_xcc_cached.cache_clear()
    runtime = _mocked_runtime()
    try:
        with mock.patch.object(hip, "_is_amd_backend", True), mock.patch.object(
            hip, "get_rocm_version", return_value=(7, 1)
        ), mock.patch.object(hip, "get_device_id", return_value=0), mock.patch.object(
            hip, "gpu_runtime", runtime
        ), mock.patch.object(hip, "gpu_try"):
            hip.get_num_xcc()
            hip.get_num_xcc()
            assert runtime.hipDeviceGetAttribute.call_count == 1
    finally:
        hip._get_num_xcc_cached.cache_clear()


def test_get_num_xcc_keys_the_cache_on_device_id(tmp_path):
    hip._get_num_xcc_cached.cache_clear()
    runtime = _mocked_runtime()
    try:
        with mock.patch.object(hip, "_is_amd_backend", True), mock.patch.object(
            hip, "get_rocm_version", return_value=(7, 1)
        ), mock.patch.object(hip, "gpu_runtime", runtime), mock.patch.object(hip, "gpu_try"):
            hip.get_num_xcc(device_id=0)
            hip.get_num_xcc(device_id=1)
            assert runtime.hipDeviceGetAttribute.call_count == 2
    finally:
        hip._get_num_xcc_cached.cache_clear()
