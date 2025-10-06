# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for backend detection and runtime module selection.

These tests verify that the backend detection logic works correctly
and that the appropriate backend module is selected based on configuration.

Backend selection priority:
1. Build-time configuration (--config-settings backend=nvidia)
2. IRIS_BACKEND environment variable
3. Auto-detection based on available libraries
4. Default to HIP
"""

import os
import sys
import pytest
import importlib
import importlib.util


def test_backend_detection_default():
    """Test that default backend is HIP when no environment variable is set."""
    # Clear any existing IRIS_BACKEND setting
    old_env = os.environ.pop("IRIS_BACKEND", None)

    try:
        # Load hip.py directly to test detection logic
        spec = importlib.util.spec_from_file_location(
            "hip_test",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
        )
        hip_module = importlib.util.module_from_spec(spec)

        # Execute the module - detection happens before trying to load backend
        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found, but detection logic runs first
            pass

        # Check that backend was set to 'hip' (default)
        assert hasattr(hip_module, '_backend')
        assert hip_module._backend == 'hip', "Default backend should be 'hip'"

    finally:
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env


def test_backend_detection_cuda_env():
    """Test that CUDA backend is selected when IRIS_BACKEND=cuda."""
    old_env = os.environ.get("IRIS_BACKEND")

    try:
        os.environ["IRIS_BACKEND"] = "cuda"

        # Load hip.py directly
        spec = importlib.util.spec_from_file_location(
            "hip_test_cuda",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that backend was set to 'cuda'
        assert hasattr(hip_module, '_backend')
        assert hip_module._backend == 'cuda', "Backend should be 'cuda' when IRIS_BACKEND=cuda"

    finally:
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_backend_detection_nvidia_alias():
    """Test that CUDA backend is selected when IRIS_BACKEND=nvidia."""
    old_env = os.environ.get("IRIS_BACKEND")

    try:
        os.environ["IRIS_BACKEND"] = "nvidia"

        # Load hip.py directly
        spec = importlib.util.spec_from_file_location(
            "hip_test_nvidia",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that backend was set to 'cuda'
        assert hasattr(hip_module, '_backend')
        assert hip_module._backend == 'cuda', "Backend should be 'cuda' when IRIS_BACKEND=nvidia"

    finally:
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_backend_detection_hip_env():
    """Test that HIP backend is selected when IRIS_BACKEND=hip."""
    old_env = os.environ.get("IRIS_BACKEND")

    try:
        os.environ["IRIS_BACKEND"] = "hip"

        # Load hip.py directly
        spec = importlib.util.spec_from_file_location(
            "hip_test_hip",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that backend was set to 'hip'
        assert hasattr(hip_module, '_backend')
        assert hip_module._backend == 'hip', "Backend should be 'hip' when IRIS_BACKEND=hip"

    finally:
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_backend_detection_amd_alias():
    """Test that HIP backend is selected when IRIS_BACKEND=amd."""
    old_env = os.environ.get("IRIS_BACKEND")

    try:
        os.environ["IRIS_BACKEND"] = "amd"

        # Load hip.py directly
        spec = importlib.util.spec_from_file_location(
            "hip_test_amd",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that backend was set to 'hip'
        assert hasattr(hip_module, '_backend')
        assert hip_module._backend == 'hip', "Backend should be 'hip' when IRIS_BACKEND=amd"

    finally:
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_ipc_handle_size_definition():
    """Test that IPC handle size functions are defined correctly in source files."""
    import re

    # Check _hip.py defines get_ipc_handle_size returning 64
    hip_file = os.path.join(os.path.dirname(__file__), "../../iris/_hip.py")
    with open(hip_file, 'r') as f:
        hip_content = f.read()

    assert 'def get_ipc_handle_size()' in hip_content, "_hip.py should define get_ipc_handle_size"
    assert 'return 64' in hip_content, "_hip.py should return 64 for IPC handle size"

    # Check cuda.py defines get_ipc_handle_size returning 128
    cuda_file = os.path.join(os.path.dirname(__file__), "../../iris/cuda.py")
    with open(cuda_file, 'r') as f:
        cuda_content = f.read()

    assert 'def get_ipc_handle_size()' in cuda_content, "cuda.py should define get_ipc_handle_size"
    assert 'return 128' in cuda_content, "cuda.py should return 128 for IPC handle size"


def test_hip_module_structure():
    """Test that hip.py has the expected structure for backend redirection."""
    hip_file = os.path.join(os.path.dirname(__file__), "../../iris/hip.py")
    with open(hip_file, 'r') as f:
        hip_content = f.read()

    # Check for backend detection function
    assert '_detect_backend' in hip_content, "hip.py should have _detect_backend function"

    # Check for get_backend function
    assert 'def get_backend():' in hip_content, "hip.py should have get_backend function"

    # Check for environment variable handling
    assert 'IRIS_BACKEND' in hip_content, "hip.py should check IRIS_BACKEND environment variable"

    # Check for backend aliases
    assert 'cuda' in hip_content and 'nvidia' in hip_content, "hip.py should support cuda/nvidia aliases"
    assert 'hip' in hip_content and 'amd' in hip_content, "hip.py should support hip/amd aliases"


def test_iris_py_uses_get_ipc_handle_size():
    """Test that iris.py uses get_ipc_handle_size from hip module."""
    iris_file = os.path.join(os.path.dirname(__file__), "../../iris/iris.py")
    with open(iris_file, 'r') as f:
        iris_content = f.read()

    # Check that get_ipc_handle_size is imported
    assert 'get_ipc_handle_size' in iris_content, "iris.py should import get_ipc_handle_size"

    # Check that it's used instead of hardcoded 64
    assert 'ipc_handle_size = get_ipc_handle_size()' in iris_content, "iris.py should call get_ipc_handle_size()"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])



def test_build_time_config_cuda():
    """Test that build-time configuration for CUDA is respected."""
    import tempfile
    import shutil

    # Create a temporary config
    config_dir = os.path.join(os.path.dirname(__file__), "../../iris/.config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "backend.txt")

    old_env = os.environ.get("IRIS_BACKEND")

    try:
        # Write CUDA config
        with open(config_file, "w") as f:
            f.write("cuda")

        # Clear environment variable to test config priority
        os.environ.pop("IRIS_BACKEND", None)

        # Load hip.py
        spec = importlib.util.spec_from_file_location(
            "hip_test_buildtime_cuda",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py"),
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that backend was set to 'cuda' from config file
        assert hasattr(hip_module, "_backend")
        assert hip_module._backend == "cuda", "Build-time config should set backend to 'cuda'"

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(config_dir) and not os.listdir(config_dir):
            os.rmdir(config_dir)
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_build_time_config_priority():
    """Test that build-time configuration takes priority over environment variable."""
    config_dir = os.path.join(os.path.dirname(__file__), "../../iris/.config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "backend.txt")

    old_env = os.environ.get("IRIS_BACKEND")

    try:
        # Write HIP config
        with open(config_file, "w") as f:
            f.write("hip")

        # Set environment to CUDA (should be overridden by config)
        os.environ["IRIS_BACKEND"] = "cuda"

        # Load hip.py
        spec = importlib.util.spec_from_file_location(
            "hip_test_priority",
            os.path.join(os.path.dirname(__file__), "../../iris/hip.py"),
        )
        hip_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(hip_module)
        except OSError:
            # Expected - GPU library not found
            pass

        # Check that config takes priority
        assert hasattr(hip_module, "_backend")
        assert hip_module._backend == "hip", "Build-time config should take priority over env var"

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(config_dir) and not os.listdir(config_dir):
            os.rmdir(config_dir)
        # Restore environment
        if old_env is not None:
            os.environ["IRIS_BACKEND"] = old_env
        else:
            os.environ.pop("IRIS_BACKEND", None)


def test_build_backend_module():
    """Test that the build_backend module can write configuration correctly."""

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    try:
        from build_backend import _write_backend_config

        config_dir = os.path.join(os.path.dirname(__file__), "../../iris/.config")
        config_file = os.path.join(config_dir, "backend.txt")

        # Test nvidia alias
        _write_backend_config({"backend": "nvidia"})
        assert os.path.exists(config_file), "Config file should be created"
        with open(config_file, "r") as f:
            assert f.read() == "cuda", "nvidia should map to cuda"

        # Test hip
        _write_backend_config({"backend": "hip"})
        with open(config_file, "r") as f:
            assert f.read() == "hip", "hip should stay as hip"

        # Test amd alias
        _write_backend_config({"backend": "amd"})
        with open(config_file, "r") as f:
            assert f.read() == "hip", "amd should map to hip"

        # Test no config (should remove file)
        _write_backend_config({})
        assert not os.path.exists(config_file), "Config file should be removed for auto-detect"

    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        if os.path.exists(config_dir) and not os.listdir(config_dir):
            os.rmdir(config_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
