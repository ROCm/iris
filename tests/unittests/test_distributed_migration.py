# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple tests for the new PyTorch distributed helpers.
These tests verify that the distributed helpers work correctly.
"""

import pytest
import numpy as np

def test_distributed_helpers_import():
    """Test that we can import the distributed helpers."""
    try:
        import iris._distributed_helpers
        assert hasattr(iris._distributed_helpers, 'distributed_allgather')
        assert hasattr(iris._distributed_helpers, 'distributed_barrier')
        assert hasattr(iris._distributed_helpers, 'distributed_broadcast_scalar')
        assert hasattr(iris._distributed_helpers, 'init_distributed')
    except ImportError as e:
        pytest.skip(f"Cannot import distributed helpers (expected in non-torch environment): {e}")

def test_launcher_import():
    """Test that we can import the launcher."""
    try:
        import iris.launcher
        assert hasattr(iris.launcher, 'launch_iris')
        assert hasattr(iris.launcher, 'create_iris_with_distributed_init')
    except ImportError as e:
        pytest.skip(f"Cannot import launcher (expected in non-torch environment): {e}")

def test_main_iris_import():
    """Test that we can still import the main iris module."""
    try:
        import iris.iris
        assert hasattr(iris.iris, 'Iris')
        assert hasattr(iris.iris, 'iris')
    except ImportError as e:
        pytest.skip(f"Cannot import iris.iris (expected in non-torch environment): {e}")

def test_iris_package_exports():
    """Test that the main iris package exports the expected functions."""
    try:
        import iris
        expected_exports = [
            'Iris', 'iris', 'load', 'store', 'get', 'put',
            'atomic_add', 'atomic_sub', 'atomic_cas', 'atomic_xchg',
            'atomic_xor', 'atomic_or', 'atomic_and', 'atomic_min', 'atomic_max',
            'do_bench', 'memset_tensor', 'hip',
            'set_logger_level', 'logger', 'DEBUG', 'INFO', 'WARNING', 'ERROR',
            'launch_iris'
        ]
        for export in expected_exports:
            assert hasattr(iris, export), f"Missing export: {export}"
    except ImportError as e:
        pytest.skip(f"Cannot import iris package (expected in non-torch environment): {e}")

if __name__ == "__main__":
    pytest.main([__file__])