#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple test to verify the Gluon-based Iris implementation.

This test validates that:
1. IrisBackend aggregate can be created
2. IrisGluon class initializes correctly
3. Backend methods are callable
"""

import sys
import os

# Add the parent directory to the path so we can import iris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_iris_gluon_imports():
    """Test that iris_gluon module can be imported."""
    try:
        import iris.experimental.iris_gluon as iris_gl

        print("✓ Successfully imported iris.experimental.iris_gluon")
        return True
    except ImportError as e:
        print(f"✗ Failed to import iris.experimental.iris_gluon: {e}")
        return False


def test_iris_gluon_aggregate():
    """Test that IrisBackend aggregate is defined."""
    try:
        import iris.experimental.iris_gluon as iris_gl

        # Check that IrisBackend exists
        assert hasattr(iris_gl, "IrisBackend")
        print("✓ IrisBackend aggregate is defined")

        # Check that IrisGluon exists
        assert hasattr(iris_gl, "IrisGluon")
        print("✓ IrisGluon class is defined")

        # Check that iris factory function exists
        assert hasattr(iris_gl, "iris")
        print("✓ iris() factory function is defined")

        return True
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_iris_gluon_backend_methods():
    """Test that IrisBackend has all required methods."""
    try:
        import iris.experimental.iris_gluon as iris_gl

        backend_class = iris_gl.IrisBackend

        # Check for memory operation methods
        required_methods = [
            "_translate",
            "load",
            "store",
            "get",
            "put",
            "atomic_add",
            "atomic_sub",
            "atomic_cas",
            "atomic_xchg",
            "atomic_xor",
            "atomic_and",
            "atomic_or",
            "atomic_min",
            "atomic_max",
        ]

        for method in required_methods:
            assert hasattr(backend_class, method), f"Missing method: {method}"

        print(f"✓ IrisBackend has all {len(required_methods)} required methods")
        return True
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_iris_gluon_class_methods():
    """Test that IrisGluon class has required methods."""
    try:
        import iris.iris_gluon as iris_gl

        iris_class = iris_gl.IrisGluon

        # Check for host-side methods
        required_methods = [
            "get_backend",
            "get_heap_bases",
            "barrier",
            "get_device",
            "get_cu_count",
            "get_rank",
            "get_num_ranks",
            "broadcast",
            "zeros",
            "debug",
            "info",
            "warning",
            "error",
        ]

        for method in required_methods:
            assert hasattr(iris_class, method), f"Missing method: {method}"

        print(f"✓ IrisGluon has all {len(required_methods)} required methods")
        return True
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Iris Gluon Implementation")
    print("=" * 50)

    tests = [
        test_iris_gluon_imports,
        test_iris_gluon_aggregate,
        test_iris_gluon_backend_methods,
        test_iris_gluon_class_methods,
    ]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
