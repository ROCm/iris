# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest configuration for iris distributed tests.
"""

def pytest_addoption(parser):
    parser.addoption(
        "--num_ranks",
        action="store",
        default="1",
        type=int,
        help="Number of ranks to use in tests"
    )

import pytest

@pytest.fixture
def num_ranks(request):
    return request.config.getoption("--num_ranks")


# Shared fixtures for dtypes, semantics, and scopes
@pytest.fixture(params=[
    "torch.float16",
    "torch.float32", 
    "torch.int32",
    "torch.int64"
])
def dtype(request):
    """Parametrize data types for tests."""
    import torch
    return getattr(torch, request.param.split('.')[1])


@pytest.fixture(params=[
    "iris.Semantic.ACQUIRE",
    "iris.Semantic.RELEASE", 
    "iris.Semantic.ACQ_REL",
    "iris.Semantic.RELAXED"
])
def sem(request):
    """Parametrize memory semantics for atomic tests."""
    import iris
    semantic_name = request.param.split('.')[2]
    return getattr(iris.Semantic, semantic_name)


@pytest.fixture(params=[
    "iris.Scope.GPU",
    "iris.Scope.SYSTEM"
])
def scope(request):
    """Parametrize memory scope for atomic tests."""
    import iris
    scope_name = request.param.split('.')[2] 
    return getattr(iris.Scope, scope_name)