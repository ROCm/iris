# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest configuration for iris distributed tests.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--num_ranks", action="store", default="1", type=int, help="Number of ranks to use in tests")


@pytest.fixture
def num_ranks(request):
    return request.config.getoption("--num_ranks")


