# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest configuration for iris distributed tests.
"""

import os
import socket
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def pytest_addoption(parser):
    parser.addoption("--num_ranks", action="store", default="1", type=int, help="Number of ranks to use in tests")


@pytest.fixture
def num_ranks(request):
    return request.config.getoption("--num_ranks")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def global_dist_setup():
    """Global distributed setup for the entire test session."""
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(_find_free_port()))
    init_method = f"tcp://{master_addr}:{master_port}"

    # Store the init method for use by tests
    os.environ["GLOBAL_INIT_METHOD"] = init_method

    yield init_method

    # Cleanup happens automatically when session ends


def pytest_sessionfinish(session, exitstatus):
    """Clean up distributed process groups at the end of the test session."""
    if dist.is_initialized():
        dist.destroy_process_group()
