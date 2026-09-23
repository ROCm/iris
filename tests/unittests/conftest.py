# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Fixtures shared across unit test modules."""

import pytest
import torch.distributed as dist


@pytest.fixture(scope="session")
def rocshmem_runtime():
    """Initialise rocSHMEM once per process.

    The runtime is per-process, so every module that calls
    init_rocshmem_by_uniqueid itself would initialise it again. Session scope
    means all of them share one. No finalize in teardown: it would pull the
    runtime out from under anything else still running.

    Imported here rather than at module scope so tests are collected and
    individually skipped. A module-level importorskip collects zero items,
    which makes pytest exit 5 (NO_TESTS_COLLECTED) and fails the whole run.
    """
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    rshmem = pytest.importorskip("rocshmem4py", reason="rocSHMEM tests need rocshmem4py installed")
    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)
    return rshmem
