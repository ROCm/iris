# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import iris


def test_import_iris_does_not_import_xio():
    code = "import sys, iris; assert sys.modules.get('xio') is None, 'import iris loaded rocm-xio'"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], env=env)


def test_default_context_rejects_host_sdma():
    shmem = iris.iris(1 << 20)
    assert shmem.get_copy_engine_ctx() is None

    rank = shmem.get_rank()
    peer = (rank + 1) % shmem.get_num_ranks()
    buffer = shmem.zeros(16, dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"engines=\(\"sdma\",\)"):
        shmem.put(buffer, to_rank=peer)
    with pytest.raises(RuntimeError, match=r"engines=\(\"sdma\",\)"):
        shmem.quiet()
