# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import iris
import iris.device.sdma_utils as sdma_utils


@pytest.fixture
def no_xio(monkeypatch):
    """Make rocm-xio look uninstalled, as with a base `pip install iris`."""
    monkeypatch.setitem(sys.modules, "xio", None)
    monkeypatch.setattr(sdma_utils, "sdma_ep", None)


def test_import_iris_does_not_import_xio():
    code = "import sys, iris; assert sys.modules.get('xio') is None, 'import iris loaded rocm-xio'"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], env=env)


def test_sdma_without_xio_fails_before_init(no_xio):
    with pytest.raises(ImportError, match=r"iris\[sdma\]"):
        iris.iris(1 << 20, copy_engine="sdma")


@triton.jit
def _store_to_peer(buffer, n_elements, cur_rank, peer_rank, heap_bases, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.full([BLOCK_SIZE], 1, tl.float32) * (cur_rank + 1)
    iris.store(buffer + offsets, values, cur_rank, peer_rank, heap_bases, mask=mask)


def test_default_shader_store_without_xio(no_xio):
    shmem = iris.iris(1 << 20)
    assert shmem.get_copy_engine_ctx() is None

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    peer = (rank + 1) % world_size
    n_elements = 4096
    buffer = shmem.zeros(n_elements, dtype=torch.float32)
    shmem.barrier()

    _store_to_peer[(triton.cdiv(n_elements, 1024),)](
        buffer, n_elements, rank, peer, shmem.get_heap_bases(), BLOCK_SIZE=1024
    )
    shmem.barrier()

    writer = (rank - 1) % world_size
    torch.testing.assert_close(buffer, torch.full_like(buffer, writer + 1))

    with pytest.raises(RuntimeError, match='copy_engine="sdma"'):
        shmem.put(buffer, to_rank=peer)
    with pytest.raises(RuntimeError, match='copy_engine="sdma"'):
        shmem.quiet()
