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
from iris.host.iris import _parse_copy_engine_env, _resolve_enable_copy_engine


@pytest.fixture
def no_xio(monkeypatch):
    """Make rocm-xio look uninstalled, as with a base `pip install iris`."""
    monkeypatch.setitem(sys.modules, "xio", None)
    monkeypatch.setattr(sdma_utils, "sdma_ep", None)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("0", False),
        ("False", False),
        ("no", False),
        ("off", False),
        ("", None),
        ("auto", None),
    ],
)
def test_parse_copy_engine_env(value, expected):
    assert _parse_copy_engine_env(value) is expected


@pytest.mark.parametrize("value", ["2", "enable", "ture", "disabled"])
def test_parse_copy_engine_env_rejects_unknown(value):
    with pytest.raises(ValueError, match="IRIS_ENABLE_COPY_ENGINE"):
        _parse_copy_engine_env(value)


def test_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("IRIS_ENABLE_COPY_ENGINE", "1")
    assert _resolve_enable_copy_engine(False) is False
    monkeypatch.setenv("IRIS_ENABLE_COPY_ENGINE", "0")
    assert _resolve_enable_copy_engine(True) is True


@pytest.mark.parametrize("value", [1, 0, "yes"])
def test_explicit_argument_must_be_bool(value):
    with pytest.raises(TypeError, match="enable_copy_engine"):
        _resolve_enable_copy_engine(value)


def test_auto_follows_xio_install(monkeypatch, no_xio):
    monkeypatch.delenv("IRIS_ENABLE_COPY_ENGINE", raising=False)
    assert _resolve_enable_copy_engine(None) is False
    monkeypatch.setenv("IRIS_ENABLE_COPY_ENGINE", "auto")
    assert _resolve_enable_copy_engine(None) is False


def test_import_iris_does_not_import_xio():
    code = "import sys, iris; assert 'xio' not in sys.modules, 'import iris loaded rocm-xio'"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], env=env)


def test_invalid_env_fails_before_init(monkeypatch):
    monkeypatch.setenv("IRIS_ENABLE_COPY_ENGINE", "maybe")
    with pytest.raises(ValueError, match="IRIS_ENABLE_COPY_ENGINE"):
        iris.iris(1 << 20)


def test_enable_without_xio_fails_before_init(no_xio):
    with pytest.raises(ImportError, match=r"iris\[sdma\]"):
        iris.iris(1 << 20, enable_copy_engine=True)


@triton.jit
def _store_to_peer(buffer, n_elements, cur_rank, peer_rank, heap_bases, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.full([BLOCK_SIZE], 1, tl.float32) * (cur_rank + 1)
    iris.store(buffer + offsets, values, cur_rank, peer_rank, heap_bases, mask=mask)


def _check_shader_store_without_sdma(shmem):
    assert shmem.enable_copy_engine is False
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

    with pytest.raises(RuntimeError, match="copy engine is disabled"):
        shmem.put(buffer, to_rank=peer)
    with pytest.raises(RuntimeError, match="copy engine is disabled"):
        shmem.quiet()


def test_disabled_shader_store(monkeypatch):
    monkeypatch.delenv("IRIS_ENABLE_COPY_ENGINE", raising=False)
    _check_shader_store_without_sdma(iris.iris(1 << 20, enable_copy_engine=False))


def test_env_disabled_shader_store(monkeypatch):
    monkeypatch.setenv("IRIS_ENABLE_COPY_ENGINE", "0")
    _check_shader_store_without_sdma(iris.iris(1 << 20))


def test_auto_without_xio_shader_store(monkeypatch, no_xio):
    monkeypatch.delenv("IRIS_ENABLE_COPY_ENGINE", raising=False)
    _check_shader_store_without_sdma(iris.iris(1 << 20))
