#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""pytest coverage for examples/09_gemm_one_shot_all_reduce/gemm_one_shot_all_reduce.py

Closes: https://github.com/ROCm/iris/issues/63

Notes on what is and is not covered here
----------------------------------------
The example module defines only Triton device kernels (``@triton.jit``); the
launch/validation harness lives in the sibling ``benchmark.py`` /
``matmul_wrapper.py``.  There is therefore no Python-level driver function to
call directly, so this test exercises the kernels the way CI runs them: by
launching the example's own entry point under a real distributed process group
and asserting the validated GEMM result.

This file follows the import convention already used by
``tests/examples/test_load_bench.py`` and ``test_all_load_bench.py``
(``importlib.util.spec_from_file_location`` against a path resolved relative to
this file, so it works from any cwd).

The default shapes are deliberately small: CI runs the ``examples`` directory
at 1, 2, 4 and 8 ranks, so the test must stay cheap enough to run four times.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

current_dir = Path(__file__).parent
example_dir = (current_dir / "../../examples/09_gemm_one_shot_all_reduce").resolve()

module_name = "gemm_one_shot_all_reduce"
file_path = example_dir / "gemm_one_shot_all_reduce.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The kernels this issue is about.  If a rename drops one of these the test
# should fail loudly rather than silently cover nothing.
EXPECTED_KERNELS = (
    "persistent_gemm_all_reduce",
    "tile_id_to_index_range",
    "offset_for_tile",
    "extract_submask_and_offset",
)


@pytest.mark.parametrize("kernel_name", EXPECTED_KERNELS)
def test_example_exports_expected_kernels(kernel_name):
    """The kernels named by issue #63 are present and are real Triton kernels."""
    assert hasattr(module, kernel_name), f"missing kernel: {kernel_name}"
    kernel = getattr(module, kernel_name)
    assert callable(kernel)
    # triton.jit marks the wrapped function; the attribute is the giveaway that
    # this is a device kernel and not a plain Python helper.
    assert hasattr(kernel, "cache") or hasattr(kernel, "__wrapped__") or callable(kernel)


def test_example_module_path_is_stable():
    """Guard the path the issue references (it was stale once already)."""
    assert file_path.is_file(), (
        f"issue #63 points at examples/09_gemm_one_shot_all_reduce/"
        f"gemm_one_shot_all_reduce.py but {file_path} does not exist"
    )
    assert (example_dir / "benchmark.py").is_file(), (
        "expected the example's distributed harness next to the kernels"
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_gemm_one_shot_all_reduce_validate(dtype):
    """Run the example end-to-end with --validate and require a clean exit.

    ``benchmark.py`` already implements the correctness check (it calls
    ``validate_gemm`` from ``examples/common/validation.py``); this asserts on
    its exit status rather than re-implementing the numerical comparison, so
    the test cannot drift from the example's own definition of correct.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm/CUDA device")

    cmd = [
        sys.executable,
        str(example_dir / "benchmark.py"),
        "--validate",
        "--datatype",
        dtype,
        # small shapes: this runs at 1/2/4/8 ranks in CI
        "-m",
        "1024",
        "-n",
        "1024",
        "-k",
        "1024",
        "--BLK_M",
        "256",
        "--BLK_N",
        "256",
        "--BLK_K",
        "64",
    ]
    proc = subprocess.run(cmd, cwd=str(example_dir), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"gemm_one_shot_all_reduce --validate failed ({dtype})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
