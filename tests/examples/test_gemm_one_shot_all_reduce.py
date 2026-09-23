#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""pytest coverage for examples/09_gemm_one_shot_all_reduce/gemm_one_shot_all_reduce.py

Closes: https://github.com/ROCm/iris/issues/63

Notes on what is and is not covered here
----------------------------------------
The example module defines only Triton device kernels (``@triton.jit``); the
launch and validation harness lives in the sibling ``benchmark.py`` /
``matmul_wrapper.py``, so unlike the ``00_load`` / ``02_all_load`` examples
there is no Python-level driver function to import and call.  The test
therefore exercises the kernels the way a user does: by running the example's
own entry point and asserting on its exit status.

This file follows the import convention already used by
``tests/examples/test_load_bench.py`` and ``test_all_load_bench.py``
(``importlib.util.spec_from_file_location`` against a path resolved relative to
this file, so it works from any cwd).

``benchmark.py`` spawns its *own* ranks via ``mp.spawn`` (it is a standalone
program, not a function the CI process group drives), so ``--num_ranks 1``
keeps this test to what a single GPU can prove; the rank matrix is covered by
CI running the ``examples`` directory at 1/2/4/8 ranks.

Two of the example's defaults are known-not-portable and are set explicitly
below, so this test measures the kernels rather than re-discovering unrelated
bugs.  Both are filed separately rather than papered over:

1. ``gemm_sms`` auto-detection rejects itself on power-of-two CU counts.
   ``benchmark.py`` computes ``gemm_sms = 2 ** floor(log2(cu_count))`` and then
   exits if ``gemm_sms >= total_sms``.  For any GPU whose CU count is already
   a power of two that is *always* true, so the example dies with
   ``Invalid number of GEMM SMs`` before doing any work.
2. ``BLK_M``/``BLK_N`` default to 256x256, which needs 131072 bytes of shared
   memory — above the 65536-byte limit on some ROCm targets, producing
   ``triton.runtime.errors.OutOfResources``.  128x128 fits.
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
    import triton as _triton

    assert hasattr(module, kernel_name), f"missing kernel: {kernel_name}"
    kernel = getattr(module, kernel_name)
    assert isinstance(kernel, _triton.runtime.jit.JITFunction), (
        f"{kernel_name} should be a Triton JITFunction, got {type(kernel).__name__}"
    )


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
    ``validate_gemm`` from ``examples/common/validation.py`` and logs
    "Final C validation passed."); asserting on its exit status rather than
    re-implementing the numerical comparison keeps this test from drifting
    away from the example's own definition of correct.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm/CUDA device")

    # gemm_sms must be < total_sms; the example's own auto-detection can
    # violate that on power-of-two CU counts (see module docstring).
    total_sms = torch.cuda.get_device_properties(0).multi_processor_count
    gemm_sms = max(1, total_sms // 2)

    cmd = [
        sys.executable,
        "benchmark.py",
        "--validate",
        "--datatype",
        dtype,
        "--num_ranks",
        "1",
        # 128x128 blocks fit the 65536-byte shared-memory limit (see docstring)
        "-m",
        "512",
        "-n",
        "512",
        "-k",
        "512",
        "--BLK_M",
        "128",
        "--BLK_N",
        "128",
        "--BLK_K",
        "64",
        "--gemm_sms",
        str(gemm_sms),
        "--total_sms",
        str(total_sms),
    ]
    proc = subprocess.run(cmd, cwd=str(example_dir), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"gemm_one_shot_all_reduce --validate failed ({dtype})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
