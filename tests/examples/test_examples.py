#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Auto-discover and run all example scripts via torchrun.

Each example is expected to have an example.py (or example_run*.py) with a
--validate flag. The test launches it with torchrun and asserts exit code 0.

Usage:
    torchrun --nproc_per_node=2 -m pytest tests/examples/test_examples.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "tests").is_dir() or not (PROJECT_ROOT / "examples").is_dir():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise FileNotFoundError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Discover all example directories with an example.py
EXAMPLE_SCRIPTS = []
for d in sorted(EXAMPLES_DIR.iterdir()):
    if not d.is_dir() or d.name in ("common", "benchmark"):
        continue
    example_py = d / "example.py"
    if example_py.exists():
        EXAMPLE_SCRIPTS.append(example_py)


@pytest.mark.parametrize(
    "script",
    EXAMPLE_SCRIPTS,
    ids=[s.parent.name for s in EXAMPLE_SCRIPTS],
)
def test_example(script):
    """Run an example script with --validate and check it exits cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--standalone",
            str(script),
            "--validate",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(
            f"{script.parent.name} failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
