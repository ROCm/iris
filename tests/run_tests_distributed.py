#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Launcher/worker script for running pytest tests under torchrun.

Direct usage:
    python tests/run_tests_distributed.py tests/unittests/ --num_ranks 4 -v

Worker usage (invoked automatically by torchrun):
    torchrun --nproc_per_node=4 tests/run_tests_distributed.py tests/unittests/ -v
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _running_under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _parse_launcher_args(argv: list[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(description="Run pytest tests under torchrun.")
    parser.add_argument("--num_ranks", type=int, required=True, help="Number of torchrun processes to launch.")
    args, pytest_args = parser.parse_known_args(argv)

    if args.num_ranks < 1:
        parser.error("--num_ranks must be at least 1")
    if not pytest_args:
        parser.error("At least one pytest path or argument is required")

    return args.num_ranks, pytest_args


def _launch_torchrun(argv: list[str]) -> int:
    num_ranks, pytest_args = _parse_launcher_args(argv)
    script_path = str(Path(__file__).resolve())
    launch_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        "--nnodes=1",
        f"--nproc_per_node={num_ranks}",
        script_path,
        *pytest_args,
    ]
    return subprocess.run(launch_cmd, check=False).returncode


def _run_pytest_worker(pytest_args: list[str]) -> int:
    os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")

    import pytest
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
    )

    try:
        return pytest.main(pytest_args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    if _running_under_torchrun():
        return _run_pytest_worker(sys.argv[1:])
    return _launch_torchrun(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
