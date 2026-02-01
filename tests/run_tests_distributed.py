#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple wrapper to run pytest tests within a single distributed process group.
This avoids the overhead of creating/destroying process groups for each test case.
"""

import os
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import socket

# Set required environment variable for RCCL on ROCm
os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _distributed_worker(rank, world_size, test_file, pytest_args, init_method):
    """Worker function that runs pytest within a distributed process group."""
    # Set the correct GPU for this specific process
    # When ROCR_VISIBLE_DEVICES is set, devices are remapped, so rank 0 should use device 0, etc.
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Initialize distributed once for all tests
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )

    try:
        # Import and run pytest directly
        import pytest
        import sys

        # Set up sys.argv for pytest
        original_argv = sys.argv[:]
        sys.argv = ["pytest", test_file] + pytest_args

        try:
            # Run pytest directly in this process
            exit_code = pytest.main([test_file] + pytest_args)
            # If tests failed, exit with the failure code
            if exit_code != 0:
                sys.exit(exit_code)
            return exit_code
        finally:
            # Restore original argv
            sys.argv = original_argv

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests_distributed.py [--num_ranks N] [pytest_args...] <test_file>")
        sys.exit(1)

    # Get number of ranks from args or default to 2
    num_ranks = 2
    args = sys.argv[1:]

    if "--num_ranks" in args:
        idx = args.index("--num_ranks")
        if idx + 1 < len(args):
            num_ranks = int(args[idx + 1])
            # Remove --num_ranks and its value from args
            args = args[:idx] + args[idx + 2 :]

    # The test file is the first argument after --num_ranks, everything else is pytest args
    if not args:
        print("Error: No test file specified")
        sys.exit(1)

    test_file = args[0]
    pytest_args = args[1:]  # Everything after the test file

    print(f"Running {test_file} with {num_ranks} ranks")
    print(f"args={args}, test_file={test_file}, pytest_args={pytest_args}")

    # Find a free port for this test run to avoid conflicts with parallel runs
    free_port = _find_free_port()
    init_method = f"tcp://127.0.0.1:{free_port}"
    print(f"Using init_method: {init_method}")

    # Run all tests within a single distributed process group
    try:
        mp.spawn(
            _distributed_worker,
            args=(num_ranks, test_file, pytest_args, init_method),
            nprocs=num_ranks,
            join=True,
        )
    except SystemExit as e:
        # Catch sys.exit() from worker and return same exit code
        sys.exit(e.code if isinstance(e.code, int) else 1)
    except Exception:
        # Any other unhandled exception = failure
        sys.exit(1)


if __name__ == "__main__":
    main()
