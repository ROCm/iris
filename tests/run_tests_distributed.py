#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple wrapper to run pytest tests within a single distributed process group.
This avoids the overhead of creating/destroying process groups for each test case.
"""

import sys
import subprocess
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import os


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _distributed_worker(rank, world_size, test_file, pytest_args):
    """Worker function that runs pytest within a distributed process group."""
    # Initialize distributed once for all tests
    init_method = "tcp://127.0.0.1:12355"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
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

    # Run all tests within a single distributed process group
    mp.spawn(_distributed_worker, args=(num_ranks, test_file, pytest_args), nprocs=num_ranks, join=True)


if __name__ == "__main__":
    main()
