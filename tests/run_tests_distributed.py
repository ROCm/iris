#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Simple wrapper to run pytest tests using torchrun, which manages distributed process groups and avoids port conflicts through automatic port allocation.
"""
import os
import sys

# Set required environment variable for RCCL on ROCm
os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")


def _distributed_worker_main():
    """Main function for distributed worker that runs pytest."""
    import torch
    import torch.distributed as dist

    # torchrun sets these environment variables automatically
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the correct GPU for this specific process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize distributed - torchrun already set up the environment
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,
    )

    try:
        # Import and run pytest directly
        import pytest

        # Get pytest args from environment (set by launcher)
        pytest_args_str = os.environ.get("PYTEST_ARGS", "")
        pytest_args = pytest_args_str.split() if pytest_args_str else []

        # Run pytest
        exit_code = pytest.main(pytest_args)
        sys.exit(exit_code)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests_distributed.py [--num_ranks N] [pytest_args...] <test_file>")
        sys.exit(1)

    # Check if we're being called as a torchrun worker
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # We're running inside torchrun - execute as worker
        _distributed_worker_main()
        return

    # We're the launcher - parse args and start torchrun
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

    print(f"Running {test_file} with {num_ranks} ranks using torchrun")

    # Build pytest arguments string
    pytest_cmd_args = [test_file] + pytest_args
    pytest_args_str = " ".join(pytest_cmd_args)

    # Set environment variable for worker to read
    os.environ["PYTEST_ARGS"] = pytest_args_str

    # Build torchrun command - it will re-invoke this script as a worker
    import subprocess

    torchrun_cmd = [
        "torchrun",
        f"--nproc_per_node={num_ranks}",
        "--standalone",  # Single-node training
        __file__,  # Re-invoke this script
        "--worker-mode",  # Dummy arg to distinguish from launcher
    ]

    print(f"Executing: {' '.join(torchrun_cmd)}")

    # Run torchrun and return its exit code
    try:
        result = subprocess.run(torchrun_cmd, check=False, env=os.environ.copy())
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running torchrun: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
