#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

set -e  # Exit on any error

# Get num_ranks from command line argument
NUM_RANKS=$1

if [ -z "$NUM_RANKS" ]; then
    echo "Error: NUM_RANKS not provided"
    echo "Usage: $0 <num_ranks>"
    exit 1
fi

# Run examples tests one at a time using distributed wrapper
echo 'Running examples tests one at a time...'
for test_file in tests/examples/test_*.py; do
  echo "Testing: $test_file with $NUM_RANKS ranks"
  python tests/run_tests_distributed.py --num_ranks $NUM_RANKS "$test_file" -v --tb=short --durations=10
done

# Run unit tests one at a time using distributed wrapper
echo 'Running unit tests one at a time...'
for test_file in tests/unittests/test_*.py; do
  echo "Testing: $test_file with $NUM_RANKS ranks"
  python tests/run_tests_distributed.py --num_ranks $NUM_RANKS "$test_file" -v --tb=short --durations=10
done
