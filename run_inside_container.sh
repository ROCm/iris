#!/bin/bash
set -e

pip install -e . -q 2>/dev/null

echo "=== TRITON ALL_GATHER ==="
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 tests/run_tests_distributed.py tests/ccl/test_all_gather.py -v --tb=short

echo "=== GLUON ALL_GATHER ==="
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 tests/run_tests_distributed.py tests/ccl/test_all_gather_gluon.py -v --tb=short
