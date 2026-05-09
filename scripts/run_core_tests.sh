#!/bin/bash
# Run core tests (examples + unittests) with multiple rank configurations
# This is a faster subset for quick validation during development

set -e

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python interpreter not found" >&2
    exit 1
fi

count_visible_gpus() {
    local visible_devices="$1"

    if [ -z "$visible_devices" ]; then
        echo 0
        return
    fi

    IFS=',' read -r -a devices <<< "$visible_devices"
    echo "${#devices[@]}"
}

MAX_NUM_RANKS=${IRIS_MAX_NUM_RANKS:-0}
if [ "$MAX_NUM_RANKS" -eq 0 ]; then
    MAX_NUM_RANKS=$(count_visible_gpus "${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}")
fi
if [ "$MAX_NUM_RANKS" -eq 0 ]; then
    MAX_NUM_RANKS=$("$PYTHON_BIN" - <<'PY'
try:
    import torch

    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
fi

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory with timestamp subdirectory
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Main log file that captures everything
MAIN_LOG="${LOG_DIR}/_all.log"

# Test directories and their configurations (test_dir, num_ranks)
declare -a TEST_CONFIGS=(
    "unittests,1"
    "unittests,2"
    "unittests,4"
    "unittests,8"
    "examples,1"
    "examples,2"
)

{
echo "========================================"
echo "RUNNING CORE TESTS"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Test directories: examples, unittests"
echo "Rank configurations: 1, 2, 4, 8"
if [ "$MAX_NUM_RANKS" -gt 0 ]; then
    echo "Visible GPU limit: $MAX_NUM_RANKS"
fi
echo "Logs: $LOG_DIR/"
echo "  Main log: $MAIN_LOG"
echo "  Individual logs: ${LOG_DIR}/<test_dir>_<test_name>_rank*.log"
echo "========================================"
echo ""
} | tee "$MAIN_LOG"

# Run each test configuration
for config in "${TEST_CONFIGS[@]}"; do
    IFS=',' read -r test_dir num_ranks <<< "$config"

    if [ "$MAX_NUM_RANKS" -gt 0 ] && [ "$num_ranks" -gt "$MAX_NUM_RANKS" ]; then
        {
        echo ""
        echo "========================================"
        echo "Skipping tests: $test_dir with $num_ranks ranks"
        echo "Reason: visible GPU limit is $MAX_NUM_RANKS"
        echo "========================================"
        } | tee -a "$MAIN_LOG"
        continue
    fi
    
    {
    echo ""
    echo "========================================"
    echo "Running tests: $test_dir with $num_ranks ranks"
    echo "========================================"
    
    # Find all test files in the directory
    for test_file in tests/$test_dir/test_*.py; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file" .py)
            log_prefix="${LOG_DIR}/${test_dir}_${test_name}"
            
            echo "Testing: $test_file"
            echo "  Ranks: $num_ranks"
            echo "  Logs: ${log_prefix}_rank*.log"
            
            if [ "$num_ranks" -eq 1 ]; then
                # Single rank - direct log
                "$PYTHON_BIN" tests/run_tests_distributed.py --num_ranks $num_ranks "$test_file" -v --tb=short 2>&1 | tee "${log_prefix}_rank0.log"
            else
                # Multi-rank - combined log
                "$PYTHON_BIN" tests/run_tests_distributed.py --num_ranks $num_ranks "$test_file" -v --tb=short 2>&1 | tee "${log_prefix}_all_ranks.log"
            fi
            
            # Check exit code
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "  ✓ PASSED"
            else
                echo "  ✗ FAILED (see logs)"
            fi
        fi
    done
    } | tee -a "$MAIN_LOG"
done

{
echo ""
echo "========================================"
echo "Test run complete!"
echo "Logs saved to $LOG_DIR/"
echo "  Main log: $MAIN_LOG"
echo "========================================"
} | tee -a "$MAIN_LOG"

ls -lh "$LOG_DIR"/*.log
