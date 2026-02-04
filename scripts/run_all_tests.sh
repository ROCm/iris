#!/bin/bash
# Run all tests across all test directories (ccl, examples, ops, unittests, x)
# Tests with all rank configurations (1, 2, 4, 8) - matches CI

set -e

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory with timestamp subdirectory
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Main log file that captures everything
MAIN_LOG="${LOG_DIR}/_all.log"

# Test configurations: test_dir,num_ranks
declare -a TEST_CONFIGS=(
    "ccl,1"
    "ccl,2"
    "ccl,4"
    "ccl,8"
    "examples,1"
    "examples,2"
    "examples,4"
    "examples,8"
    "ops,1"
    "ops,2"
    "ops,4"
    "ops,8"
    "unittests,1"
    "unittests,2"
    "unittests,4"
    "unittests,8"
    "x,1"
    "x,2"
    "x,4"
    "x,8"
)

{
echo "========================================"
echo "RUNNING ALL TESTS (CI-STYLE)"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Test directories: ccl, examples, ops, unittests, x"
echo "Rank configurations: 1, 2, 4, 8"
echo "Logs: $LOG_DIR/"
echo "  Main log: $MAIN_LOG"
echo "  Individual logs: ${LOG_DIR}/<test_dir>_ranks<N>.log"
echo "========================================"
echo ""
} | tee "$MAIN_LOG"

# Track results
TOTAL_CONFIGS=0
PASSED_CONFIGS=0
FAILED_CONFIGS=0
declare -a FAILED_LIST

# Run each test configuration
for config in "${TEST_CONFIGS[@]}"; do
    IFS=',' read -r test_dir num_ranks <<< "$config"
    
    TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
    
    {
    echo ""
    echo "========================================"
    echo "[$TOTAL_CONFIGS/${#TEST_CONFIGS[@]}] Testing: $test_dir with $num_ranks ranks"
    echo "========================================"
    
    # Check if test directory exists and has test files
    if [ ! -d "tests/$test_dir" ]; then
        echo "⚠️  Directory tests/$test_dir does not exist, skipping..."
        continue
    fi
    
    test_file_count=$(ls tests/$test_dir/test_*.py 2>/dev/null | wc -l)
    if [ "$test_file_count" -eq 0 ]; then
        echo "⚠️  No test files in tests/$test_dir, skipping..."
        continue
    fi
    
    echo "Found $test_file_count test file(s) in tests/$test_dir"
    
    # Run all tests in this directory with this rank configuration
    log_file="${LOG_DIR}/${test_dir}_ranks${num_ranks}.log"
    
    if python tests/run_tests_distributed.py --num_ranks "$num_ranks" "tests/$test_dir" -v --tb=short 2>&1 | tee "$log_file"; then
        PASSED_CONFIGS=$((PASSED_CONFIGS + 1))
        echo "  ✅ PASSED: $test_dir (${num_ranks} ranks)"
    else
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
        FAILED_LIST+=("$test_dir (${num_ranks} ranks)")
        echo "  ❌ FAILED: $test_dir (${num_ranks} ranks) - see $log_file"
    fi
    } | tee -a "$MAIN_LOG"
done

{
echo ""
echo "========================================"
echo "TEST RUN COMPLETE"
echo "========================================"
echo "Total configurations: $TOTAL_CONFIGS"
echo "Passed: $PASSED_CONFIGS"
echo "Failed: $FAILED_CONFIGS"
echo ""

if [ $FAILED_CONFIGS -gt 0 ]; then
    echo "Failed configurations:"
    for failed in "${FAILED_LIST[@]}"; do
        echo "  - $failed"
    done
    echo ""
else
    echo "✅ All tests passed!"
    echo ""
fi
} | tee -a "$MAIN_LOG"

if [ $FAILED_CONFIGS -gt 0 ]; then
    exit 1
else
    exit 0
fi
