#!/bin/bash
# Plot normalized results for each batch size separately

set -e  # Exit on error

# Configuration
INPUT_FILE="model_sweep_results_matmul_all_reduce-w-gpu-sdma.json"
PLOT_SCRIPT="matmul_all_gather/plot_normalized_sweep_results_poster.py"
EXCLUDE_PATTERN="matmul|gemm|one"
SORT_BY="copy-engine-benefit"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="paper_plots_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "Output directory: ${OUTPUT_DIR}"
echo "================================"

# Extract unique M values from JSON
echo "Extracting batch sizes from ${INPUT_FILE}..."
M_VALUES=$(python3 -c "
import json
import sys

with open('${INPUT_FILE}', 'r') as f:
    data = json.load(f)

m_values = sorted(set(entry.get('M') for entry in data if 'M' in entry))
print(' '.join(map(str, m_values)))
")

echo "Found batch sizes: ${M_VALUES}"
echo ""

# Iterate through each M value
for M in ${M_VALUES}; do
    echo "Processing M=${M}..."

    OUTPUT_PNG="${OUTPUT_DIR}/matmul_all_reduce_M${M}_normalized.png"
    LOG_FILE="${OUTPUT_DIR}/matmul_all_reduce_M${M}.log"

    # Run plotting script and capture output
    python3 "${PLOT_SCRIPT}" \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_PNG}" \
        --exclude "${EXCLUDE_PATTERN}" \
        --m-filter "${M}" \
        --sort "${SORT_BY}" \
        2>&1 | tee "${LOG_FILE}"

    echo "  ✓ Saved plot: ${OUTPUT_PNG}"
    echo "  ✓ Saved log:  ${LOG_FILE}"
    echo ""
done

echo "================================"
echo "All plots saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"
