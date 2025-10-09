#!/bin/bash
set -e

# Arguments
EXAMPLE_PATH=$1
TFLOPS_THRESHOLD=$2
shift 2
BENCHMARK_ARGS="$@"

# Create overlay image in workspace (will be auto-cleaned by GitHub Actions)
OVERLAY="iris_overlay_perf_${EXAMPLE_PATH//\//_}.img"

echo "::group::Creating overlay image"
apptainer overlay create --size 1024 --create-dir /var/cache/iris "${OVERLAY}"
echo "::endgroup::"

echo "::group::Running performance benchmark"
apptainer exec --overlay "${OVERLAY}" --no-home --cleanenv --env HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  --bind "${PWD}:/iris_workspace" --cwd /iris_workspace \
  ~/apptainer/iris-dev.sif bash -c "
    set -e
    pip install -e .
    python examples/${EXAMPLE_PATH}/benchmark.py \
      --benchmark \
      --validate \
      -r 8 \
      ${BENCHMARK_ARGS} \
      --output_file perf_result.json
  "
echo "::endgroup::"

# Parse JSON and check performance
echo "::group::Validating performance"

# Check if benchmark succeeded
SUCCESS=$(jq -r '.success' perf_result.json)
if [ "$SUCCESS" != "true" ]; then
  echo "::error::Benchmark failed (success: $SUCCESS)"
  jq '.' perf_result.json
  exit 1
fi

TFLOPS=$(jq -r '.tflops' perf_result.json)

if [ -z "$TFLOPS" ] || [ "$TFLOPS" = "null" ]; then
  echo "::error::Failed to extract tflops from benchmark output"
  jq '.' perf_result.json
  exit 1
fi

echo "::notice::Achieved TFLOPs: $TFLOPS"

# Convert to integer for comparison
TFLOPS_INT=${TFLOPS%.*}
if (( TFLOPS_INT < TFLOPS_THRESHOLD )); then
  echo "::error::Performance regression detected! TFLOPs ($TFLOPS) is below threshold ($TFLOPS_THRESHOLD)"
  jq '.' perf_result.json
  exit 1
fi

echo "✅ Performance test passed! TFLOPs: $TFLOPS (threshold: >$TFLOPS_THRESHOLD)"
echo "::endgroup::"

