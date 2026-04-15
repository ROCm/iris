#!/bin/bash
# Run cross-warp coalescing probe under FFM simulation.
# Uses iris tracing infrastructure (IRIS_TRACE_ALLGATHER env var).
#
# Usage (inside FFM container on alola):
#   cd /workspace/iris
#   pip install -e .
#   bash examples/33_coalescing_probe/run_on_ffm.sh [--with-cap] [--export-trace]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IRIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Sourcing FFM environment ==="
source /ffm/ffmlite_env.sh

# IRIS_TRACE_ALLGATHER is set inside the probe script itself
echo "=== Running coalescing probe ==="
EXTRA_ARGS=""
if [[ "${1:-}" == "--export-trace" ]] || [[ "${2:-}" == "--export-trace" ]]; then
    EXTRA_ARGS="--export-trace"
fi

python3 "$SCRIPT_DIR/coalescing_probe.py" \
    --block_size_m 8 --block_size_n 256 --num_warps 4 --dtype fp32 \
    $EXTRA_ARGS

echo ""
echo "=== Running analysis ==="
python3 "$SCRIPT_DIR/analyze_coalescing.py" \
    --input coalescing_results_rank0.json \
    --output coalescing_plot.png

if [[ "${1:-}" == "--with-cap" ]]; then
    echo ""
    echo "=== Generating .cap trace via roccap ==="
    source /ffm/roccap_env.sh 2>/dev/null || true
    CAP_DIR="${CAP_DIR:-/tmp/coalescing-caps}"
    mkdir -p "$CAP_DIR"
    timeout 120 roccap capture --file "$CAP_DIR/coalescing_probe.cap" \
        python3 "$SCRIPT_DIR/coalescing_probe.py" \
            --block_size_m 8 --block_size_n 256 --num_warps 4 --dtype fp32 \
        || echo "WARNING: roccap exited (timeout on exit is expected with GPUVM vars)"
    echo "Cap files:"
    ls -la "$CAP_DIR"/*.cap 2>/dev/null || echo "  (none generated)"
fi

echo ""
echo "=== Done ==="
echo "Results: coalescing_results_rank0.json"
echo "Plot:    coalescing_plot.png"
