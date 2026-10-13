#!/bin/bash
# bench.sh - vLLM Llama 3.3 70B FP8 TP=8 PERF run (one baked arm), unprofiled.
#
# The headline-number run: `vllm bench serve --ignore-eos` at the operating point, no
# profiler hooks (the tracing tax is asymmetric, so compared numbers come from unprofiled
# runs). Siblings source _serve.sh so they hit the IDENTICAL server: profile.sh (same
# workload + profiler), eval.sh (gsm8k correctness gate). The arm is whatever the image
# baked; bench.sh sets up nothing.
#
# Usage:
#   ./bench.sh                              # decode64, random data
#   WORKLOAD=confluence ./bench.sh          # 1024-in/4-conc kernel shape
#   DATA=real ./bench.sh                    # ShareGPT requests vs synthetic
#   INPUT_LEN=2048 CONCURRENCY=8 ./bench.sh # custom kernel shape

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source "$SCRIPT_DIR/_serve.sh"   # server + operating point + run_workload + output dirs

# Artifacts follow the per-arm layout from _serve.sh: vllm/ (the --save-result JSON).
# bench is unprofiled - no profile/.

print_env
prefetch_model
ensure_sharegpt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo ""
echo "=========================================="
echo "[bench] data=$DATA input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$CONCURRENCY warmup=$WARMUP (unprofiled)"
echo "=========================================="

start_server
wait_ready
run_workload   # PROFILE unset -> no --profile, no traces; result -> vllm/
stop_server

# RAW only: the arm produces vllm/ and stops. The runner validates completeness
# (run.sh calls `preprocess.py --validate`); analysis is a separate read-only pass.
echo "[bench] done. arm dir: $ARM_DIR (vllm/ - raw)"
