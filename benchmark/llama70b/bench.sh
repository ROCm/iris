#!/bin/bash
# bench.sh - vLLM Llama-3.3-70B-FP8 TP=8 all-reduce PERF run (one baked arm).
#
# The headline-number run: `vllm bench serve --ignore-eos` at the operating point,
# no profiler hooks (tracing tax is asymmetric, so published/compared numbers come
# from unprofiled runs). Siblings, all sourcing _serve.sh so they hit the IDENTICAL
# server: profile.sh (same workload + profiler, dumps traces), eval.sh (gsm8k
# correctness gate), test.sh (collective correctness).
#
# The arm is whatever the image baked (Dockerfile.baseline | .exp). The A/B is two
# runs - build+run each image, then compare their result JSONs:
#   docker build -f Dockerfile.baseline -t iris-repro:baseline . && $RUN iris-repro:baseline ./bench.sh
#   docker build -f Dockerfile.exp      -t iris-repro:exp      . && $RUN iris-repro:exp      ./bench.sh
#
# Usage:
#   ./bench.sh                              # decode64 (8192/1024/conc-64), random data, warm
#   WORKLOAD=confluence ./bench.sh          # the guide's 1024/1024/conc-4 example
#   WARMUP=0 ./bench.sh                     # cold config (no warmup exclusion)
#   DATA=real ./bench.sh                    # ShareGPT requests vs synthetic
#   INPUT_LEN=2048 CONCURRENCY=8 ./bench.sh # custom shape

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
COMMAND=bench                    # output lands under output/<arm>-bench/
source "$SCRIPT_DIR/_serve.sh"   # server + operating point + run_workload + output dirs

dump_arm
prefetch_model
ensure_sharegpt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo ""
echo "=========================================="
echo "[bench] arm=$ARM data=$DATA input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$CONCURRENCY warmup=$WARMUP (unprofiled)"
echo "=========================================="

start_server
wait_ready
run_workload "$ARM"   # PROFILE unset -> no --profile, no traces; result -> results/
stop_server

# A wrapper must report the real outcome: a missing result JSON means the workload
# didn't finish (the engine died mid-run). Fail loudly instead of a silent exit 0.
[ -f "$RESULTS_DIR/vllm_${ARM}.json" ] || { echo "[bench] FATAL: no result JSON ($RESULTS_DIR/vllm_${ARM}.json) - workload did not complete (check the log for EngineDeadError)." >&2; exit 1; }

echo "[bench] done. result: $RESULTS_DIR/vllm_${ARM}.json"
