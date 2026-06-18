#!/bin/bash
# vLLM Llama-3.3-70B-FP8 TP=8 all-reduce benchmark. Runs the single arm baked into
# the image (baseline | exp) and prints the serving metrics. The A/B is two runs:
#   docker build -f Dockerfile.baseline -t repro:baseline . && docker run ... repro:baseline ./bench.sh
#   docker build -f Dockerfile.exp      -t repro:exp      . && docker run ... repro:exp      ./bench.sh
# then compare the two outputs.
#
# Options (both off by default):
#   EVAL=1     also run a gsm8k accuracy pass against the live server (correctness gate)
#   PROFILE=1  also capture torch profiler traces
set -euo pipefail

MODEL="amd/Llama-3.3-70B-Instruct-FP8-KV"
TP=8
HOST=localhost
PORT=8000
EVAL="${EVAL:-0}"
PROFILE="${PROFILE:-0}"
OUTDIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/traces"
mkdir -p "$OUTDIR"

# Fixed decode-heavy operating point (in=8192, out=1024, concurrency=64).
INPUT_LEN=8192
OUTPUT_LEN=1024
CONCURRENCY=64
NUM_PROMPTS=128

# The arm is whatever the image baked, not a runtime choice.
ARM=$([[ "${VLLM_ROCM_USE_AITER_COMMS:-0}" == "1" ]] && echo exp || echo baseline)

# Runtime env shared by both arms.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# V1 spawns worker/engine subprocesses titled "VLLM::*" that outlive "vllm serve";
# reap both so a crash doesn't leave GPUs pinned. || true: nothing to kill is fine.
trap 'pkill -f "vllm serve" 2>/dev/null || true; pkill -9 -f "VLLM::" 2>/dev/null || true' EXIT INT TERM

echo "=========================================="
echo "[repro] arm=$ARM  in=$INPUT_LEN out=$OUTPUT_LEN conc=$CONCURRENCY prompts=$NUM_PROMPTS eval=$EVAL profile=$PROFILE"
echo "=========================================="

# Pre-fetch the model, then go offline.
for attempt in 1 2 3; do
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" \
        --exclude "original/*" --exclude "metal/*" && break
    echo "[repro] HF download failed, retry $attempt/3"; sleep 10
done
if [[ "$EVAL" == "1" ]]; then
    python3 -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Start the server.
serve_args=(
    --host "$HOST" --port "$PORT"
    --max-model-len 10240
    --tensor-parallel-size "$TP"
    --max-num-seqs 1024
    --kv-cache-dtype fp8
    --gpu-memory-utilization 0.90
    --max-num-batched-tokens 131072
    --no-enable-prefix-caching
    --async-scheduling
)
[[ "$PROFILE" == "1" ]] && serve_args+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$OUTDIR\", \"torch_profiler_dump_cuda_time_total\": true, \"torch_profiler_with_stack\": true}")
vllm serve "$MODEL" "${serve_args[@]}" &
SERVER_PID=$!

# Wait for ready (server has been up in 3-8 min; fail at 15).
waited=0
while [[ $waited -lt 900 ]]; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server died during startup"; exit 1; }
    [[ "$(curl -s -o /dev/null -w '%{http_code}' http://${HOST}:${PORT}/v1/models 2>/dev/null || echo 000)" == "200" ]] && break
    sleep 5; waited=$((waited + 5))
    (( waited % 30 == 0 )) && echo "[repro] waiting for server... (${waited}s)"
done
[[ $waited -ge 900 ]] && { echo "ERROR: server not ready within 900s"; exit 1; }
echo "[repro] server ready in ${waited}s"

# Benchmark. --ignore-eos fixes the output length so both arms run the same shape.
bench_args=()
[[ "$PROFILE" == "1" ]] && bench_args+=(--profile)
timeout 1200 vllm bench serve \
    --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$CONCURRENCY" --num-prompts "$NUM_PROMPTS" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --save-result --result-filename "$OUTDIR/result_${ARM}.json" \
    --label "$ARM" \
    "${bench_args[@]}"

# Optional gsm8k accuracy gate against the same server.
if [[ "$EVAL" == "1" ]]; then
    echo "=========================================="
    echo "[repro] gsm8k accuracy (5-shot, 200 samples) arm=$ARM"
    echo "=========================================="
    lm_eval --model local-completions \
        --model_args "model=${MODEL},base_url=http://${HOST}:${PORT}/v1/completions,tensor_parallel_size=${TP},add_bos_token=true,trust_remote_code=true" \
        --batch_size auto --tasks gsm8k --num_fewshot 5 --limit 200 \
        --output_path "$OUTDIR/eval_${ARM}"
fi

kill -TERM "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "[repro] done. results: $OUTDIR/result_${ARM}.json"
