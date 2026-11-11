#!/bin/bash
# _serve.sh - shared vLLM server lifecycle for the llama70b harness. SOURCED by
# bench.sh / profile.sh (perf) and eval.sh (gsm8k correctness gate); never run directly.
#
# The perf run and the correctness gate must hit an IDENTICAL server, so the server
# config + lifecycle live here once. Callers differ only in what they send the live
# server (vllm bench serve for perf; lm_eval gsm8k for correctness).
#
# Contract: the image's baked env decides behavior; nothing is exported at runtime. Set
# ARM_DIR (the output dir) and optionally PROFILE=summary,traces,ir, then call:
# prefetch_model; ensure_sharegpt; start_server; wait_ready; run_workload; stop_server.

MODEL="amd/Llama-3.3-70B-Instruct-FP8-KV"
TP="${TP:-8}"
HOST=localhost
PORT=8000
PROFILE="${PROFILE:-}"               # non-empty => profiling run; bench.sh leaves it empty

# Operating point (the regime being measured). Shared so a profiled run can't drift from
# the perf run; only NUM_PROMPTS (the sample size) is meant to differ. eval.sh ignores
# these (it drives gsm8k, not the bench client).
WORKLOAD="${WORKLOAD:-decode64}"
DATA="${DATA:-random}"
case "$WORKLOAD" in
    confluence) _input=1024; _output=1024; _conc=4;  _prompts=32 ;;
    decode64)   _input=8192; _output=1024; _conc=64; _prompts=128 ;;
    *) echo "ERROR: Unknown WORKLOAD '$WORKLOAD'. Expected: confluence, decode64" >&2; exit 1 ;;
esac
case "$DATA" in
    random|real) ;;
    *) echo "ERROR: Unknown DATA '$DATA'. Expected: random, real" >&2; exit 1 ;;
esac
SHAREGPT_JSON="${SHAREGPT_JSON:-$HOME/.cache/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
INPUT_LEN="${INPUT_LEN:-$_input}"
OUTPUT_LEN="${OUTPUT_LEN:-$_output}"
CONCURRENCY="${CONCURRENCY:-$_conc}"
NUM_PROMPTS="${NUM_PROMPTS:-$_prompts}"

# WARMUP is part of the operating point, not the sample size. Cold (WARMUP=0) folds
# one-time init (NCCL bring-up, first-call/cudagraph costs) into the measured window;
# warm (WARMUP>0) measures steady state. A cold win does not transfer to warm.
# --num-warmups sends that many warmup requests and excludes them from the metrics.
WARMUP="${WARMUP:-0}"

# V1 spawns worker/engine subprocesses retitled "VLLM::Worker_TP"/"VLLM::EngineCore"
# that don't match "vllm serve"; on a crash they orphan and spin in NCCL, pegging every
# GPU until killed. Reap both patterns on exit. `|| true`: pkill returns non-zero when
# nothing matches, and a failing last command in an EXIT trap would fail the whole run.
trap 'pkill -f "vllm serve" 2>/dev/null || true; pkill -9 -f "VLLM::" 2>/dev/null || true' EXIT INT TERM

# All vllm env (behavior flags + shared server config) is BAKED in the Dockerfile, never
# exported here - the image IS the environment. Shared across arms: MHA=0,
# QUICK_REDUCE=INT4, NCCL_DEBUG, the RPC/ready timeouts. Per-arm: VLLM_ROCM_USE_AITER and
# the comms backend selector.

# Echo the baked behavior env to the log, for eyeball. The image IS the arm, so the raw
# flags ARE its identity; print them rather than a derived label. A misconfigured image
# is caught by eye (e.g. baseline must show USE_AITER <unset>, not 1).
print_env() {
    echo "[serve] resolved: gpu_mem_util=$GPU_MEM_UTIL profile=${PROFILE:-(none)}"
    echo "[serve] baked env (from the image; scripts export none of it):"
    local v
    for v in VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_COMMS AITER_COMMS_BACKEND \
             VLLM_ROCM_USE_AITER_MHA VLLM_ROCM_QUICK_REDUCE_QUANTIZATION \
             NCCL_DEBUG VLLM_RPC_TIMEOUT VLLM_ENGINE_READY_TIMEOUT_S; do
        echo "    $v=${!v:-<unset>}"
    done
}

# gpu-memory-utilization is per-arm (declared in each arm's env), not branched here.
# Default 0.94; an arm that needs headroom overrides it (e.g. the gluon path OOMs at
# 0.94). At conc=64 this isn't KV-bound, so the value only sets unused KV headroom.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"

# Fixed `vllm serve` params, hoisted to vars so the serve command reads them directly.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

# Output layout. ARM_DIR (the arm's output dir) is REQUIRED - the runner sets it, we
# crash rather than guess. One folder per output kind, so the analysis maps a folder to a
# parser: vllm/ (bench result JSON), profile/ (torch traces/tables/IR), eval/ (lm_eval
# result JSON). RAW source only; analysis is a separate read-only pass.
ARM_DIR="${ARM_DIR:?ARM_DIR must be set to the arm output dir}"
VLLM_DIR="$ARM_DIR/vllm"          # vLLM bench serve result (bench + profile arms)
EVAL_DIR="$ARM_DIR/eval"          # lm_eval gsm8k result (eval arms)
PROFILE_DIR="$ARM_DIR/profile"
# profile/ splits by kind: summary/ = cheap per-rank key_averages tables; traces/ = heavy
# chrome traces; ir/ = compiled Triton IR. vLLM dumps tables AND traces into one dir, so
# the profiler targets scratch TMP_DIR and profile.sh promotes the keepers afterward.
PROFILE_SUMMARY_DIR="$PROFILE_DIR/summary"
PROFILE_TRACES_DIR="$PROFILE_DIR/traces"
PROFILE_IR_DIR="$PROFILE_DIR/ir"
# Scratch: in-flight output lands here, promoted to the structured dirs at the end.
# Deleted on success; kept on failure for debugging.
TMP_DIR="$ARM_DIR/tmp"

start_server() {
    # Server params match the AMD guide's Llama 3.x FP8 config. gpu-memory-utilization is
    # the per-arm knob (see GPU_MEM_UTIL above).
    local serve_args=(
        --host "$HOST"
        --port "$PORT"
        --max-model-len "$MAX_MODEL_LEN"
        --tensor-parallel-size "$TP"
        --max-num-seqs "$MAX_NUM_SEQS"
        --kv-cache-dtype "$KV_CACHE_DTYPE"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --no-enable-prefix-caching
        --async-scheduling
    )
    # Only wire the torch profiler when profiling. vLLM dumps the chrome trace AND the
    # per-rank key_averages tables into torch_profiler_dir, so it targets scratch TMP_DIR
    # and profile.sh promotes the keepers (tables -> summary/, traces -> traces/) after.
    if [[ -n "${PROFILE:-}" ]]; then
        mkdir -p "$TMP_DIR"
        serve_args+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TMP_DIR\", \"torch_profiler_dump_cuda_time_total\": true, \"torch_profiler_with_stack\": true}")
    fi

    vllm serve "$MODEL" "${serve_args[@]}" &
    SERVER_PID=$!
    echo "[serve] server PID=$SERVER_PID profiler_scratch=$TMP_DIR"
}

wait_ready() {
    # Ready in 3-8 min with warm caches; 15 min means a hang - fail fast.
    local max_wait=900 waited=0 http_code
    while [[ $waited -lt $max_wait ]]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server died during startup"
            return 1
        fi
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://${HOST}:${PORT}/v1/models" 2>/dev/null || echo 000)
        if [[ "$http_code" == "200" ]]; then
            echo "[serve] server ready in ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        (( waited % 30 == 0 )) && echo "  still waiting... (${waited}s, last=${http_code})"
    done
    echo "ERROR: server not ready within ${max_wait}s"
    return 1
}

stop_server() {
    # Proper shutdown (vLLM RFC #24885): SIGTERM the PARENT only; it drains in-flight then
    # reaps its TP workers. Give it a bounded grace (an unbounded wait hangs forever if a
    # worker is wedged in a collective), then escalate to SIGKILL on both "vllm serve" and
    # the retitled "VLLM::" subprocesses that hold GPU memory if orphaned.
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    fi
    if pgrep -f "vllm serve" >/dev/null 2>&1 || pgrep -f "VLLM::" >/dev/null 2>&1; then
        pkill -TERM -f "vllm serve" 2>/dev/null || true
        sleep 3
        pkill -9 -f "vllm serve" 2>/dev/null || true
        pkill -9 -f "VLLM::" 2>/dev/null || true
    fi
    wait_vram_reclaim
}

# KFD memory reclaim lags the server kill by tens of seconds for a 70B TP=8 server, so a
# fixed sleep races the next arm's startup (which needs ~271/288 GiB free per GPU at
# util 0.94). Poll until every GPU is below 10 GiB used; bounded and loud.
wait_vram_reclaim() {
    local smi=/opt/rocm/bin/rocm-smi  # not on PATH in the container
    local deadline=$((SECONDS + 180)) max_used_gb=""
    while (( SECONDS < deadline )); do
        # `found` guards the empty-probe case: no output must NOT read as "0 GiB used"
        # (that fake-success can race the next arm into a startup OOM).
        max_used_gb=$($smi --showmeminfo vram 2>/dev/null \
            | awk '/Used Memory/ {gsub(/[^0-9]/,"",$NF); gb=$NF/1024/1024/1024; if (gb>m) m=gb; found=1} END {if (found) printf "%.0f", m}')
        if [[ -z "$max_used_gb" ]]; then
            echo "[serve] WARNING: rocm-smi VRAM probe failed; sleeping 60s flat instead"
            sleep 60
            return 0
        fi
        (( max_used_gb < 10 )) && return 0
        sleep 5
    done
    echo "[serve] WARNING: VRAM still ~${max_used_gb}GiB used on some GPU after 180s; starting server anyway"
}

# Pre-fetch the model so the run is HF-offline (weights only; datasets are caller-specific).
prefetch_model() {
    echo ""
    echo "=========================================="
    echo "[serve] pre-fetch model"
    echo "=========================================="
    # `hf download` is the current unified CLI (huggingface-cli was removed).
    local attempt
    for attempt in 1 2 3; do
        if hf download "$MODEL" --exclude "original/*" --exclude "metal/*"; then
            return 0
        fi
        echo "[serve] HF download failed, retrying ($((attempt+1))/3)..."
        sleep 10
    done
}

# Fetch the ShareGPT dataset on demand (DATA=real only). No-op otherwise.
ensure_sharegpt() {
    [[ "$DATA" == "real" && ! -f "$SHAREGPT_JSON" ]] || return 0
    echo "[serve] downloading ShareGPT dataset to $SHAREGPT_JSON"
    mkdir -p "$(dirname "$SHAREGPT_JSON")"
    curl -fL --retry 3 -o "$SHAREGPT_JSON" \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
        || { echo "ERROR: ShareGPT download failed (needed for DATA=real)"; exit 1; }
}

# Drive the vllm bench-serve workload against the live server. --ignore-eos fixes output
# length (perf/trace only, never correctness - that's eval.sh). PROFILE set adds --profile.
# Shared by bench.sh (PROFILE empty) and profile.sh (PROFILE set); only NUM_PROMPTS differs.
run_workload() {
    mkdir -p "$VLLM_DIR"
    local profile_args=()
    [[ -n "${PROFILE:-}" ]] && profile_args+=(--profile)
    local dataset_args=()
    if [[ "$DATA" == "real" ]]; then
        dataset_args+=(--dataset-name sharegpt --dataset-path "$SHAREGPT_JSON")
    else
        dataset_args+=(--dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
    fi
    # No in-script timeout: the wall-clock cap is enforced from OUTSIDE, by the runner's
    # per-run container deadline (docker stop then rm -f). An in-script `timeout` would
    # wrap only the bench command and leave a wedged server/teardown unbounded.
    vllm bench serve \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        "${dataset_args[@]}" \
        --max-concurrency "$CONCURRENCY" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$WARMUP" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        --save-result --result-filename "$VLLM_DIR/vllm.json" \
        "${profile_args[@]}" || {
        rc=$?
        echo "ERROR: workload failed (exit $rc)"
        return "$rc"
    }
}
