#!/bin/bash
# _serve.sh - shared vLLM server lifecycle for the llama70b reproducer. SOURCED by
# bench.sh (perf), profile.sh (traces), and eval.sh (gsm8k correctness); not a
# command (leading underscore), never run directly.
#
# Why this file exists: the perf run, the profiling run, and the correctness gate
# MUST run against an IDENTICAL server config, or they don't describe the same
# thing. So the server config + lifecycle live HERE, once - no caller can drift it.
# The callers differ only in what they send the live server (vllm bench serve
# --ignore-eos for perf/trace; lm_eval gsm8k with natural EOS for correctness),
# never in how the server is brought up.
#
# All vLLM behavior env (the arm flags AND the shared server config) is BAKED in the
# Dockerfiles, never exported here - the image IS the environment. The arm is
# whatever the image baked: Dockerfile.exp bakes the comms trio (USE_AITER +
# USE_AITER_COMMS + AITER_COMMS_BACKEND=iris); baseline bakes none (QuickReduce).

MODEL="amd/Llama-3.3-70B-Instruct-FP8-KV"
TP="${TP:-8}"
HOST=localhost
PORT=8000
PROFILE="${PROFILE:-}"               # set (any non-empty value) => profiling run; bench.sh leaves it empty

# Operating point - the regime being measured (input/output len, concurrency).
# Shared by bench.sh and profile.sh so a profiled run CANNOT drift from the perf
# run's operating point; only NUM_PROMPTS (the sample size) is meant to differ.
# eval.sh ignores these (it drives gsm8k, not the bench client).
WORKLOAD="${WORKLOAD:-decode64}"
DATA="${DATA:-random}"
case "$WORKLOAD" in
    confluence) _input=1024; _output=1024; _conc=4;  _prompts=32 ;;   # the guide's light example
    decode64)   _input=8192; _output=1024; _conc=64; _prompts=128 ;;  # long context, high concurrency
    *) echo "ERROR: Unknown WORKLOAD '$WORKLOAD'. Expected: confluence, decode64" >&2; exit 1 ;;
esac
case "$DATA" in
    random|real) ;;
    *) echo "ERROR: Unknown DATA '$DATA'. Expected: random, real" >&2; exit 1 ;;
esac
SHAREGPT_JSON="${SHAREGPT_JSON:-$HOME/.cache/repro/ShareGPT_V3_unfiltered_cleaned_split.json}"
INPUT_LEN="${INPUT_LEN:-$_input}"
OUTPUT_LEN="${OUTPUT_LEN:-$_output}"
CONCURRENCY="${CONCURRENCY:-$_conc}"
NUM_PROMPTS="${NUM_PROMPTS:-$_prompts}"

# WARMUP sends that many warmup requests that are EXCLUDED from the reported metrics,
# so one-time init (NCCL bring-up, first-call/cudagraph costs) is not folded into the
# measured window. The published numbers came from the warm config (WARMUP=64); set
# WARMUP=0 for the cold config the AMD guide's command measures.
WARMUP="${WARMUP:-64}"

# V1 spawns worker/engine subprocesses retitled "VLLM::Worker_TP"/"VLLM::EngineCore"
# that DON'T match "vllm serve"; on a crash they orphan and spin in NCCL, pegging
# every GPU until killed. Reap both patterns on exit. || true: pkill returns
# non-zero when nothing matches, and a failing last command in an EXIT trap would
# mark the whole run failed even though the work succeeded.
trap 'pkill -f "vllm serve" 2>/dev/null || true; pkill -9 -f "VLLM::" 2>/dev/null || true' EXIT INT TERM

# The arm is whatever the image baked - never a runtime choice. Dockerfile.exp bakes
# VLLM_ROCM_USE_AITER_COMMS=1 (the iris comms path); baseline bakes nothing.
ARM=$([[ "${VLLM_ROCM_USE_AITER_COMMS:-0}" == "1" ]] && echo exp || echo baseline)

# The command is set by each caller before sourcing (bench.sh=bench, profile.sh=profile,
# eval.sh=eval). It keeps each command's output in its own dir (below), so a profiling run
# does not overwrite the bench run's result JSON, and preprocess.py can label rows by it.
COMMAND="${COMMAND:-bench}"

# gpu-memory-utilization is part of the operating point. exp (iris) needs 0.90: the
# gluon path's 8GB symmetric heap OOMs at the guide's 0.94. baseline runs the guide's
# 0.94. At conc=64 this is not KV-bound (~15-19x KV headroom), so the value only sets
# unused KV and changes no measured number - the two differ only to give iris headroom.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-$([[ "$ARM" == exp ]] && echo 0.90 || echo 0.94)}"

# Fixed `vllm serve` server params (the AMD guide config) - hoisted to vars so the
# serve command below AND dump_arm's record share ONE source (no drift).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

# Per-arm output layout, one dir per (container, command). The arm produces RAW
# artifacts only: results/ (the workload result JSON) and, for a profiling run,
# profile/{summary,traces,ir}/. tmp/ is the profiler's scratch dumping ground, promoted
# to the structured dirs at the end. preprocess.py reads output/*/ to build data.csv.
SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$(realpath "${BASH_SOURCE[0]}")")}"
ARM_DIR="$SCRIPT_DIR/output/${ARM}-${COMMAND}"
RESULTS_DIR="$ARM_DIR/results"
PROFILE_DIR="$ARM_DIR/profile"
PROFILE_SUMMARY_DIR="$PROFILE_DIR/summary"   # per-rank key_averages tables (profiler_out_*.txt, ~20KB)
PROFILE_TRACES_DIR="$PROFILE_DIR/traces"     # heavy chrome traces (*.pt.trace.json.gz)
PROFILE_IR_DIR="$PROFILE_DIR/ir"             # compiled Triton IR (.ttgir/.llir/.amdgcn)
TMP_DIR="$ARM_DIR/tmp"

# Record what actually ran inside this container: the resolved arm + operating point,
# the baked behavior env, and the installed code SHAs (lineage). Console for eyeball
# (<unset> shown explicitly so a misconfigured image is caught by eye) + arm.json.
dump_arm() {
    echo "[serve] resolved: arm=$ARM gpu_mem_util=$GPU_MEM_UTIL profile=${PROFILE:-(none)}"
    echo "[serve] baked env (from the image; scripts export none of it):"
    local v
    for v in VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_COMMS AITER_COMMS_BACKEND \
             VLLM_ROCM_USE_AITER_MHA VLLM_ROCM_QUICK_REDUCE_QUANTIZATION \
             NCCL_DEBUG VLLM_RPC_TIMEOUT VLLM_ENGINE_READY_TIMEOUT_S; do
        echo "    $v=${!v:-<unset>}"
    done
    mkdir -p "$ARM_DIR"
    ARM="$ARM" COMMAND="$COMMAND" MODEL="$MODEL" TP="$TP" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="$MAX_NUM_SEQS" KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    DATA="$DATA" INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" CONCURRENCY="$CONCURRENCY" \
    NUM_PROMPTS="$NUM_PROMPTS" WARMUP="$WARMUP" \
    python3 - "$ARM_DIR/arm.json" <<'PY'
import json, os, subprocess, sys
out = sys.argv[1]
e = os.environ.get
def sha(d):
    try:
        return subprocess.check_output(["git", "-C", d, "log", "--oneline", "-1"], text=True).strip()
    except Exception as ex:
        return f"(unavailable: {ex})"
serve_params = {
    "model": e("MODEL"),
    "max_model_len": int(e("MAX_MODEL_LEN")),
    "tensor_parallel_size": int(e("TP")),
    "max_num_seqs": int(e("MAX_NUM_SEQS")),
    "kv_cache_dtype": e("KV_CACHE_DTYPE"),
    "gpu_memory_utilization": float(e("GPU_MEM_UTIL")),
    "max_num_batched_tokens": int(e("MAX_NUM_BATCHED_TOKENS")),
    "enable_prefix_caching": False,   # scripts always pass --no-enable-prefix-caching
    "async_scheduling": True,         # scripts always pass --async-scheduling
}
bench_params = {
    "dataset": e("DATA"),
    "input_len": int(e("INPUT_LEN")),
    "output_len": int(e("OUTPUT_LEN")),
    "concurrency": int(e("CONCURRENCY")),
    "num_prompts": int(e("NUM_PROMPTS")),
    "num_warmups": int(e("WARMUP")),
    "ignore_eos": True,               # bench/profile pass --ignore-eos
}
behavior = {k: os.environ[k] for k in (
    "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_COMMS", "AITER_COMMS_BACKEND",
    "VLLM_ROCM_USE_AITER_MHA", "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION") if os.environ.get(k)}
repos = {n: sha(f"/src/{n}") for n in ("vllm", "aiter", "iris")}
json.dump({
    "arm": e("ARM"),
    "container": e("ARM"),
    "command": e("COMMAND"),
    "serve_params": serve_params,
    "bench_params": bench_params,
    "behavior_env": behavior,
    "repos": repos,
}, open(out, "w"), indent=2)
print(f"[serve] arm record -> {out}")
PY
}

start_server() {
    # Flags match the AMD guide's Llama 3.x FP8 section. Deviations: no
    # --max-seq-len-to-capture / --swap-space (V0-engine flags the guide sets for
    # vllm 0.10.1; this newer V1 build removed them - --swap-space killed the server
    # at startup). gpu-memory-utilization is the per-arm knob (see GPU_MEM_UTIL above).
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
    # Only wire up the torch profiler when profiling. vLLM dumps BOTH the heavy chrome
    # trace AND the per-rank key_averages tables (profiler_out_*.txt) into
    # torch_profiler_dir - no config to suppress the trace - so it targets the scratch
    # TMP_DIR and profile.sh promotes the keepers (tables -> summary/, traces ->
    # traces/) afterward.
    if [[ -n "${PROFILE:-}" ]]; then
        mkdir -p "$TMP_DIR"
        serve_args+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TMP_DIR\", \"torch_profiler_dump_cuda_time_total\": true, \"torch_profiler_with_stack\": true}")
    fi

    vllm serve "$MODEL" "${serve_args[@]}" &
    SERVER_PID=$!
    echo "[serve] server PID=$SERVER_PID"
}

wait_ready() {
    # Server has historically been ready in 3-8 min; 15 min means a hang - fail fast.
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
    # Proper shutdown (vLLM RFC #24885): SIGTERM the PARENT only; it stops accepting
    # requests, drains in-flight, then reaps its TP worker children. Never signal the
    # workers directly. Give the parent a BOUNDED grace period (an unbounded `wait`
    # hangs forever if the engine is wedged); only if it doesn't exit escalate to KILL.
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

# KFD memory reclaim lags the server kill by tens of seconds for a 70B TP=8 server,
# so back-to-back runs can race the next server's startup allocation. Poll until every
# GPU is below 10 GiB used; bounded + loud. `found` guards the empty-probe case so no
# output does NOT read as "0 GiB used".
wait_vram_reclaim() {
    local smi=/opt/rocm/bin/rocm-smi  # not on PATH in the container
    local deadline=$((SECONDS + 180)) max_used_gb=""
    while (( SECONDS < deadline )); do
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

# Pre-fetch the model so the run is HF-offline. `hf download` (the unified CLI;
# `huggingface-cli` was removed). Dataset prefetch is caller-specific (sharegpt for
# bench/profile, gsm8k for eval) and happens in the caller BEFORE HF_HUB_OFFLINE.
prefetch_model() {
    echo ""
    echo "=========================================="
    echo "[serve] pre-fetch model (arm=$ARM)"
    echo "=========================================="
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

# Drive the vllm bench-serve workload against the live server. $1=arm. --ignore-eos
# fixes output length (deterministic decode) - PERF/TRACE only, never correctness
# (that's eval.sh, gsm8k, natural EOS). A non-empty PROFILE adds --profile, dumping
# per-rank torch traces. The result JSON lands in results/. Shared by bench.sh
# (PROFILE empty) and profile.sh (PROFILE set) so both drive the IDENTICAL client.
run_workload() {
    local arm="$1"
    mkdir -p "$RESULTS_DIR"
    local profile_args=()
    [[ -n "${PROFILE:-}" ]] && profile_args+=(--profile)
    local dataset_args=()
    if [[ "$DATA" == "real" ]]; then
        dataset_args+=(--dataset-name sharegpt --dataset-path "$SHAREGPT_JSON")
    else
        dataset_args+=(--dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
    fi
    vllm bench serve \
        --host "$HOST" --port "$PORT" --model "$MODEL" \
        "${dataset_args[@]}" \
        --max-concurrency "$CONCURRENCY" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$WARMUP" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        --save-result --result-filename "$RESULTS_DIR/vllm_${arm}.json" \
        --label "$arm" \
        "${profile_args[@]}" || {
        rc=$?
        echo "ERROR: workload for $arm failed (exit $rc)"
        return "$rc"
    }
}
