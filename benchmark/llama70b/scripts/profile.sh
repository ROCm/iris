#!/bin/bash
# profile.sh - vLLM Llama 3.3 70B FP8 TP=8 PROFILING run (one baked arm).
#
# Same workload + operating point as bench.sh, but with the profiler ON, dumping the
# source data that explains WHY e2e looks the way it does: per-rank torch traces, the
# Triton JIT cache (.ttgir/.llir/.amdgcn), and the e2e result JSON. RAW only; analysis is
# a separate read-only pass. A profiling run is a DIFFERENT run from the bench (the
# tracing tax is asymmetric), so it's its own script, never the headline number. Sample
# size is smaller (traces are huge) but the operating point is identical (via _serve.sh),
# so per-call kernel costs transfer.
#
# The arm is whatever the image baked (Dockerfile.baseline | .exp | .torch).
#
# Usage (PROFILE = comma list of artifacts to keep; summary is always kept):
#   ./profile.sh                         # SUMMARY-ONLY (the cheap key_averages tables)
#   PROFILE=summary,traces ./profile.sh  # also keep the GB chrome traces (timeline)
#   PROFILE=summary,ir ./profile.sh      # also keep the Triton IR (codegen debugging)
#   PROFILE=all ./profile.sh             # everything: summary + traces + ir
#   NUM_PROMPTS=32 ./profile.sh          # smaller/faster trace

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export NUM_PROMPTS="${NUM_PROMPTS:-64}"  # smaller sample than bench (traces are huge); same operating point
# PROFILE is the one profiling knob. Its PRESENCE tells _serve.sh this is a profiling run
# (bench.sh never sets it); its VALUE is the comma list of artifacts to KEEP. summary is
# always kept (cheap per-rank key_averages tables); traces (GB timeline) and ir (Triton
# codegen) are heavy opt-ins. Export so the sourced _serve.sh sees it. `all` = all three.
export PROFILE="${PROFILE:-summary}"
_keep() { [[ ",$PROFILE," == *",$1,"* || ",$PROFILE," == *",all,"* ]]; }
KEEP_TRACES=0; _keep traces && KEEP_TRACES=1
KEEP_IR=0;     _keep ir     && KEEP_IR=1
source "$SCRIPT_DIR/_serve.sh"           # server + operating point + run_workload + output dirs

# Per-arm layout from _serve.sh: vllm/ (result JSON), profile/{summary,traces,ir}/, tmp/
# (scratch - the profiler dumps here, promoted to the structured dirs at the end). Only
# make the dirs for what we keep (no empty dirs).
mkdir -p "$VLLM_DIR" "$PROFILE_SUMMARY_DIR" "$TMP_DIR"
(( KEEP_TRACES )) && mkdir -p "$PROFILE_TRACES_DIR"

# IR (opt-in): route Triton's JIT cache into profile/ir/ so the per-kernel
# .ttgir/.llir/.amdgcn (which localize codegen/fence bugs) ride back. A fresh dir forces a
# COLD compile so the dump is complete. When NOT keeping IR, leave TRITON_CACHE_DIR at its
# default (the warm, mounted ~/.triton cache) - no cold-compile tax, nothing promoted.
if (( KEEP_IR )); then
    export TRITON_CACHE_DIR="$PROFILE_IR_DIR"
    mkdir -p "$TRITON_CACHE_DIR"
fi

print_env
prefetch_model
ensure_sharegpt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo ""
echo "=========================================="
echo "[profile] input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$CONCURRENCY warmup=$WARMUP profile=$PROFILE (PROFILED)"
echo "=========================================="

start_server                    # PROFILE set -> server wires the torch profiler to tmp/ (scratch)
wait_ready
run_workload                    # PROFILE set -> --profile, dumps per-rank torch traces to tmp/
stop_server

# Promote the keepers out of scratch tmp/ into the structured dirs. vLLM dumps BOTH the
# key_averages tables (profiler_out_*.txt) AND the chrome traces into tmp/. Tables always
# promote to summary/; the GB traces promote to traces/ only when `traces` is in PROFILE
# (else they stay in tmp/, cleaned on success).
mv "$TMP_DIR"/profiler_out_*.txt "$PROFILE_SUMMARY_DIR"/ 2>/dev/null || true
if (( KEEP_TRACES )); then
    mv "$TMP_DIR"/*.pt.trace.json.gz "$PROFILE_TRACES_DIR"/ 2>/dev/null || true
else
    echo "[profile] PROFILE=$PROFILE: chrome traces not kept (left in tmp/, cleaned on success)"
fi

echo ""
echo "[profile] raw source under: $VLLM_DIR (result+provenance), $PROFILE_DIR (summary/ traces/ ir/)"
echo "[profile] arm dir: $ARM_DIR (vllm/ profile/ - raw)"

# tmp/ is scratch. Clean it only when the profiler tables promoted (the run's keepers are
# safe in summary/); if none did, a /stop_profile crash left tmp/ un-promoted - KEEP it for
# debugging. The runner gates the arm pass/fail (run.sh calls `preprocess.py --validate`,
# which catches an empty profile/summary/); this is just scratch hygiene, not the gate.
shopt -s nullglob
_tables=("$PROFILE_SUMMARY_DIR"/profiler_out_*.txt)
if (( ${#_tables[@]} )); then
    rm -rf "$TMP_DIR"
    echo "[profile] cleaned scratch tmp/"
else
    echo "[profile] no profiler tables promoted - keeping tmp/ for debugging"
fi
