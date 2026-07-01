#!/bin/bash
# profile.sh - vLLM Llama-3.3-70B-FP8 TP=8 PROFILING run (one baked arm).
#
# The mechanism run: same workload + operating point as bench.sh (shared via
# _serve.sh), but with the profiler ON, dumping the source data needed to explain WHY
# e2e looks the way it does - per-rank torch traces, the per-rank key_averages tables,
# optionally the Triton IR. A profiling run is a DIFFERENT run from the bench (tracing
# tax is asymmetric), so it's its own script, never the headline number. It writes RAW
# only; the analysis (preprocess.py -> data.csv, report.ipynb) is a separate pass.
#
# The arm is whatever the image baked (Dockerfile.baseline | .exp).
#
# Usage (PROFILE selects WHICH artifacts to keep - a comma list):
#   ./profile.sh                         # DEFAULT summary,traces - what data.csv needs
#   PROFILE=summary ./profile.sh         # tables only (skip the GB chrome traces)
#   PROFILE=summary,ir ./profile.sh      # also keep the Triton IR (.ttgir/.llir/.amdgcn)
#   PROFILE=all ./profile.sh             # everything: summary + traces + ir
#   NUM_PROMPTS=32 ./profile.sh          # smaller/faster trace

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export NUM_PROMPTS="${NUM_PROMPTS:-64}"  # smaller sample than bench (traces are huge); same operating point
# PROFILE is the one profiling knob. Its PRESENCE tells the shared _serve.sh this is a
# profiling run (bench.sh never sets it, so it adds no --profile); its VALUE is the
# comma list of artifacts to KEEP. Default summary,traces is what data.csv needs: the
# per-rank key_averages tables (L3 per-kernel) AND the chrome traces (L2 compute/comms
# split). ir (the Triton codegen dump) is opt-in. `all` = summary,traces,ir.
export PROFILE="${PROFILE:-summary,traces}"
_keep() { [[ ",$PROFILE," == *",$1,"* || ",$PROFILE," == *",all,"* ]]; }
KEEP_TRACES=0; _keep traces && KEEP_TRACES=1
KEEP_IR=0;     _keep ir     && KEEP_IR=1
COMMAND=profile                          # output lands under output/<arm>-profile/
source "$SCRIPT_DIR/_serve.sh"           # server + operating point + run_workload + output dirs

# Only make the dirs for what we keep (no empty dirs).
mkdir -p "$RESULTS_DIR" "$PROFILE_SUMMARY_DIR" "$TMP_DIR"
(( KEEP_TRACES )) && mkdir -p "$PROFILE_TRACES_DIR"

# IR (opt-in): route Triton's JIT cache into profile/ir/ so the per-kernel
# .ttgir/.llir/.amdgcn (which localize codegen/fence bugs) ride along. A fresh dir
# forces a COLD compile so the dump is complete. When NOT keeping IR, leave
# TRITON_CACHE_DIR at its default (the warm cache) - no cold-compile tax.
if (( KEEP_IR )); then
    export TRITON_CACHE_DIR="$PROFILE_IR_DIR"
    mkdir -p "$TRITON_CACHE_DIR"
fi

dump_arm
prefetch_model
ensure_sharegpt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo ""
echo "=========================================="
echo "[profile] arm=$ARM input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS concurrency=$CONCURRENCY warmup=$WARMUP profile=$PROFILE (PROFILED)"
echo "=========================================="

start_server                    # PROFILE set -> server wires the torch profiler to tmp/ (scratch)
wait_ready
run_workload "$ARM"             # PROFILE set -> --profile, dumps per-rank torch traces to tmp/
stop_server

# Promote the keepers out of the scratch tmp/ into the structured dirs. vLLM dumps
# BOTH the per-rank key_averages tables (profiler_out_*.txt) AND the chrome traces
# into tmp/. The tables (cheap, high-signal) always promote to summary/; the GB .gz
# traces promote to traces/ only when `traces` is in PROFILE.
mv "$TMP_DIR"/profiler_out_*.txt "$PROFILE_SUMMARY_DIR"/ 2>/dev/null || true
if (( KEEP_TRACES )); then
    mv "$TMP_DIR"/*.pt.trace.json.gz "$PROFILE_TRACES_DIR"/ 2>/dev/null || true
else
    echo "[profile] PROFILE=$PROFILE: chrome traces not kept (left in tmp/, cleaned on success)"
fi

# A wrapper must report the real outcome, and `vllm bench serve --profile` swallows a
# failed /stop_profile (it 500s but the CLI still exits 0), so a crashed profiler-stop
# would otherwise record a silent "done exit 0". Fail loudly (BEFORE the tmp/ cleanup,
# so the scratch survives for debugging) when the expected RAW output is missing.
shopt -s nullglob
[ -f "$RESULTS_DIR/vllm_${ARM}.json" ] || { echo "[profile] FATAL: no result JSON ($RESULTS_DIR/vllm_${ARM}.json) - workload did not complete (engine likely died; check the log for EngineDeadError)." >&2; exit 1; }
_tables=("$PROFILE_SUMMARY_DIR"/profiler_out_*.txt)
(( ${#_tables[@]} )) || { echo "[profile] FATAL: no profiler_out_*.txt under $PROFILE_SUMMARY_DIR - the collective /stop_profile crashed the engine (worker died mid-stop). NOT a valid profile; failing the arm." >&2; exit 1; }

echo ""
echo "[profile] raw source under: $RESULTS_DIR (result), $PROFILE_DIR (summary/ traces/ ir/)"

# tmp/ is scratch. Reaching this line means the run SUCCEEDED (set -e aborts on any
# real failure before here), so the un-promoted leftovers (the heavy chrome traces a
# summary-only run didn't keep) are no longer needed: delete tmp/ to keep the output
# small. The keepers were already promoted (summary/ traces/ ir/).
rm -rf "$TMP_DIR"
echo "[profile] run ok - cleaned scratch tmp/"
