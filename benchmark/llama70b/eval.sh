#!/bin/bash
# eval.sh - e2e correctness GATE for the llama70b arm: gsm8k accuracy via lm_eval
# (5-shot, 200 samples) against the live server. The companion to bench.sh: bench.sh
# measures speed (--ignore-eos, perf-only), eval.sh proves the arm produces RIGHT
# answers (natural EOS, no forced lengths). Separate scripts, never a mode flag on the
# perf run.
#
# The server lifecycle + flags come from _serve.sh, IDENTICAL to bench.sh's - that is
# the point: the gate must certify the SAME server config the perf run measured.
# lm_eval sends its own completion requests with natural stop criteria (never
# --ignore-eos), so the answers are real. The gate is met when BOTH arms (baseline
# image + exp image) pass and match; run this for each image and compare the scores.
#
# The arm is whatever the image baked (Dockerfile.baseline | .exp).
#
# Usage:
#   ./eval.sh                 # gsm8k 5-shot, 200 samples, on the baked arm
#   LIMIT=500 ./eval.sh       # more samples
#   FEWSHOT=8 ./eval.sh       # more shots

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
COMMAND=eval                     # output lands under output/<arm>-eval/ (not used by the report)
source "$SCRIPT_DIR/_serve.sh"   # server lifecycle + output dirs, shared with bench.sh

LIMIT="${LIMIT:-200}"
FEWSHOT="${FEWSHOT:-5}"

run_eval() {
    local arm="$1"
    echo ""
    echo "=========================================="
    echo "[eval] lm_eval gsm8k (${FEWSHOT}-shot, ${LIMIT} samples) arm=$arm"
    echo "=========================================="
    lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://${HOST}:${PORT}/v1/completions,tensor_parallel_size=${TP},add_bos_token=true,trust_remote_code=true" \
        --batch_size auto \
        --tasks gsm8k \
        --num_fewshot "$FEWSHOT" \
        --limit "$LIMIT" \
        --output_path "$RESULTS_DIR" || { echo "ERROR: lm_eval failed for $arm"; return 1; }
}

dump_arm
prefetch_model
# gsm8k must be cached before HF goes offline (lm_eval loads it at eval time).
echo "[eval] pre-caching gsm8k dataset"
python3 -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')" \
    || echo "[eval] WARNING: gsm8k pre-cache failed - eval will fail offline"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULTS_DIR"
start_server
wait_ready
run_eval "$ARM"
stop_server

# Echo the gsm8k score to STDOUT for an at-a-glance "did this arm pass".
echo ""
echo "[eval] done. results under: $RESULTS_DIR"
results_json="$(find "$RESULTS_DIR" -name 'results*.json' 2>/dev/null | sort | tail -1)"
if [[ -n "$results_json" ]]; then
    python3 - "$results_json" "$ARM" <<'PY'
import json, sys
path, arm = sys.argv[1], sys.argv[2]
res = json.load(open(path)).get("results", {}).get("gsm8k", {})
print(f"=== gsm8k correctness gate (arm={arm}) ===")
print(f"source: {path}")
for k, v in sorted(res.items()):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        print(f"  {k}: {v:.4f}")
PY
else
    # A wrapper must report the real outcome: no lm_eval results = the eval didn't
    # complete (engine likely died). Fail the arm, don't pass silently.
    echo "[eval] FATAL: no lm_eval results JSON under $RESULTS_DIR - eval did not complete (check the log for EngineDeadError)." >&2
    exit 1
fi
