#!/bin/bash
# eval.sh - e2e correctness GATE for the llama70b arm: gsm8k accuracy via lm_eval
# (5-shot, 200 samples) against the live server. The companion to bench.sh: bench.sh
# measures speed (--ignore-eos, perf-only), eval.sh proves the arm produces RIGHT answers
# (natural EOS). It sources _serve.sh so the gate certifies the SAME server config the perf
# run measured; the arms should score the same.
#
# The arm is whatever the image baked (Dockerfile.baseline | .exp | .torch).
#
# Usage:
#   ./eval.sh                 # gsm8k 5-shot, 200 samples, on the baked arm
#   LIMIT=500 ./eval.sh       # more samples
#   FEWSHOT=8 ./eval.sh       # more shots

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source "$SCRIPT_DIR/_serve.sh"   # server lifecycle + output dirs, shared with bench.sh

LIMIT="${LIMIT:-200}"
FEWSHOT="${FEWSHOT:-5}"

# Canonical layout (from _serve.sh): lm_eval's gsm8k output is the eval RESULT,
# its own output kind -> eval/ (preprocess parses eval/ with parse_eval).

run_eval() {
    echo ""
    echo "=========================================="
    echo "[eval] lm_eval gsm8k (${FEWSHOT}-shot, ${LIMIT} samples)"
    echo "=========================================="
    pip install -q "lm_eval[api]" 2>/dev/null || true
    # --log_samples: write per-sample records (question, target, model output, exact_match)
    # to eval/ alongside the score JSON, so a wrong arm can be eyeballed (garbage tokens vs
    # empty vs a real miss) instead of just seeing 0.000. Small (a few hundred rows).
    lm_eval \
        --model local-completions \
        --model_args "model=${MODEL},base_url=http://${HOST}:${PORT}/v1/completions,tensor_parallel_size=${TP},add_bos_token=true,trust_remote_code=true" \
        --batch_size auto \
        --tasks gsm8k \
        --num_fewshot "$FEWSHOT" \
        --limit "$LIMIT" \
        --log_samples \
        --output_path "$EVAL_DIR" || { echo "ERROR: lm_eval failed"; return 1; }
}

print_env
prefetch_model
# gsm8k must be cached before HF goes offline (lm_eval loads it at eval time).
echo "[eval] pre-caching gsm8k dataset"
python3 -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')" \
    || echo "[eval] WARNING: gsm8k pre-cache failed - eval will fail offline"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_DIR"
start_server
wait_ready
run_eval
stop_server

# RAW only: lm_eval's results JSON lands in eval/ (the source). The runner validates
# completeness and prints the gsm8k score (run.sh calls `preprocess.py --validate`);
# analysis surfaces gsm8k from the raw eval/ JSON.
echo "[eval] done. arm dir: $ARM_DIR (eval/ - raw)"
