#!/bin/bash
# Run the 2-rank stress on every GPU pair, one pair at a time, to look for a bad GPU or link.
#   pair_sweep.sh <host|device> [iters] [m_local]
# Stops at the first failing or hung pair. Resumable: pairs that already passed are skipped.
#   OUT=<dir> (default ./stress_out)   TIMEOUT=200   NGPUS=8
# RAS counters (rocm-smi --showrasinfo all) are saved before the sweep and after every pair under $OUT/ras.
export HSA_NO_SCRATCH_RECLAIM=1
VARIANT=${1:?host|device}; ITERS=${2:-150}; MLOC=${3:-1024}
OUT=${OUT:-./stress_out}; TIMEOUT=${TIMEOUT:-200}; NGPUS=${NGPUS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs" "$OUT/ras"
RES="$OUT/pair_results.txt"
KEY="$VARIANT iters=$ITERS m_local=$MLOC"
ras() { rocm-smi --showrasinfo all 2>/dev/null; }

# Disjoint cover first, then every other pair.
PAIRS=""
for ((a = 0; a + 1 < NGPUS; a += 2)); do PAIRS="$PAIRS $a,$((a + 1))"; done
for ((a = 0; a < NGPUS; a++)); do for ((b = a + 1; b < NGPUS; b++)); do PAIRS="$PAIRS $a,$b"; done; done
PAIRS=$(echo $PAIRS | tr ' ' '\n' | awk '!seen[$0]++')

ras > "$OUT/ras/before_$(date +%H%M%S).txt"
for P in $PAIRS; do
  if grep -q "^PASS $KEY gpus=$P\$" "$RES" 2>/dev/null; then continue; fi
  TAG="pair${P/,/-}_${VARIANT}_i${ITERS}_$(date +%H%M%S)"
  echo "$(date +%T) START $KEY gpus=$P" | tee -a "$RES"
  sync
  HIP_VISIBLE_DEVICES=$P timeout "$TIMEOUT" torchrun --nproc_per_node=2 "$HERE/sdma_stress.py" --mode "$VARIANT" \
    --iters "$ITERS" --m_local "$MLOC" --tag "$TAG" --logdir "$OUT/logs" > "$OUT/logs/$TAG.out" 2>&1
  RC=$?
  OK=$(cat "$OUT/logs/${TAG}_${VARIANT}"_r*.log 2>/dev/null | grep -c "done correct=True")
  ras > "$OUT/ras/after_$TAG.txt"
  if [ "$RC" -eq 0 ] && [ "$OK" -eq 2 ]; then
    echo "PASS $KEY gpus=$P" | tee -a "$RES"
  else
    echo "FAIL $KEY gpus=$P rc=$RC correct_ranks=$OK tag=$TAG" | tee -a "$RES"
    for f in "$OUT/logs/${TAG}_${VARIANT}"_r*.log; do echo "  $(basename "$f"): $(tail -n1 "$f")"; done
    echo "Stopping at the first failure. Reboot and 'sudo modprobe amdgpu' if the GPUs stopped responding."
    exit 1
  fi
done
echo "ALL PAIRS PASSED $KEY" | tee -a "$RES"
