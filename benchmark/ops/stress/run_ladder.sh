#!/bin/bash
# Escalate rank count and iteration count for one copy-engine variant and stop at the first failure.
#   run_ladder.sh <host|device> [m_local]
# Runs in the current Python environment (iris + tritonblas + rocm-xio already installed).
# A hung run wedges the amdgpu driver on the machine that hit the problem, so the ladder starts small, uses a
# SIGTERM-only timeout, and writes a progress line (fsynced) before every step to $OUT/logs.
#   OUT=<dir>          where logs and ladder_results.txt go (default ./stress_out)
#   RANKS="2 4 8"      rank counts to try, in order
#   ITERS="10 40 80 160"  iteration counts to try, in order, for each rank count
#   TIMEOUT=240        seconds per run
export HSA_NO_SCRATCH_RECLAIM=1
VARIANT=${1:?host|device}; MLOC=${2:-1024}
OUT=${OUT:-./stress_out}; RANKS=${RANKS:-"2 4 8"}; ITERS=${ITERS:-"10 40 80 160"}; TIMEOUT=${TIMEOUT:-240}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
RES="$OUT/ladder_results.txt"
for N in $RANKS; do
  for I in $ITERS; do
    KEY="$VARIANT ranks=$N iters=$I m_local=$MLOC"
    if grep -q "^PASS $KEY\$" "$RES" 2>/dev/null; then continue; fi
    TAG="ladder_n${N}_i${I}_m${MLOC}_$(date +%H%M%S)"
    echo "$(date +%T) START $KEY" | tee -a "$RES"
    sync
    timeout "$TIMEOUT" torchrun --nproc_per_node="$N" "$HERE/sdma_stress.py" --mode "$VARIANT" --iters "$I" \
      --m_local "$MLOC" --tag "$TAG" --logdir "$OUT/logs" > "$OUT/logs/$TAG.out" 2>&1
    RC=$?
    OK=$(cat "$OUT/logs/${TAG}_${VARIANT}"_r*.log 2>/dev/null | grep -c "done correct=True")
    if [ "$RC" -eq 0 ] && [ "$OK" -eq "$N" ]; then
      echo "PASS $KEY" | tee -a "$RES"
    else
      echo "FAIL $KEY rc=$RC correct_ranks=$OK tag=$TAG" | tee -a "$RES"
      echo "last progress line per rank:"; for f in "$OUT/logs/${TAG}_${VARIANT}"_r*.log; do echo "  $(basename "$f"): $(tail -n1 "$f")"; done
      echo "Stopping. If the GPUs stopped responding, reboot and 'sudo modprobe amdgpu' before anything else."
      exit 1
    fi
  done
done
echo "LADDER COMPLETE $VARIANT m_local=$MLOC" | tee -a "$RES"
