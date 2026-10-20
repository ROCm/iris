#!/bin/bash
set -euo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

DOCKER_FLAGS=(--rm --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host --network host --shm-size 16g -v "$HOME/.cache:/root/.cache" -v "$HOME/.triton:/root/.triton" -v "$(pwd):/repro" -w /repro)
ATTEMPTS="${ATTEMPTS:-2}"
FAILED=()

build()   { echo "=== build $1 ($2) ==="; docker build -f "$2" -t "$1" .; }
run_arm() {
  local name="$1" img="$2" script="$3" deadline="$4"; shift 4
  local envs=(); for e in "$@"; do envs+=(-e "$e"); done
  local attempt
  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "=== run $name (attempt $attempt/$ATTEMPTS) -> data/$name/ ==="
    rm -rf "data/$name"   # fresh attempt: clear any partial output from a prior try
    if timeout "$deadline" docker run "${DOCKER_FLAGS[@]}" "${envs[@]}" \
         -e "ARM_DIR=/repro/data/$name" "$img" bash -lc "$script" \
       && python3 -u preprocess.py --validate "data/$name"; then   # produced usable output? (-u: stream live)
      echo "=== ok $name ==="; return 0
    fi
    echo "=== $name attempt $attempt failed ==="
  done
  echo "=== FAILED $name (after $ATTEMPTS attempts) ==="; FAILED+=("$name"); return 1
}

build repro:Dockerfile.baseline docker/Dockerfile.baseline
build repro:Dockerfile.aiter docker/Dockerfile.aiter
build repro:Dockerfile.exp docker/Dockerfile.exp
run_arm decode64-baseline-profile-traces-warm repro:Dockerfile.baseline "./scripts/profile.sh" 3600 PROFILE=summary,traces WARMUP=64 WORKLOAD=decode64 || true
run_arm decode64-aiter-profile-traces-warm repro:Dockerfile.aiter "./scripts/profile.sh" 3600 PROFILE=summary,traces WARMUP=64 WORKLOAD=decode64 || true
run_arm decode64-exp-profile-traces-warm repro:Dockerfile.exp "./scripts/profile.sh" 3600 GPU_MEM_UTIL=0.90 PROFILE=summary,traces WARMUP=64 WORKLOAD=decode64 || true
run_arm decode64-baseline-eval repro:Dockerfile.baseline "./scripts/eval.sh" 3600 WORKLOAD=decode64 || true
run_arm decode64-aiter-eval repro:Dockerfile.aiter "./scripts/eval.sh" 3600 WORKLOAD=decode64 || true
run_arm decode64-exp-eval repro:Dockerfile.exp "./scripts/eval.sh" 3600 GPU_MEM_UTIL=0.90 WORKLOAD=decode64 || true
run_arm decode64-baseline-bench-warm repro:Dockerfile.baseline "./scripts/bench.sh" 3600 WARMUP=64 WORKLOAD=decode64 || true
run_arm decode64-aiter-bench-warm repro:Dockerfile.aiter "./scripts/bench.sh" 3600 WARMUP=64 WORKLOAD=decode64 || true
run_arm decode64-exp-bench-warm repro:Dockerfile.exp "./scripts/bench.sh" 3600 GPU_MEM_UTIL=0.90 WARMUP=64 WORKLOAD=decode64 || true

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== done WITH FAILURES (${#FAILED[@]}): ${FAILED[*]} ==="
  echo "the arms that succeeded have raw output in data/ (analyze.sh builds data.csv from them)"
  exit 1
fi
echo "done: raw output in data/ (run ./analyze.sh to build data.csv + report.ipynb)"
