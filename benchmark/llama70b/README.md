# llama70b reproducer

Self-contained reproducer. `./run.sh` builds each arm's image, runs its workload,
then extracts a flat `data.csv` and renders `report.ipynb`. It runs standalone:
`./run.sh` is exactly what produced these numbers.

## Requirements

- Docker with ROCm device access (the run mounts /dev/kfd, /dev/dri, video group).
- python3 with pandas, matplotlib, and perfetto (for `preprocess.py` + the notebook).

## Run

```sh
./run.sh       # build images, run every arm -> raw output in data/
./analyze.sh   # preprocess data/ -> data.csv, render report.ipynb (re-runnable)
```

## Arms

- `decode64-baseline-profile-traces-warm`: build `docker/Dockerfile.baseline`, run `./scripts/profile.sh` (PROFILE=summary,traces, WARMUP=64, WORKLOAD=decode64)
- `decode64-aiter-profile-traces-warm`: build `docker/Dockerfile.aiter`, run `./scripts/profile.sh` (PROFILE=summary,traces, WARMUP=64, WORKLOAD=decode64)
- `decode64-exp-profile-traces-warm`: build `docker/Dockerfile.exp`, run `./scripts/profile.sh` (GPU_MEM_UTIL=0.90, PROFILE=summary,traces, WARMUP=64, WORKLOAD=decode64)
- `decode64-baseline-eval`: build `docker/Dockerfile.baseline`, run `./scripts/eval.sh` (WORKLOAD=decode64)
- `decode64-aiter-eval`: build `docker/Dockerfile.aiter`, run `./scripts/eval.sh` (WORKLOAD=decode64)
- `decode64-exp-eval`: build `docker/Dockerfile.exp`, run `./scripts/eval.sh` (GPU_MEM_UTIL=0.90, WORKLOAD=decode64)
- `decode64-baseline-bench-warm`: build `docker/Dockerfile.baseline`, run `./scripts/bench.sh` (WARMUP=64, WORKLOAD=decode64)
- `decode64-aiter-bench-warm`: build `docker/Dockerfile.aiter`, run `./scripts/bench.sh` (WARMUP=64, WORKLOAD=decode64)
- `decode64-exp-bench-warm`: build `docker/Dockerfile.exp`, run `./scripts/bench.sh` (GPU_MEM_UTIL=0.90, WARMUP=64, WORKLOAD=decode64)

## Output

- `data/<arm>/` - raw per-arm output (results JSONs, profiler tables, traces).
- `data.csv` - flat long-format extract, one row per fact (built by `preprocess.py data/*`).
- `report.ipynb` - the A/B analysis rendered from `data.csv`.
