# iris all-reduce reproducer

Correctness and performance for the iris one-shot collectives on
Llama-3.3-70B-FP8 at TP=8. Two arms, each a baked image (vllm, aiter, iris pinned):

- **baseline** (`Dockerfile.baseline`) - AMD production config: broad aiter off,
  the all-reduce is QuickReduce INT4.
- **exp** (`Dockerfile.exp`) - the iris one-shot all-reduce (`AITER_COMMS_BACKEND=iris`).

The A/B isolates the all-reduce path. All vLLM behavior env is baked into the images;
the scripts only run the workload against the server.

## Requirements

- 8x MI350 (gfx950)
- Docker with ROCm device access

```sh
RUN="docker run --rm -it \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc host --network host --shm-size 16g \
  -v $(pwd):/repro -w /repro"
```

## Run

Build each image once, then run the commands you need against it. The A/B is the
baseline image vs the exp image; compare their outputs.

```sh
docker build -f Dockerfile.baseline -t iris-repro:baseline .
docker build -f Dockerfile.exp      -t iris-repro:exp      .

$RUN iris-repro:baseline ./bench.sh     # perf: serving metrics (TTFT/TPOT/E2EL/throughput)
$RUN iris-repro:exp      ./bench.sh

$RUN iris-repro:baseline ./profile.sh   # traces + per-kernel tables (what data.csv needs)
$RUN iris-repro:exp      ./profile.sh

$RUN iris-repro:baseline ./eval.sh      # gsm8k accuracy gate (correctness)
$RUN iris-repro:exp      ./eval.sh

$RUN iris-repro:exp      ./test.sh      # iris collective correctness (exp stack, no server)
```

The four commands share one server config (`_serve.sh`) so perf, traces, and the
correctness gate all describe the same server. Each writes RAW artifacts under
`output/<arm>/`: `results/` (the workload result JSON), `profile/{summary,traces,ir}/`
(profiling run), and `arm.json` (the resolved operating point + installed code SHAs).

Default operating point is `decode64` (8192 in / 1024 out, concurrency 64), warm
(`WARMUP=64`, warmup requests excluded from the metrics). Knobs: `WORKLOAD=confluence`
(the guide's 1024/1024/conc-4 example), `WARMUP=0` (cold), `DATA=real` (ShareGPT).

## Analysis

`data.csv` is the flat, long-format extract of every arm's raw artifacts (one row per
fact: e2e metrics, profiler-table per-kernel times, trace per-kernel times). `report.ipynb`
renders the A/B from it (pandas + matplotlib only). It ships pre-built, so you can open the
notebook directly.

To rebuild it from arms you ran yourself, point `preprocess.py` at their output dirs (each
`output/<arm>-<command>/` holds `arm.json` + `results/` + `profile/`):

```sh
python preprocess.py output/*     # parses each arm dir -> data.csv (reads arm.json for the labels)
jupyter nbconvert --to notebook --execute report.ipynb   # or just open report.ipynb
```
