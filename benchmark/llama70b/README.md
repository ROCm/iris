# iris all-reduce reproducer

Correctness and performance numbers for the iris one-shot collectives on
Llama-3.3-70B-FP8 at TP=8. Each arm is a baked image (vllm, aiter, iris pinned);
the scripts only run the workload.

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

## Baseline

```sh
docker build -f Dockerfile.baseline -t iris-repro:baseline .
$RUN iris-repro:baseline ./bench.sh    # performance
$RUN iris-repro:baseline ./test.sh     # correctness
```

## Experiment

```sh
docker build -f Dockerfile.exp -t iris-repro:exp .
$RUN iris-repro:exp ./bench.sh         # performance
$RUN iris-repro:exp ./test.sh          # correctness
```

`bench.sh` writes serving metrics (TTFT/TPOT/E2EL/throughput) to `./traces/`;
add `TEST=1` for a gsm8k accuracy pass or `PROFILE=1` for profiler traces.
