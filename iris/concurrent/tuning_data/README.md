# iris.concurrent tuning corpus

Committed measurements of the `iris.concurrent.gemm` config space (the GEMM/comm
CU split `gemm_wgs` and the GEMM tile `gemm_block`) across shapes, collectives,
world sizes, and **architectures**. This is the empirical basis for:

1. the **universally-good candidate set** the autotuner seeds from (so it works
   with no cost model — see `iris/concurrent/autotune.py`), and
2. the **validation set** for the analytical predictor (`predictor.py`) added later.

Only the **derived** `candidate_set.json` (a runtime dependency of the tuner) and
the collect/derive **tooling** live here. The **raw per-shape grid measurements**
(`data/<arch>_world<W>.json`, several MB and growing per arch × world) are
research/provenance, not needed at runtime, and are kept in the origami_comms
workbench instead of bloating this repo:

```
origami_comms  ->  microbench/.../iris/concurrent/tuning_data/data/<arch>_world<W>.json
```

Point `collect_corpus.py --out` / `derive_candidates.py` at that directory to
regenerate. The committed `candidate_set.json` here is the derived output of record.

## Layout

```
tuning_data/
  collect_corpus.py     # on-device grid sweep  -> <out>/<arch>_world<W>.json
  derive_candidates.py  # corpus -> candidate_set.json (pure Python, no GPU)
  candidate_set.json    # DERIVED: per-arch ordered candidate list + coverage (shipped)
  # data/ (raw measurements) lives in origami_comms, not here
```

## Raw data schema (`data/<arch>_world<W>.json`)

```jsonc
{
  "arch": "gfx942",             // gcnArchName, ":"-stripped (gfx942=MI300X, gfx950=MI350X/MI355X)
  "device_name": "...",
  "world_size": 2,
  "cu_count": 304,
  "torch_version": "...", "python": "...", "timestamp_utc": "...",
  "grid": {                     // the config grid that was swept
    "split_fracs": [0.55, ...], // gemm_wgs = round(frac * cu_count)
    "tiles": [[256,256,64], ...],
    "default_frac": 0.75        // the current static default (num_wgs*3//4)
  },
  "n_warmup": 3, "n_repeat": 8,
  "results": [
    {
      "collective": "all_gather",
      "M": 4096, "N": 4096, "K": 4096, "comm_m": 512, "comm_n": 2048,
      "best":     {"frac": 0.96, "gemm_wgs": 292, "gemm_block": [256,256,64], "ms": 0.49},
      "baseline": {"frac": 0.75, "gemm_wgs": 228, "gemm_block": [256,64,64],  "ms": 0.70},
      "grid": [ {"frac":..., "gemm_wgs":..., "gemm_block":[...], "ms":...}, ... ]  // every point
    },
    ...
  ]
}
```

`ms` is the **median** over `n_repeat` timed iterations (via `iris.do_bench`,
cross-rank barriers + L2 flush). `NaN` means that config failed to launch.

## Derived schema (`candidate_set.json`)

Per arch, an **ordered** list of `(split_frac, gemm_block)` plus a coverage curve.
The order is a greedy set-cover: item `k` is the config that makes the most
still-uncovered corpus shapes land within `TOL` (default 3%) of their measured
best. `coverage[k]` reports `pct_within_tol`, `mean_slowdown`, `max_slowdown` for
benchmarking the first `k` configs — i.e. how good "top-k, no cost model" is.

## How the autotuner uses this

- **No cost model:** seed candidates from `candidate_set.json[arch].order`,
  benchmark the top-`k` (+ always the static default), keep the fastest, cache it.
- **With cost model:** `predictor.predict_split` re-sorts the *same* candidate
  list per shape, so a small `k` reliably contains the optimum. The candidate
  set and the benchmark/cache loop are unchanged — the model is a drop-in ranker.

Either way the on-device benchmark + always-include-default guard means a
suboptimal ordering costs tuning *time*, never correctness or a regression.

## Regenerate

On a node with the target GPUs (see repo `AGENTS.md` / cluster skills for the
ROCm 7.x environment iris needs):

```bash
# CORPUS = the origami_comms data dir (raw measurements live there, not in iris)
CORPUS=/path/to/origami_comms/.../iris/concurrent/tuning_data/data

# one run per world size you care about (W = nproc = #GPUs used)
torchrun --nproc_per_node=2 -m iris.concurrent.tuning_data.collect_corpus --out "$CORPUS"
torchrun --nproc_per_node=4 -m iris.concurrent.tuning_data.collect_corpus --out "$CORPUS"
torchrun --nproc_per_node=8 -m iris.concurrent.tuning_data.collect_corpus --out "$CORPUS"
# smoke test first with:  ... collect_corpus --out "$CORPUS" --quick

# then derive candidate_set.json from that corpus (pure Python, no GPU):
python -m iris.concurrent.tuning_data.derive_candidates --data "$CORPUS"
```

Each `collect_corpus` run writes one `<arch>_world<W>.json` into the corpus dir
(commit those to origami_comms). Commit the regenerated `candidate_set.json` here.

## Coverage status

<!-- keep this table in sync with committed data/ files -->

| arch | device | worlds collected | shape-points | notes |
|------|--------|------------------|--------------|-------|
| gfx942 | MI300X | 2, 4, 8 | 180 | — |
| gfx950 | MI350X | 2, 4, 8 | 180 | — |

(per-arch CU count is recorded in each `data/*.json` `cu_count` field.)

Each `(arch, world)` file = 60 rows (4 collectives × 15 shapes), each row a full
`10-frac × 5-tile` grid (median of 8 timed iters). Collected 2026-07-20.

### Derived candidate-set coverage (top-k, no cost model)

`derive_candidates.py` output — fraction of the 180 per-arch shape-points whose
optimum is recovered within 3% by benchmarking the first `k` configs, and the
mean/worst slowdown vs the per-shape measured best:

| arch | k=1 | k=4 | k=8 | full (100%) | k=8 mean / max slowdown |
|------|-----|-----|-----|-------------|--------------------------|
| gfx942 (MI300X) | 25.0% | 51.1% | 73.9% | k=25 | 1.02× / 1.23× |
| gfx950 (MI350X) | 37.8% | 71.7% | 91.7% | k=15 | 1.009× / 1.07× |

Takeaway: the two archs need **different orderings** (gfx942 leads with
`frac=0.92, 256×256`; gfx950 with `frac=0.84, 256×128`) and different depths
(gfx942 converges at k=25, gfx950 at k=15). Even at k=8 with **no cost model**,
mean is within ~1–2% of optimal — the cost model's job is to shrink k further by
re-sorting per shape.
