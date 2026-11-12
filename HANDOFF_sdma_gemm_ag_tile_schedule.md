# Handoff: GEMM + all-gather copy engine on the tritonBLAS tile schedule

Branch `dev/dasidler/sdma-gemm-ag-tile-schedule` (iris) with tritonBLAS `dev/dasidler/expose-tile-schedule-on-main`.
Written 2026-11-12 so the work can be continued and validated on a different machine. Remove this file before
opening a PR.

## 1. What this is

Use the tile-schedule API from Ryan Swann's tritonBLAS branch (`swann/expose-tile-schedule`) in the
`matmul_all_gather` copy-engine ops (device-initiated and host-initiated), instead of iris's own re-implementation
of the GEMM's tile order (`iris/ops/tritonblas_launch_wave_schedule.py`, now deleted).

| Repo | Branch | Commit | Based on |
|---|---|---|---|
| iris | `dev/dasidler/sdma-gemm-ag-tile-schedule` | see `git log` | `origin/dev/dasidler/sdma-benchmark` = `d76a9d9c` |
| tritonBLAS | `dev/dasidler/expose-tile-schedule-on-main` | `426fad22` | tritonBLAS `main` (`b9baf7d`) + Ryan's branch merged + `table` map type |

`pyproject.toml` pins tritonBLAS to `426fad2296bfb79222efbfe80784182e688393be` and rocm-xio to `97fa1663` (`pip install -e ".[sdma]"`).

### iris changes
- `iris/ops/tritonblas_tile_schedule.py` (new): `build_launch_wave_signal_plan(m, n, selector, num_cus)` calls
  `tritonblas.schedule_for_selector`, assigns one gate per launch wave with `TileSchedule.with_signal_layout`, and
  coalesces each gate's tiles into rectangular copies (per `GROUP_M` band, which reproduces the old transfer counts).
- `iris/ops/matmul_all_gather_copy_engine.py` and `..._host_copy_engine.py`: gates, producer counts and SDMA queue
  order now come from that plan. The GEMM signals through `EpilogueConfig(map_type="table", tile_to_slot=...)`.
- `tests/unittests/test_tritonblas_tile_schedule.py` (new, host only) and an extended
  `tests/ops/test_matmul_all_gather_copy_engine.py` (second multi-wave shape, two launches with increasing `flag_iteration`).
- `benchmark/ops/stress/` (new): `sdma_stress.py`, `run_ladder.sh`, `pair_sweep.sh` (see section 5).

### tritonBLAS changes (on top of Ryan's branch)
- Merge of `swann/expose-tile-schedule` onto current main, keeping main's `EpilogueConfig`/`PrologueConfig`.
- `EpilogueConfig(map_type="table", tile_to_slot=...)`: arbitrary tile -> counter-offset table (indexed by row-major
  linear tile id), validated once at construction, copied to the device. `PrologueConfig` rejects it.
- Test fixes in `tests/test_schedule.py` (an `import` inside `@triton.jit` and a list-vs-tuple compare).
- Ryan's `gates=` / `signal_of_tile=` / `gate_stride=` / `gate_base=` / `signal_times=` path is still there, unused by iris.

## 2. Verified (original machine: 8x MI300X, container with torch 2.9.1+rocm7.1, triton 3.6.0)
- tritonBLAS: `tests/test_sync_config.py tests/test_schedule.py tests/test_gate_layout.py` -> 98 passed, including the
  on-device check that the published tile order matches the kernel, and table-vs-`launch_wave` counter equality.
- iris: `tests/unittests/test_tritonblas_tile_schedule.py` (7 passed); `tests/ops/test_matmul_all_gather_copy_engine.py`
  passed at 2 and 8 ranks; `test_all_gather_matmul_copy_engine.py`, `test_matmul_all_gather.py`,
  `test_all_gather_matmul.py` passed at 2 ranks (41 tests). All of these run only a couple of iterations.
- Planned transfers match the old implementation (per-wave tile sets and producer counts identical for
  M_local 1024/4096/16384, N=3584, K=8192; rectangle counts equal after per-band coalescing).
- Stress runs (`sdma_stress.py`, fp16, N=3584, K=8192): 2 ranks, host and device, 40 iterations (baseline and this branch)
  and 150 iterations at M_local 16384 (baseline) all passed with correct output.
- Throughput, baseline `d76a9d9c`, 8 ranks, 5 warmup + 2 timed iterations, GPU time in ms for M_local 1024 / 4096 / 16384:
  host 0.447 / 1.003 / 3.327, device 0.522 / 1.250 / 3.578. **No numbers exist yet for this branch** (they were lost
  when the machine reset).

## 3. NOT verified: a hang that also hits the baseline

On the original machine the copy-engine ops hung or reset the host above 2 ranks after enough iterations, including on
the unmodified baseline `d76a9d9c`:

- 4 ranks, host variant, M_local 1024, baseline: iterations 0-74 finish in milliseconds, then all ranks hang at
  iteration 75 inside the op. Afterwards the amdgpu driver is wedged (`ttm` kworkers in D state, leaked VRAM,
  `torch.cuda.is_available()` False) and only a reboot plus `sudo modprobe amdgpu` recovers.
- 8 ranks, `benchmark/ops/bench_matmul_all_gather_copy_engine.py`: 5+2 iterations fine; 10+30 hard-reset the host; 25+100 hung 24 minutes at 0% GPU use.
- This branch was **never run above 2 ranks**, so it is not cleared either.
- Kernel log (`dmesg`): `ring sdmaX.Y timeout` on every SDMA ring of all 8 GPUs, failed ring resets, `device lost from
  bus!` on all 8, then `RAS Intr triggered, device disabled!!` 66 times. The first timeouts were on GPUs 4-6 (PCI `9f`,
  `af`, `bf`), then 7, with tiny sequence numbers, although the 4-rank job only used GPUs 0-3, so a stuck user SDMA
  queue does not explain them. A hardware/firmware fault or a hive-wide reset is possible. The start of the log was
  truncated, so the first event is unknown.
- The user reports the local branch `dev/dasidler/sdma-benchmark` (`531efd3b`) works. That branch has 210 commits that
  `d76a9d9c` lacks, pins a different tritonBLAS (`dev/dasidler/merge-main-into-muhaawad-iris`), and differs by about
  400 lines in `iris/mem/triton/ops.py`. **It exists only locally**; to use it for comparison, push it first:
  `git push origin 531efd3b:refs/heads/dev/dasidler/sdma-benchmark-pre-v3`.

## 4. Setup on the new machine

Needs at least 2 AMD GPUs with SDMA (8 for the full ladder), ROCm, and a torch/triton that matches (the original used
torch 2.9.1+rocm7.1 and the ROCm triton 3.6.0 build). Use a venv per code line so tritonBLAS pins never mix.

```bash
git clone git@github.com:ROCm/iris.git iris-new && cd iris-new
git checkout dev/dasidler/sdma-gemm-ag-tile-schedule
pip install -e ".[sdma]"            # tritonBLAS 426fad22 + rocm-xio 97fa1663
python -c "import tritonblas; print(hasattr(tritonblas, 'schedule_for_selector'))"   # True
python -c "from xio import sdma_ep"                                                  # no error

# Baseline in its own checkout and venv (it pins tritonBLAS b9baf7d3, which lacks the schedule API):
git worktree add ../iris-baseline origin/dev/dasidler/sdma-benchmark
cd ../iris-baseline && python -m venv --system-site-packages ../venv-base && . ../venv-base/bin/activate   # reuse the system torch/triton
pip install -e ".[sdma]"
```

Always set `HSA_NO_SCRATCH_RECLAIM=1` (the scripts do).

## 5. Steps, in this order

Safety: a hung run can wedge the driver and take the machine down. Never `kill -9` a hung GPU run (the scripts use
`timeout`, which sends SIGTERM). Keep logs on persistent storage (`/tmp` is wiped on reboot). Stop at the first failure.

1. **Correctness, safe and quick** (this branch):
   ```bash
   pytest tests/unittests/test_tritonblas_tile_schedule.py
   python tests/run_tests_distributed.py tests/ops/test_matmul_all_gather_copy_engine.py --num_ranks 2 -v
   ( cd <tritonBLAS checkout at 426fad22> && pytest tests/test_sync_config.py tests/test_schedule.py tests/test_gate_layout.py )
   ```
2. **Does the hang reproduce on this machine? Baseline first.** In the baseline checkout and venv:
   ```bash
   OUT=out_base benchmark/ops/stress/run_ladder.sh host      # ranks 2,4,8 x iterations 10,40,80,160
   OUT=out_base benchmark/ops/stress/run_ladder.sh device
   ```
   `benchmark/ops/stress/` exists only on this branch; copy the directory next to the baseline checkout, or run the
   scripts from this branch with `PYTHONPATH=<baseline checkout>` so the baseline iris is imported.
   Then the same two commands on this branch (`OUT=out_new`).
3. **Read the result** (`out_*/ladder_results.txt`; the last progress line per rank is printed on failure):

   | Baseline | This branch | Meaning |
   |---|---|---|
   | passes everything | passes everything | The hang was specific to the first machine; go to step 4. |
   | hangs (around 4 ranks, ~75 iterations?) | hangs at the same point | Existing issue in the SDMA path or hardware, independent of this change; report it. |
   | passes | hangs | This change. Compare gates: run with `EpilogueConfig(map_type="launch_wave", block_group_m=num_cus)` instead of the table to separate the table from the schedule. Check that the host plan and the in-kernel mapping agree (`test_epilogue_table_matches_launch_wave`). |
   | hangs | passes | Not expected; rerun both before concluding. |

4. **Benchmark** (only if step 2/3 are clean): start tiny, then raise.
   ```bash
   python benchmark/ops/bench_matmul_all_gather_copy_engine.py --engines sdma --n_warmup 2 --n_repeat 5
   python benchmark/ops/bench_matmul_all_gather_copy_engine.py --engines sdma --n_warmup 10 --n_repeat 30
   ```
   Alternate baseline and this branch at least three times and compare the `host` and `device` rows. Use a unique output
   file for every run. Compare against the baseline numbers in section 2.
5. **Hardware check on the original machine** (if it keeps failing there while the new machine is clean):
   ```bash
   benchmark/ops/stress/pair_sweep.sh host 150     # every GPU pair, 2 ranks; also saves rocm-smi RAS info
   ```
   Also look at `amd-smi metric -e` (ECC counters), the rasdaemon database, and `sudo ipmitool sel elist`, and keep
   `sudo journalctl -k -b 0 --since <time> --until <time>` for the failure window (the kernel ring buffer overflows).

## 6. Open items and decisions

- **Gates vs `EpilogueConfig`.** iris uses the table map type. Ryan's gate path is still in the tritonBLAS branch and
  unused; decide whether to drop it (his example and `tests/test_gate_layout.py` use it) and talk to him first. His
  64-byte gate layout (`gate_stride`/`gate_base`) maps to element offsets in the table; that needs the counter-buffer
  dtype relaxed from int32 to int64 (not done). `signal_times` tracing is not in `EpilogueConfig`.
- **tritonBLAS PR 102** (`wait_stores()` before the epilogue barrier): needed on Triton builds where
  `tl.debug_barrier()` is a bare `s_barrier`. The table path inherits it through `OutputView.store` once merged; Ryan's
  gate epilogue does not get it.
- **Table lookup cost**: one L2-resident int32 load per tile after the barrier; not measured. It could be hoisted before the K-loop.
- **Pin and PRs**: the tritonBLAS branch contains Ryan's commits. Open a tritonBLAS PR (or wait for his), then move the
  iris `pyproject.toml` pin to the merged commit. Neither branch has a PR.
- **HIP graph capture**: `flag_iteration` is a Python int baked into the kernel arguments, so replays would reuse one
  wait value (silent stale reads). Needs a device-resident generation counter (cf. `distributed_device_barrier`).
  Only the device-initiated path can be captured; the host-initiated path enqueues SDMA packets from the host.
- Not run by me: `ruff` on the files outside the ones I changed; the full unit-test suite.
