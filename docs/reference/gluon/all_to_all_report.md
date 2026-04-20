# iris.x all_to_all: Triton vs Gluon – Performance & Assembly Report

```{note}
This document describes the Gluon port of `iris.x.all_to_all` and how to reproduce
the performance comparison and assembly analysis.  Actual numbers require AMD GPUs
(MI300X / MI350X / MI355X recommended) with ROCm 7.0+ and the matching Triton build.
```

## Overview

`iris.x.all_to_all` is a *tile-level* collective that lets a user-written kernel
perform an all-to-all exchange one tile at a time.  The primitive is provided in
two backends that produce identical results:

| Backend | Decorator | Context type | Remote-read API |
|---------|-----------|--------------|-----------------|
| Triton  | `@triton.jit` | `iris.DeviceContext` | `iris.load(ptr, cur_rank, src_rank, heap_bases, mask)` |
| Gluon   | `@gluon.jit`  | `IrisDeviceCtx`      | `ctx.load(ptr, src_rank, mask)` |

The Gluon implementation lives in `iris/x/all_to_all.py` (the `all_to_all_gluon`
function) and is exported from `iris.x` when Gluon is available.

---

## Semantic Equivalence

Both backends implement the same all-to-all algorithm:

- **Input** `(M, world_size × N_per_rank)`: each rank's chunk `[:, r*N:(r+1)*N]`
  holds the data to be sent to rank `r`.
- **Output** `(M, world_size × N_per_rank)`: after the operation,
  `output[:, r*N:(r+1)*N]` contains the data that rank `r` sent to the current rank.

Correctness is validated against PyTorch `dist.all_to_all` in the test file
`tests/x/test_all_to_all_gluon.py` across five shapes and three dtypes.

---

## Key Algorithmic Differences

### Loop structure

The **Triton** version iterates over only the source ranks that overlap with the
current output tile (`range(first_src_rank, last_src_rank + 1)`).  The loop bounds
are *runtime* values, which means the compiler cannot unroll the loop statically.

The **Gluon** version iterates over `range(world_size)` where `world_size` is a
`gl.constexpr`.  The compiler *fully unrolls* this loop, resolving each branch
(local vs. remote) at compile time.  This trades compile time for a potentially
shorter hot path when tiles are well-aligned with rank boundaries.

### Memory access pattern

Triton processes whole `BLOCK_SIZE_M × BLOCK_SIZE_N` tiles in a single vectorised
load/store using 2-D index tensors.

Gluon processes the tile *row by row* (inner `for i in range(BLOCK_SIZE_M)` loop,
also unrolled) with 1-D column index vectors and a
`gl.BlockedLayout([1], [64], [4], [0])` layout hint that maps 256 threads over
`BLOCK_SIZE_N` columns.  This matches the access pattern used by
`persistent_all_to_all_gluon` in `iris/ccl/all_to_all.py` and allows the Gluon
compiler to apply traffic-shaping optimisations.

### RMA call

| Operation | Triton | Gluon |
|-----------|--------|-------|
| Remote read | `iris.load(ptr, cur_rank, src_rank, heap_bases, mask)` | `ctx.load(ptr, src_rank, mask)` |
| Local read  | `tl.load(ptr + offsets, mask)` | `gl.load(ptr + offsets, mask)` |

`ctx.load` in Gluon internally calls `_translate()` which computes the pointer
offset from the heap base of the remote rank.  The Triton `iris.load` does the
same but requires the caller to explicitly pass `heap_bases`.

---

## Running the Benchmark

### Requirements

```bash
pip install matplotlib   # for scatter plot generation
```

### Validate both backends

```bash
cd benchmark/ccl/all_to_all
python benchmark_x.py -v -r 8 --datatype fp16
```

Expected output:

```
  [Triton]  M=  4096 N_per_rank=  256: validation PASS
  [Gluon]   M=  4096 N_per_rank=  256: validation PASS
```

### Sweep across many problem sizes

```bash
python benchmark_x.py -v -b --sweep -r 8 \
    --datatype fp16 \
    --output_file results.json
```

This runs 14 problem sizes (see the [Problem Size Sweep Grid](#problem-size-sweep-grid) below) and writes a JSON
file with per-size timing and bandwidth for both backends.

### Generate the scatter plot

```bash
python plot_x_all_to_all.py results.json --output scatter.png
```

The script also prints a plain-text comparison table to stdout.

### Dump generated assembly

```bash
python benchmark_x.py --dump_asm -m 4096 -n 256 -r 8
```

This writes two files:

```
triton_all_to_all_M4096_N256_bm64_bn256.asm
gluon_all_to_all_M4096_N256_bm64_bn256.asm
```

Use `diff` or a merge tool to compare them side by side.

---

## Problem Size Sweep Grid

The default sweep covers 14 `(M, N_per_rank)` configurations spanning small to
extra-large tensors:

| Category     | M     | N per rank | Total bytes per rank (8 GPUs, fp16) |
|-------------|-------|------------|--------------------------------------|
| Small        | 128   | 64         | ~112 KiB                            |
| Small        | 256   | 64         | ~224 KiB                            |
| Small        | 512   | 64         | ~448 KiB                            |
| Medium       | 1024  | 128        | ~1.75 MiB                           |
| Medium       | 2048  | 128        | ~3.5 MiB                            |
| Medium       | 1024  | 256        | ~3.5 MiB                            |
| Medium       | 2048  | 256        | ~7 MiB                              |
| Large        | 4096  | 128        | ~7 MiB                              |
| Large        | 4096  | 256        | ~14 MiB                             |
| Large        | 4096  | 512        | ~28 MiB                             |
| Large        | 8192  | 256        | ~28 MiB                             |
| Large        | 8192  | 512        | ~56 MiB                             |
| Extra-large  | 16384 | 128        | ~28 MiB                             |
| Extra-large  | 16384 | 256        | ~56 MiB                             |

---

## Performance Results

> **Placeholder** – Run the benchmark on MI300X hardware and paste the JSON
> output here, or embed the scatter plot image.

```
python benchmark_x.py -v -b --sweep -r 8 --datatype fp16 --output_file results.json
python plot_x_all_to_all.py results.json
```

After running you will have a scatter plot similar to:

```
                iris.x all_to_all: Triton vs Gluon bandwidth
                ─────────────────────────────────────────────
  Bandwidth  │          ◉ Triton     ■ Gluon
   (GB/s)    │  ■ ◉               ◉
             │        ■        ■ ◉
             │    ■ ◉       ◉
             │ ◉■       ■
             │─────────────────────────────────────────────
                    Total bytes per rank (log₂ scale)
```

---

## Assembly Analysis

The AMDGCN ISA files produced by `--dump_asm` let you compare the quality of code
generated by the two compilation paths.

### What to look for

| Metric | Description |
|--------|-------------|
| VGPR count | Fewer VGPRs → more warps can be in flight simultaneously (higher occupancy) |
| Spill count | Non-zero spills hurt performance through stack memory traffic |
| Load/store width | `global_load_dwordx4` / `global_store_dwordx4` instructions indicate 128-bit vectorisation |
| Buffer instructions | `buffer_load_dwordx4` / `buffer_store_dwordx4` indicate coalesced cache-line access |
| Branch instructions | Fewer branches → simpler control flow after loop unrolling |

### Expected observations (Gluon)

- The fully-unrolled rank loop in the Gluon version eliminates loop-overhead
  branches visible in the Triton output.
- Because `world_size` is a `gl.constexpr`, each `src_rank == cur_rank` branch
  is resolved at compile time, resulting in separate code paths without runtime
  predication.
- The row-by-row processing in Gluon may show higher VGPR usage compared to the
  2-D tile processing in Triton, depending on the block sizes chosen.

### Sample diff command

```bash
diff -u triton_all_to_all_M4096_N256_bm64_bn256.asm \
        gluon_all_to_all_M4096_N256_bm64_bn256.asm | less
```

---

## Running the Tests

The functional correctness tests for the Gluon backend are in
`tests/x/test_all_to_all_gluon.py`:

```bash
pytest tests/x/test_all_to_all_gluon.py -v
```

Tests are automatically skipped when Gluon is not available.

---

## Source Files

| File | Description |
|------|-------------|
| `iris/x/all_to_all.py` | Triton (`all_to_all`) and Gluon (`all_to_all_gluon`) tile-level primitives |
| `iris/x/__init__.py` | Module exports – `all_to_all_gluon` exported when Gluon available |
| `tests/x/test_all_to_all_gluon.py` | Correctness tests vs. PyTorch `dist.all_to_all` |
| `benchmark/ccl/all_to_all/benchmark_x.py` | Validation + benchmark sweep + assembly dump |
| `benchmark/ccl/all_to_all/plot_x_all_to_all.py` | Scatter-plot generation from JSON results |
