#!/usr/bin/env python3
"""
Mini MLP layer benchmark: compute + communication.

Simulates what a single transformer MLP block looks like in vLLM inference
with tensor parallelism. Each iteration does:

    1. gate_proj  (ColumnParallel): (M, hidden) × (hidden, intermediate/TP) → (M, intermediate/TP)
    2. up_proj    (ColumnParallel): (M, hidden) × (hidden, intermediate/TP) → (M, intermediate/TP)
    3. SiLU(gate) * up  (activation)
    4. down_proj  (RowParallel):    (M, intermediate/TP) × (intermediate/TP, hidden) → (M, hidden)
    5. all_reduce (sum partial results across TP ranks)

We time three things:
    - compute only (steps 1-4, no all-reduce)
    - comm only (step 5 only)
    - full layer (steps 1-5)

Usage:
    torchrun --nproc_per_node=8 benchmark/ccl/bench_mlp_layer.py
"""
import os
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(backend="nccl")

import iris
from iris.ccl import Config

rank = dist.get_rank()
world_size = dist.get_world_size()

# ── Model configs ──────────────────────────────────────────────────────────
MODELS = {
    "LLaMA-7B":   {"hidden": 4096,  "intermediate": 11008},
    "LLaMA-70B":  {"hidden": 8192,  "intermediate": 28672},
    "LLaMA-405B": {"hidden": 16384, "intermediate": 53248},
}

# Token counts: M = batch_size * seq_len
TOKEN_COUNTS = [1, 32, 128, 512, 2048]

dtype = torch.bfloat16
element_size = 2


# ── Helpers ────────────────────────────────────────────────────────────────
def bench_cuda_events(fn, warmup=10, rep=50):
    """Time a function using CUDA events, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        dist.barrier()
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


# ── MLP layer ─────────────────────────────────────────────────────────────
class MLPLayer:
    """Simulates a single LLaMA MLP block with tensor parallelism."""

    def __init__(self, hidden, intermediate, tp_size):
        self.hidden = hidden
        self.intermediate = intermediate
        self.tp_size = tp_size
        self.inter_per_rank = intermediate // tp_size

        # Weights (Column-parallel: gate_proj, up_proj)
        self.w_gate = torch.randn(
            hidden, self.inter_per_rank, dtype=dtype, device="cuda"
        ) * 0.01
        self.w_up = torch.randn(
            hidden, self.inter_per_rank, dtype=dtype, device="cuda"
        ) * 0.01

        # Weight (Row-parallel: down_proj)
        self.w_down = torch.randn(
            self.inter_per_rank, hidden, dtype=dtype, device="cuda"
        ) * 0.01

    def compute(self, x):
        """Steps 1-4: matmuls + activation. Returns partial sum (M, hidden)."""
        gate = x @ self.w_gate              # (M, inter/TP)
        up = x @ self.w_up                  # (M, inter/TP)
        act = torch.nn.functional.silu(gate) * up  # (M, inter/TP)
        out = act @ self.w_down             # (M, hidden) — partial sum
        return out


# ── Main ───────────────────────────────────────────────────────────────────
heap_size = 2**33
# Use vmem allocator to enable as_symmetric (zero-copy import)
try:
    ctx = iris.iris(heap_size, allocator_type="vmem")
    has_as_symmetric = hasattr(ctx, "as_symmetric")
except Exception:
    ctx = iris.iris(heap_size)
    has_as_symmetric = False


def run_model(model_name, cfg):
    hidden = cfg["hidden"]
    intermediate = cfg["intermediate"]
    mlp = MLPLayer(hidden, intermediate, world_size)

    if rank == 0:
        inter_per_rank = intermediate // world_size
        print(f"\n{'='*90}")
        print(f"  {model_name}  (hidden={hidden}, intermediate={intermediate}, TP={world_size})")
        print(f"  Per-rank matmul shapes: ({hidden}→{inter_per_rank}) and ({inter_per_rank}→{hidden})")
        print(f"  All-reduce shape: (M, {hidden}), dtype={dtype}")
        if has_as_symmetric:
            print(f"  as_symmetric: AVAILABLE (zero-copy import)")
        else:
            print(f"  as_symmetric: not available (will use copy)")
        print(f"{'='*90}")
        hdr = (f"{'M':>6} | {'Compute':>10} {'Comm(RCCL)':>11} {'Full(RCCL)':>11} {'Comm%':>6} | "
               f"{'iris+copy':>11} {'Full+copy':>11} {'Comm%':>6} {'vs RCCL':>8} |")
        if has_as_symmetric:
            hdr += (f" {'iris(0cp)':>11} {'Full(0cp)':>11} {'Comm%':>6} {'vs RCCL':>8} |")
        print(hdr)
        print("-" * (len(hdr)))

    for M in TOKEN_COUNTS:
        # Input tensor
        x = torch.randn(M, hidden, dtype=dtype, device="cuda") * 0.01

        # ── Compute only ──
        compute_ms = bench_cuda_events(lambda: mlp.compute(x), warmup=10, rep=50)

        # ── RCCL comm only ──
        partial = mlp.compute(x)
        rccl_tensor = partial.clone()

        def rccl_comm():
            rccl_tensor.copy_(partial)
            dist.all_reduce(rccl_tensor, op=dist.ReduceOp.SUM)

        rccl_comm_ms = bench_cuda_events(rccl_comm, warmup=10, rep=50)

        # ── RCCL full layer ──
        def rccl_full():
            out = mlp.compute(x)
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out

        rccl_full_ms = bench_cuda_events(rccl_full, warmup=10, rep=50)

        # ── iris with copy (baseline) ──
        iris_inp = ctx.zeros((M, hidden), dtype=dtype)
        iris_out = ctx.zeros((M, hidden), dtype=dtype)
        iris_inp.copy_(partial)
        config = Config(all_reduce_variant="flat")
        workspace = ctx.ccl.all_reduce_preamble(iris_out, iris_inp, config=config)
        ctx.barrier()

        def iris_comm_copy():
            iris_inp.copy_(partial)
            ctx.ccl.all_reduce(iris_out, iris_inp, config=config, workspace=workspace)

        iris_copy_comm_ms = bench_cuda_events(iris_comm_copy, warmup=10, rep=50)

        def iris_full_copy():
            out = mlp.compute(x)
            iris_inp.copy_(out)
            ctx.ccl.all_reduce(iris_out, iris_inp, config=config, workspace=workspace)

        iris_copy_full_ms = bench_cuda_events(iris_full_copy, warmup=10, rep=50)

        # ── iris with as_symmetric (zero-copy) ──
        iris_zc_comm_ms = None
        iris_zc_full_ms = None

        if has_as_symmetric:
            # as_symmetric maps the matmul output into the heap — no copy needed.
            # We need to set up workspace with a symmetric tensor of the right shape.
            sym_partial = ctx.as_symmetric(partial)
            iris_zc_out = ctx.zeros((M, hidden), dtype=dtype)
            config_zc = Config(all_reduce_variant="flat")
            workspace_zc = ctx.ccl.all_reduce_preamble(
                iris_zc_out, sym_partial, config=config_zc
            )
            ctx.barrier()

            def iris_comm_zc():
                # partial is already in heap via as_symmetric — just all_reduce
                ctx.ccl.all_reduce(
                    iris_zc_out, sym_partial, config=config_zc, workspace=workspace_zc
                )

            iris_zc_comm_ms = bench_cuda_events(iris_comm_zc, warmup=10, rep=50)

            def iris_full_zc():
                # Compute directly into the pre-registered buffer
                out = mlp.compute(x)
                # as_symmetric was called once during setup — `partial` and
                # `sym_partial` share storage. We just need to get the matmul
                # output into that same buffer.
                partial.copy_(out)
                ctx.ccl.all_reduce(
                    iris_zc_out, sym_partial, config=config_zc, workspace=workspace_zc
                )

            iris_zc_full_ms = bench_cuda_events(iris_full_zc, warmup=10, rep=50)

        # ── Print ──
        rccl_comm_pct = rccl_comm_ms / rccl_full_ms * 100 if rccl_full_ms > 0 else 0
        iris_copy_comm_pct = iris_copy_comm_ms / iris_copy_full_ms * 100 if iris_copy_full_ms > 0 else 0
        copy_speedup = rccl_full_ms / iris_copy_full_ms if iris_copy_full_ms > 0 else 0

        line = (
            f"{M:>6} | "
            f"{compute_ms:>9.3f}ms "
            f"{rccl_comm_ms:>9.3f}ms "
            f"{rccl_full_ms:>9.3f}ms "
            f"{rccl_comm_pct:>5.1f}% | "
            f"{iris_copy_comm_ms:>9.3f}ms "
            f"{iris_copy_full_ms:>9.3f}ms "
            f"{iris_copy_comm_pct:>5.1f}% "
            f"{copy_speedup:>7.2f}x |"
        )

        if has_as_symmetric and iris_zc_full_ms is not None:
            iris_zc_comm_pct = iris_zc_comm_ms / iris_zc_full_ms * 100 if iris_zc_full_ms > 0 else 0
            zc_speedup = rccl_full_ms / iris_zc_full_ms if iris_zc_full_ms > 0 else 0
            line += (
                f" {iris_zc_comm_ms:>9.3f}ms "
                f"{iris_zc_full_ms:>9.3f}ms "
                f"{iris_zc_comm_pct:>5.1f}% "
                f"{zc_speedup:>7.2f}x |"
            )

        if rank == 0:
            print(line, flush=True)


# ── Pre-compile ────────────────────────────────────────────────────────────
if rank == 0:
    print("Pre-compiling kernels...", flush=True)

# Warm up iris kernels for all sizes we'll use
for cfg in MODELS.values():
    for M in TOKEN_COUNTS:
        inp = ctx.zeros((M, cfg["hidden"]), dtype=dtype)
        out = ctx.zeros((M, cfg["hidden"]), dtype=dtype)
        config = Config(all_reduce_variant="flat")
        ws = ctx.ccl.all_reduce_preamble(out, inp, config=config)
        ctx.ccl.all_reduce(out, inp, config=config, workspace=ws)

        # Also warm up with as_symmetric path if available
        if has_as_symmetric:
            ext = torch.zeros(M, cfg["hidden"], dtype=dtype, device="cuda")
            sym = ctx.as_symmetric(ext)
            ws2 = ctx.ccl.all_reduce_preamble(out, sym, config=config)
            ctx.ccl.all_reduce(out, sym, config=config, workspace=ws2)

torch.cuda.synchronize()
dist.barrier()

if rank == 0:
    print("Done.\n", flush=True)

# ── Run ────────────────────────────────────────────────────────────────────
for model_name, cfg in MODELS.items():
    run_model(model_name, cfg)

if rank == 0:
    print()

dist.destroy_process_group()
