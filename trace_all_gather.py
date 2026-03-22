#!/usr/bin/env python3
"""
Run gluon all-gather with tracing enabled and analyze the traffic.
Run with: torchrun --nproc_per_node=4 trace_all_gather.py
"""

import os
import json
import torch
import torch.distributed as dist
import iris.experimental.iris_gluon as iris_gluon
from iris.ccl import Config
from iris.ccl.all_gather import all_gather

M, N = 8192, 8192
DTYPE = torch.float16
HEAP_SIZE = 2**33


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    shmem = iris_gluon.iris(HEAP_SIZE)

    # Enable tracing
    shmem.tracing.enable(max_events=2_000_000)

    inp = shmem.zeros((M, N), dtype=DTYPE)
    out = shmem.zeros((world_size * M, N), dtype=DTYPE)
    inp.fill_(float(rank + 1))

    config = Config(use_gluon=True, block_size_m=32, block_size_n=1024)

    # Warmup (without tracing reset — these events get recorded too)
    for _ in range(5):
        out.zero_()
        shmem.barrier()
        all_gather(out, inp, shmem, config=config)
        shmem.barrier()

    # Reset trace counters for clean capture
    shmem.tracing.reset()

    # Single traced run
    out.zero_()
    shmem.barrier()
    all_gather(out, inp, shmem, config=config)
    shmem.barrier()
    torch.cuda.synchronize()

    # Validate
    expected = torch.zeros(world_size * M, N, dtype=DTYPE, device=f"cuda:{rank}")
    for r in range(world_size):
        expected[r * M : (r + 1) * M, :] = float(r + 1)
    valid = torch.allclose(out, expected, atol=1e-3)

    if rank == 0:
        print(f"Correctness: {'PASS' if valid else 'FAIL'}")
        print(f"Events captured on rank 0: {shmem.tracing.trace_counter.item()}")

    # Export per-rank traces
    trace_dir = os.path.dirname(os.path.abspath(__file__))
    shmem.tracing.export(os.path.join(trace_dir, "trace_ag.json"), merge=False)
    shmem.barrier()

    # Rank 0 does analysis
    if rank == 0:
        analyze_traces(world_size)

    dist.destroy_process_group()


def analyze_traces(world_size):
    """Analyze trace data from all ranks."""
    import pandas as pd

    all_events = []
    for r in range(world_size):
        trace_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(trace_dir, f"trace_ag_rank{r}.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        events = [e for e in data["traceEvents"] if e.get("ph") not in ("M", None)]
        for e in events:
            e["source_rank"] = r
            if "args" in e:
                for k, v in e["args"].items():
                    e[k] = v
                del e["args"]
        all_events.extend(events)

    if not all_events:
        print("No events captured!")
        return

    df = pd.DataFrame(all_events)

    print(f"\n{'=' * 80}")
    print(f"TRACE ANALYSIS: gluon all-gather {M}x{N} fp16, {world_size} ranks")
    print(f"{'=' * 80}")

    print(f"\nTotal events: {len(df)}")
    print("\nEvents by type:")
    print(df.groupby("name").size().to_string())

    # Separate loads vs stores
    loads = df[df["name"] == "load"]
    stores = df[df["name"] == "store"]

    if len(loads) > 0:
        print("\n--- LOADS (local input reads) ---")
        print(f"  Count: {len(loads)}")
        print(f"  Avg duration (cycles): {loads['dur'].mean():.1f}")
        print(f"  Min duration (cycles): {loads['dur'].min()}")
        print(f"  Max duration (cycles): {loads['dur'].max()}")
        print(f"  P50 duration (cycles): {loads['dur'].quantile(0.5):.1f}")
        print(f"  P99 duration (cycles): {loads['dur'].quantile(0.99):.1f}")
        print(f"  Total payload (bytes): {loads['payload_size'].sum():,}")

    if len(stores) > 0:
        print("\n--- STORES (all output writes) ---")
        print(f"  Count: {len(stores)}")
        print(f"  Avg duration (cycles): {stores['dur'].mean():.1f}")
        print(f"  Min duration (cycles): {stores['dur'].min()}")
        print(f"  Max duration (cycles): {stores['dur'].max()}")
        print(f"  P50 duration (cycles): {stores['dur'].quantile(0.5):.1f}")
        print(f"  P99 duration (cycles): {stores['dur'].quantile(0.99):.1f}")
        print(f"  Total payload (bytes): {stores['payload_size'].sum():,}")

        # Break down stores by local vs remote
        local_stores = stores[stores["source_rank"] == stores["target_rank"]]
        remote_stores = stores[stores["source_rank"] != stores["target_rank"]]

        print(f"\n  Local stores:  {len(local_stores)} (avg {local_stores['dur'].mean():.1f} cycles)")
        print(f"  Remote stores: {len(remote_stores)} (avg {remote_stores['dur'].mean():.1f} cycles)")

        if len(remote_stores) > 0:
            print("\n  Remote stores by target rank:")
            for target_r in sorted(remote_stores["target_rank"].unique()):
                subset = remote_stores[remote_stores["target_rank"] == target_r]
                print(
                    f"    -> rank {target_r}: {len(subset)} events, "
                    f"avg {subset['dur'].mean():.1f} cycles, "
                    f"total {subset['payload_size'].sum():,} bytes"
                )

    # Per-rank breakdown
    print("\n--- PER-RANK SUMMARY ---")
    for r in range(world_size):
        rank_events = df[df["source_rank"] == r]
        rank_loads = rank_events[rank_events["name"] == "load"]
        rank_stores = rank_events[rank_events["name"] == "store"]
        rank_local = rank_stores[rank_stores["target_rank"] == r]
        rank_remote = rank_stores[rank_stores["target_rank"] != r]
        print(f"  Rank {r}: {len(rank_loads)} loads, {len(rank_local)} local stores, {len(rank_remote)} remote stores")

    # Timing analysis: sequential store pattern
    print("\n--- TIMING ANALYSIS ---")
    if len(stores) > 0:
        # For each source rank, look at the store ordering per row
        for r in range(min(world_size, 2)):  # Just show rank 0 and 1
            rank_stores = stores[stores["source_rank"] == r].sort_values("ts")
            if len(rank_stores) == 0:
                continue

            # Look at the first few rows to see the interleaving pattern
            # Group by op_index to see sequential operations
            first_ops = rank_stores.head(20)
            print(f"\n  Rank {r} first 20 store events (sorted by timestamp):")
            print(f"  {'op_idx':>7s} {'target':>6s} {'dur':>8s} {'ts':>12s} {'payload':>8s}")
            for _, row in first_ops.iterrows():
                target = int(row["target_rank"])
                target_label = "LOCAL" if target == r else f"-> r{target}"
                print(
                    f"  {int(row['op_index']):>7d} "
                    f"{target_label:>6s} "
                    f"{int(row['dur']):>8d} "
                    f"{int(row['ts']):>12d} "
                    f"{int(row['payload_size']):>8d}"
                )

    # Time spent in loads vs stores
    if len(loads) > 0 and len(stores) > 0:
        total_load_cycles = loads["dur"].sum()
        total_store_cycles = stores["dur"].sum()
        total_local_store_cycles = local_stores["dur"].sum() if len(local_stores) > 0 else 0
        total_remote_store_cycles = remote_stores["dur"].sum() if len(remote_stores) > 0 else 0

        print("\n--- CYCLE BREAKDOWN (all ranks combined) ---")
        print(f"  Load cycles:         {total_load_cycles:>15,}")
        print(f"  Local store cycles:  {total_local_store_cycles:>15,}")
        print(f"  Remote store cycles: {total_remote_store_cycles:>15,}")
        print(f"  Total store cycles:  {total_store_cycles:>15,}")
        pct_remote = total_remote_store_cycles / (total_load_cycles + total_store_cycles) * 100
        pct_load = total_load_cycles / (total_load_cycles + total_store_cycles) * 100
        pct_local = total_local_store_cycles / (total_load_cycles + total_store_cycles) * 100
        print(f"\n  Load:         {pct_load:5.1f}%")
        print(f"  Local store:  {pct_local:5.1f}%")
        print(f"  Remote store: {pct_remote:5.1f}%")


if __name__ == "__main__":
    main()
