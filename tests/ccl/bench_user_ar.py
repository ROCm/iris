#!/usr/bin/env python3
"""Benchmark user-facing iris allreduce vs RCCL.

Usage:
    torchrun --nproc_per_node=8 tests/ccl/bench_user_ar.py
"""

import torch
import torch.distributed as dist


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def bench(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    rank, world_size = setup()
    dtype = torch.bfloat16

    import iris
    from iris.ccl.all_reduce import all_reduce

    ctx = iris.iris(heap_size=1 << 30)

    sizes = [
        (1, 1024),  # 1K
        (4, 1024),  # 4K
        (16, 1024),  # 16K
        (64, 1024),  # 64K
        (128, 1024),  # 128K
        (256, 1024),  # 256K
    ]

    if rank == 0:
        print(f"{'Shape':>12} {'Elements':>10} {'RCCL (us)':>12} {'iris (us)':>12} {'speedup':>10}")

    for shape in sizes:
        numel = shape[0] * shape[1]
        user_tensor = torch.randn(shape, dtype=dtype, device="cuda")
        rccl_tensor = torch.randn(shape, dtype=dtype, device="cuda")

        # RCCL in-place
        rccl_us = bench(lambda: dist.all_reduce(rccl_tensor))

        # iris (copy_in baked into all_reduce + two_shot async)
        out_buf = torch.empty(shape, dtype=dtype, device="cuda")
        iris_us = bench(lambda: all_reduce(out_buf, user_tensor, ctx, async_op=True))

        speedup = rccl_us / iris_us if iris_us > 0 else 0

        if rank == 0:
            print(f"{str(shape):>12} {numel:>10} {rccl_us:>12.1f} {iris_us:>12.1f} {speedup:>9.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
