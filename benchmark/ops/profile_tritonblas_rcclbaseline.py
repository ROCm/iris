#!/usr/bin/env python3
"""Profile the TritonBLAS + RCCL matmul_all_reduce baseline.

Run with: ./benchmark/ops/profile_tritonblas_rcclbaseline.py
Then open /tmp/tritonblas_rcclbaseline_trace_rank0.json in chrome://tracing.
"""

import torch
import torch.distributed as dist
import tritonblas
import iris.bench as bench
from tritonblas.matmul import persistent_matmul_lt


# Single shape for profiling.
# Keep this in sync with profile_copy_engine.py when comparing traces.
# M, N, K = 16384, 7168, 7168
M, N, K = 16384, 16384, 53248
dtype = torch.float16


@bench.register
@bench.axis("num_ranks", [8])
def profile_tritonblas_rcclbaseline(state, ctx):
    """Profile TritonBLAS GEMM followed by RCCL all-reduce."""
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)

    selector = tritonblas.OrigamiMatmulSelector(
        M,
        N,
        K,
        A.dtype,
        B.dtype,
        C.dtype,
        A.device,
    )

    def run_once(*, annotate: bool = False):
        if annotate:
            with torch.profiler.record_function("tritonblas_gemm"):
                persistent_matmul_lt(A, B, C, selector, config=None, work_stealing=False)
            with torch.profiler.record_function("rccl_all_reduce"):
                dist.all_reduce(C, op=dist.ReduceOp.SUM)
            return

        persistent_matmul_lt(A, B, C, selector, config=None, work_stealing=False)
        dist.all_reduce(C, op=dist.ReduceOp.SUM)

    state.set_flops(2 * M * N * K)
    state.set_bytes((world_size - 1) * M * N * C.element_size())
    state.exec(run_once)

    if rank == 0:
        print(f"\n{'=' * 120}")
        print(f"Running TritonBLAS + RCCL profiler for {M}x{N}x{K}...")
        print(f"{'=' * 120}\n")

    for _ in range(5):
        run_once()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(10):
            run_once(annotate=True)
        torch.cuda.synchronize()

    if rank == 0:
        print("\n" + "=" * 120)
        print("TritonBLAS + RCCL CUDA Timeline (sorted by CUDA time)")
        print("=" * 120)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

        trace_path = "/tmp/tritonblas_rcclbaseline_trace_rank0.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to {trace_path}")
        print("Open in chrome://tracing to inspect GEMM vs RCCL kernels")
        print(f"{'=' * 120}\n")


if __name__ == "__main__":
    bench.main()
