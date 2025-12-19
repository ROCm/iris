# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import torch.cuda.nvtx as nvtx


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    m, n, k = 4864, 4096, 8256
    benchmark = True

    rank = 0
    A_full = torch.randn(m, k, device=f"cuda:{rank}")
    B_full = torch.randn(k, n, device=f"cuda:{rank}")

    def run_experiment():
        nvtx.range_push("GEMM Launch")
        result = A_full @ B_full
        nvtx.range_pop()
        return result

    C_global = run_experiment()

    if benchmark:
        nvtx.range_push("GEMM Benchmark")
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        ms = triton.testing.do_bench(run_experiment)
        print(f"Rank {rank}: {ms:.3f} ms  {perf(ms):.3f} TFLOPS")
        nvtx.range_pop()


if __name__ == "__main__":
    main()
