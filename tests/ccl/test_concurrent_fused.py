# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness tests for the fused work-stealing GEMM+collective kernels
(``iris.concurrent.gemm``). Each op runs an INDEPENDENT GEMM (C = A @ B)
concurrently with a collective on ``comm_src`` via one persistent dual-queue
kernel (``mode="fused"``) or two work-stealing kernels (``mode="concurrent"``).

We validate BOTH halves against references:
  * GEMM   : C ~= A @ B  (torch.matmul)
  * comm   : comm_dst ~= the matching torch.distributed collective on comm_src

Requires a distributed context (run via tests/run_tests_distributed.py).
"""

import gc

import pytest
import torch
import torch.distributed as dist

import iris
import iris.concurrent.gemm as cg

GB = (256, 256, 64)  # GEMM tile
CB = (256, 64)  # comm tile


def _skip_if_no_dist():
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")


def _gemm_rel_err(C, A, B):
    ref = torch.matmul(A.float(), B.float())
    return (C.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


@pytest.mark.parametrize("mode", ["fused", "concurrent"])
@pytest.mark.parametrize(
    "collective",
    ["all_gather", "all_reduce", "reduce_scatter", "all_to_all", "broadcast"],
)
@pytest.mark.parametrize("M, N, K", [(1024, 512, 1024), (2048, 1024, 2048)])
def test_fused_gemm_collective(mode, collective, M, N, K):
    _skip_if_no_dist()
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    W = shmem.get_num_ranks()
    cu = shmem.get_cu_count()
    dtype = torch.float16
    gw = max(64, cu - 64)  # a non-degenerate CU split

    # comm operands must tile cleanly and (RS/A2A) divide by world size
    cm = 256 * W
    cn = 512
    try:
        A = shmem.randn(M, K, device="cuda", dtype=dtype)
        B = shmem.randn(N, K, device="cuda", dtype=dtype).T
        C = shmem.zeros((M, N), device="cuda", dtype=dtype)
        src = shmem.full((cm, cn), float(rank + 1), device="cuda", dtype=dtype)

        # collective-specific dst + torch reference on a clone of src
        ref_src = torch.full((cm, cn), float(rank + 1), device=f"cuda:{rank}", dtype=dtype)
        if collective == "all_gather":
            dst = shmem.zeros((W * cm, cn), device="cuda", dtype=dtype)
            ref = torch.empty((W * cm, cn), device=f"cuda:{rank}", dtype=dtype)
            dist.all_gather_into_tensor(ref, ref_src)
        elif collective == "all_reduce":
            dst = shmem.zeros((cm, cn), device="cuda", dtype=dtype)
            ref = ref_src.clone()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        elif collective == "reduce_scatter":
            dst = shmem.zeros((cm // W, cn), device="cuda", dtype=dtype)
            ref = torch.empty((cm // W, cn), device=f"cuda:{rank}", dtype=dtype)
            dist.reduce_scatter_tensor(ref, ref_src, op=dist.ReduceOp.SUM)
        elif collective == "all_to_all":
            dst = shmem.zeros((cm, cn), device="cuda", dtype=dtype)
            ref = torch.empty((cm, cn), device=f"cuda:{rank}", dtype=dtype)
            dist.all_to_all_single(ref, ref_src)
        else:  # broadcast (root 0)
            dst = shmem.zeros((cm, cn), device="cuda", dtype=dtype)
            ref = ref_src.clone()
            dist.broadcast(ref, 0)
        torch.cuda.synchronize()

        shmem.barrier()
        getattr(cg, collective)(
            shmem, A, B, src, C=C, comm_dst=dst, mode=mode, num_wgs=cu, gemm_wgs=gw, gemm_block=GB, comm_block=CB
        )
        torch.cuda.synchronize()
        shmem.barrier()

        # GEMM half
        rel = _gemm_rel_err(C, A, B)
        assert rel < 0.05, f"[{collective}/{mode}] GEMM rel err {rel:.3f} too high"

        # comm half (copies are exact; fp32-accumulated reduces get a small tol)
        atol = 1e-2 if collective in ("all_reduce", "reduce_scatter") else 1e-3
        max_diff = (dst.float() - ref.float()).abs().max().item()
        assert torch.allclose(dst, ref, atol=atol), f"[{collective}/{mode}] comm mismatch: max_diff={max_diff}"
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("collective", ["all_gather", "all_reduce"])
def test_fused_matches_concurrent(collective):
    """fused and concurrent modes must produce identical results for the same op."""
    _skip_if_no_dist()
    shmem = iris.iris(1 << 33)
    rank = shmem.get_rank()
    W = shmem.get_num_ranks()
    cu = shmem.get_cu_count()
    dtype = torch.float16
    M, N, K = 1024, 512, 1024
    cm, cn = 256 * W, 512
    try:
        A = shmem.randn(M, K, device="cuda", dtype=dtype)
        B = shmem.randn(N, K, device="cuda", dtype=dtype).T
        src = shmem.full((cm, cn), float(rank + 1), device="cuda", dtype=dtype)
        dm = W * cm if collective == "all_gather" else cm
        C1 = shmem.zeros((M, N), device="cuda", dtype=dtype)
        C2 = shmem.zeros((M, N), device="cuda", dtype=dtype)
        d1 = shmem.zeros((dm, cn), device="cuda", dtype=dtype)
        d2 = shmem.zeros((dm, cn), device="cuda", dtype=dtype)
        shmem.barrier()
        getattr(cg, collective)(
            shmem,
            A,
            B,
            src,
            C=C1,
            comm_dst=d1,
            mode="fused",
            num_wgs=cu,
            gemm_wgs=cu - 64,
            gemm_block=GB,
            comm_block=CB,
        )
        getattr(cg, collective)(
            shmem,
            A,
            B,
            src,
            C=C2,
            comm_dst=d2,
            mode="concurrent",
            num_wgs=cu,
            gemm_wgs=cu - 64,
            gemm_block=GB,
            comm_block=CB,
        )
        torch.cuda.synchronize()
        shmem.barrier()
        assert torch.allclose(d1, d2, atol=1e-3), f"[{collective}] fused vs concurrent comm differ"
        assert torch.allclose(C1, C2, atol=1e-2), f"[{collective}] fused vs concurrent GEMM differ"
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
