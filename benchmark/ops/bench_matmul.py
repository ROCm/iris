#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmarks for standalone GEMM."""

import torch
import iris.bench as bench

from iris.host.tracing.kernel_artifacts import iris_launch
from iris.ops.matmul import matmul as _matmul
from iris.ops.matmul import matmul_preamble as _matmul_preamble
from iris.ops.matmul_all_reduce_copy_engine import (
    _matmul_all_reduce_copy_engine_launch_params,
    _partitioned_xcd_gemm_num_sms,
    _partitioned_xcd_matmul_kernel,
)
from tritonblas.matmul import _make_matmul_selector


def _register_local_matmul(state, ctx, *, m_key: str, pytorch: bool, work_stealing: bool = False, enable_streamk: bool = False) -> None:
    M, N, K = state[m_key], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()

    state.set_flops(2 * M * N * K)
    state.set_bytes(((M * K) + (K * N) + (M * N)) * torch.tensor([], dtype=dtype).element_size())

    torch.manual_seed(123 + rank)
    A_data = torch.randn((M, K), device="cuda", dtype=dtype)
    torch.manual_seed(456)
    B_data = torch.randn((K, N), device="cuda", dtype=dtype)

    if pytorch:
        C_torch = torch.empty((M, N), device="cuda", dtype=dtype)
        state.exec(lambda: torch.mm(A_data, B_data, out=C_torch))
    else:
        # Use iris heap allocation (same as tritonblas_rcclbaseline)
        torch.manual_seed(123 + rank)
        A = ctx.randn((M, K), dtype=dtype)
        torch.manual_seed(456)
        B = ctx.randn((K, N), dtype=dtype)
        C = ctx.zeros((M, N), dtype=dtype)

        workspace = _matmul_preamble(ctx, A, B)
        # Using async_op=True to match torch
        state.exec(
            lambda: _matmul(ctx, C, A, B, workspace=workspace, async_op=True, work_stealing=work_stealing, enable_streamk=enable_streamk),
        )


def _make_copy_engine_selector(M: int, N: int, K: int, dtype: torch.dtype, device: torch.device):
    return _make_matmul_selector(
        M,
        N,
        K,
        dtype,
        dtype,
        dtype,
        device,
        streamk=False,
    )


def _register_copy_engine_partitioned_gemm(state, ctx) -> None:
    M, N, K = state["M"], state["N"], state["K"]
    dtype = state["dtype"]
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    state.set_flops(2 * M * N * K)
    state.set_bytes(((M * K) + (K * N) + (M * N)) * torch.tensor([], dtype=dtype).element_size())

    torch.manual_seed(123 + rank)
    A = ctx.randn((M, K), dtype=dtype)
    torch.manual_seed(456)
    B = ctx.randn((K, N), dtype=dtype)
    C = ctx.zeros((M, N), dtype=dtype)
    locks = ctx.zeros((1,), dtype=torch.int32)

    selector = _make_copy_engine_selector(M, N, K, dtype, device)
    launch = _matmul_all_reduce_copy_engine_launch_params(M, N, selector, device)
    gemm_num_sms = _partitioned_xcd_gemm_num_sms(launch, world_size)
    launch["gemm_num_sms"] = gemm_num_sms

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    even_k = K % launch["block_size_k"] == 0
    launch_kwargs = {
        "num_warps": launch["num_warps"],
        "matrix_instr_nonkdim": launch["matrix_instr_nonkdim"],
    }
    if launch["num_stages"] is not None:
        launch_kwargs["num_stages"] = launch["num_stages"]

    def _run_copy_engine_partitioned_gemm():
        iris_launch(
            _partitioned_xcd_matmul_kernel,
            (gemm_num_sms,),
            A,
            B,
            C,
            C,
            locks,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_cm,
            stride_cn,
            rank,
            launch["block_size_m"],
            launch["block_size_n"],
            launch["block_size_k"],
            launch["group_size_m"],
            gemm_num_sms,
            launch["num_xcds"],
            launch["chunk_size"],
            world_size,
            1,
            False,
            False,
            even_k,
            launch["allow_tf32"],
            algorithm="matmul_copy_engine_partitioned_gemm",
            rank=rank,
            dtype=A.dtype,
            **launch_kwargs,
        )

    state.exec(_run_copy_engine_partitioned_gemm)

    state.add_counter("block_m", launch["block_size_m"])
    state.add_counter("block_n", launch["block_size_n"])
    state.add_counter("block_k", launch["block_size_k"])
    state.add_counter("group_size_m", launch["group_size_m"])
    state.add_counter("num_xcds", launch["num_xcds"])
    state.add_counter("chunk_size", launch["chunk_size"])
    state.add_counter("grid_size", launch["num_sms"])
    state.add_counter("gemm_num_sms", gemm_num_sms)
    state.add_counter("total_tiles", launch["total_tiles"])
    state.add_counter("num_stages", launch["num_stages"] or 0)
    state.add_counter("store_local_reduce_shard", 0)
    state.add_counter("signal_locks", 0)
    state.add_counter("selector_fallback", 0)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_only_local(state, ctx):
    _register_local_matmul(state, ctx, m_key="M_local", pytorch=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M_local", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_only_local(state, ctx):
    _register_local_matmul(state, ctx, m_key="M_local", pytorch=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_only(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, work_stealing=False)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_work_stealing(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, work_stealing=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_streamk(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=False, enable_streamk=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def pytorch_matmul_only(state, ctx):
    _register_local_matmul(state, ctx, m_key="M", pytorch=True)


@bench.register
@bench.axis("num_ranks", [8])
@bench.axis("M", [1024, 4096, 16384])
@bench.axis("N", [3584])
@bench.axis("K", [8192])
@bench.axis("dtype", [torch.float16])
def matmul_copy_engine_partitioned_gemm(state, ctx):
    _register_copy_engine_partitioned_gemm(state, ctx)


if __name__ == "__main__":
    bench.main()
