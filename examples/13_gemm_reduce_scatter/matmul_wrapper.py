# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import random
import sys
import os

from gemm__reduce_scatter import persistent_gemm_reduce_scatter

from examples.common.utils import is_triton_interpret_set

gemm_kernel = persistent_gemm_reduce_scatter


class matmul_reduce_scatter(torch.autograd.Function):
    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul_reduce_scatter._debug = debug

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_local: torch.Tensor,  # 修改：本地输出而不是全局输出
        bias: torch.Tensor,
        P: torch.Tensor,
        locks: torch.Tensor,
        tile_completed: torch.Tensor,
        rank: int,
        world_size: int,
        total_programs_streamk: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        two_tiles: bool,
        num_stages: int,
        num_warps: int,
        waves_per_eu: int,
        mfmaInstrSize: int,
        kpack: int,
        heap_bases_ptr: torch.Tensor = None,
        cu_count: int = 304,
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        # 关键修改：计算每个rank负责的输出分区大小
        rows_per_rank = (M + world_size - 1) // world_size
        local_M = rows_per_rank
        if rank == world_size - 1:  # 最后一个rank处理剩余的行
            local_M = M - rank * rows_per_rank
        
        # 验证本地输出缓冲区大小是否正确
        assert c_local.shape[0] == local_M, f"c_local shape mismatch: expected {local_M}, got {c_local.shape[0]}"
        assert c_local.shape[1] == N, f"c_local shape mismatch: expected {N}, got {c_local.shape[1]}"

        num_xcds = 1
        if cu_count == 304:
            num_xcds = 8

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0

        if total_programs_streamk > 0:
            total_tiles_streamk = total_tiles % total_programs_streamk
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
        else:
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if matmul_reduce_scatter._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"Rank {rank}/{world_size} responsible for {local_M} rows")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")

        use_bias = False
        stride_bias = bias.stride(0) if use_bias else 0

        # 关键修改：使用ReduceScatter内核
        grids = total_programs_streamk
        kk = gemm_kernel[(grids,)](
            a,
            b,
            c,
            c_local,  # 修改：传递本地输出缓冲区
            bias,
            P,
            locks,
            tile_completed,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            c_local.stride(0),  # 修改：本地输出的stride
            c_local.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs_streamk,
            STREAMK_TILES=total_tiles_streamk,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
            EVEN_K=even_k,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfmaInstrSize,
            kpack=kpack,
            heap_bases=heap_bases_ptr,
            cur_rank=rank,
            world_size=world_size,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS,
            mm_begin_timestamp_ptr=mm_begin_timestamp,
            mm_end_timestamp_ptr=mm_end_timestamp,
        )

        if matmul_reduce_scatter._debug and not is_triton_interpret_set():
            matmul_reduce_scatter.streamk_registers = kk.n_regs
            matmul_reduce_scatter.streamk_spills = kk.n_spills
            print(f"{kk.n_regs} registers used, {kk.n_spills} spills")

        return c_local  # 修改：返回本地输出

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_local: torch.Tensor,  # 修改：本地输出缓冲区
        bias: torch.Tensor,
        P: torch.Tensor,
        locks: torch.Tensor,
        tile_completed: torch.Tensor,
        rank: int,
        world_size: int,
        grid: int,
        BLK_M=128,
        BLK_N=128,
        BLK_K=32,
        gsize_m=1,
        two_tiles=True,
        num_stages=3,
        num_warps=4,
        waves_per_eu=2,
        mfmaInstrSize=16,
        kpack=1,
        heap_bases_ptr: torch.Tensor = None,
        cu_count: int = 304,
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        result = matmul_reduce_scatter._call(
            a=a,
            b=b,
            c=c,
            c_local=c_local,  # 修改：传递本地输出
            bias=bias,
            P=P,
            locks=locks,
            tile_completed=tile_completed,
            rank=rank,
            world_size=world_size,
            total_programs_streamk=grid,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            gsize_m=gsize_m,
            two_tiles=two_tiles,
            num_warps=num_warps,
            num_stages=num_stages,
            waves_per_eu=waves_per_eu,
            mfmaInstrSize=mfmaInstrSize,
            kpack=kpack,
            heap_bases_ptr=heap_bases_ptr,
            cu_count=cu_count,
            COLLECT_TIMESTAMPS=COLLECT_TIMESTAMPS,
            mm_begin_timestamp=mm_begin_timestamp,
            mm_end_timestamp=mm_end_timestamp,
        )
        return result  # 修改：返回本地输出结果


# 使用示例函数
# def gemm_reduce_scatter_example():
#     # 初始化参数
#     M, N, K = 1024, 1024, 1024
#     world_size = 4
#     rank = 0  # 在实际中需要根据当前rank设置
    
#     # 计算每个rank负责的行数
#     rows_per_rank = (M + world_size - 1) // world_size
#     local_M = rows_per_rank
#     if rank == world_size - 1:
#         local_M = M - rank * rows_per_rank
    
#     # 创建输入张量
#     a = torch.randn(M, K, device='cuda')
#     b = torch.randn(K, N, device='cuda')
    
#     # 创建中间缓冲区和输出缓冲区
#     c = torch.zeros(M, N, device='cuda')  # 中间结果缓冲区
#     c_local = torch.zeros(local_M, N, device='cuda')  # 本地输出缓冲区
    
#     # 创建同步所需的张量
#     total_tiles = (M + 127) // 128 * (N + 127) // 128
#     tile_completed = torch.zeros(total_tiles, dtype=torch.int32, device='cuda')
#     locks = torch.zeros(1024, dtype=torch.int32, device='cuda')  # 锁缓冲区
#     P = torch.zeros(1, dtype=torch.int32, device='cuda')  # 占位符
    
#     # 调用ReduceScatter GEMM
#     result = matmul_reduce_scatter.apply(
#         a, b, c, c_local, None, P, locks, tile_completed,
#         rank, world_size, 256,  # grid size = 256
#         128, 128, 32  # BLK_M, BLK_N, BLK_K
#     )
    
#     return result


# if __name__ == "__main__":
#     # 测试代码
#     result = gemm_reduce_scatter_example()
#     print(f"ReduceScatter result shape: {result.shape}")