# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

# from streamk_kernel import streamk_gemm
# from streamk_kernel_atomic import streamk_gemm
from gemm_one_shot_all_reduce import persistent_gemm_all_reduce

from examples.common.utils import is_triton_interpret_set
import iris

gemm_kernel = persistent_gemm_all_reduce


class matmul(torch.autograd.Function):
    _debug = True

    _num_xcds = iris.hip.get_num_xcc()

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug
        matmul.streamk_registers = 0
        matmul.streamk_spills = 0

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
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
        cu_count: int = None,
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        #        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        num_xcds = matmul._num_xcds

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0

        if total_programs_streamk > 0:  # GEMM
            # last wave may occupy less than total_programs_streamk SMs
            total_tiles_streamk = total_tiles % total_programs_streamk
            # for two-tile GEMM + data-parallel from original paper
            #            if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            #                total_tiles_streamk += total_programs_streamk
            # remaining tiles are computed using classical blocking
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            # iterations related to full waves
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            # iterations related to last (partial) wave
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        else:  # all tiles are computed using classical blocking
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if matmul._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{total_full_tiles_streamk=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")
            print("total_remainder_iters_streamk=", total_partial_tiles_streamk)
        use_bias = False

        # compute grid (work to do per SM on the first wave)
        grids = total_programs_streamk
        stride_bias = bias.stride(0) if use_bias else 0
        kk = gemm_kernel[(grids,)](
            a,
            b,
            c,
            c_global,
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
            c_global.stride(0),
            c_global.stride(1),
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

        if matmul._debug and not is_triton_interpret_set():
            matmul.streamk_registers = kk.n_regs
            matmul.streamk_spills = kk.n_spills
            print(f"{kk.n_regs} registers used, {kk.n_spills} spills")
            # print(kk.asm['ttgir'])
            # print(kk.asm['amdgcn'])

        return c

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
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
        cu_count: int = None,
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp: torch.Tensor = None,
        mm_end_timestamp: torch.Tensor = None,
    ):
        matmul._call(
            a=a,
            b=b,
            c=c,
            c_global=c_global,
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
        return c
