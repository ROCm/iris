# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch

from gemm_reduce_scatter import persistent_gemm_reduce_scatter_wg_specialized

import iris

gemm_kernel = persistent_gemm_reduce_scatter_wg_specialized


class MatMulReduceScatterWgSpecialized(torch.autograd.Function):
    _num_xcds = iris.hip.get_num_xcc()

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
        locks: torch.Tensor,
        rank: int,
        world_size: int,
        gemm_sms: int,
        num_sms: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        num_stages: int,
        heap_bases_ptr: torch.Tensor = None,
        arch: str = "gfx942",
    ):
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        num_xcds = MatMulReduceScatterWgSpecialized._num_xcds
        num_warps = 8
        waves_per_eu = 0
        mfma = 16
        kpack = 1
        even_k = K % BLK_K == 0

        grid = (num_sms,)

        gemm_kernel[grid](
            a,
            b,
            c,
            c_global,
            locks,
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
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            GEMM_SMS=gemm_sms,
            NUM_SMS=num_sms,
            NUM_XCDS=num_xcds,
            EVEN_K=even_k,
            num_stages=num_stages,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=mfma,
            kpack=kpack,
            heap_bases=heap_bases_ptr,
            cur_rank=rank,
            world_size=world_size,
        )

        return c_global

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        c_global: torch.Tensor,
        locks: torch.Tensor,
        rank: int,
        world_size: int,
        gemm_sms: int,
        num_sms: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        num_stages: int,
        heap_bases_ptr: torch.Tensor = None,
        arch: str = "gfx942",
    ):
        return MatMulReduceScatterWgSpecialized._call(
            a=a,
            b=b,
            c=c,
            c_global=c_global,
            locks=locks,
            rank=rank,
            world_size=world_size,
            gemm_sms=gemm_sms,
            num_sms=num_sms,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            gsize_m=gsize_m,
            num_stages=num_stages,
            heap_bases_ptr=heap_bases_ptr,
            arch=arch,
        )
