# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


import torch

from gemm_all_reduce_ring_based import persistent_gemm

import iris

gemm_kernel = persistent_gemm


class matmul(torch.autograd.Function):
    _num_xcds = iris.hip.get_num_xcc()

    @staticmethod
    def _call(
        a: torch.Tensor,
        b: torch.Tensor,
        ring_buffer: torch.Tensor,
        bias: torch.Tensor,
        locks: torch.Tensor,
        rank: int,
        world_size: int,
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

        num_xcds = matmul._num_xcds

        num_warps = 8
        waves_per_eu = 0
        mfma = 16
        kpack = 1

        even_k = K % BLK_K == 0
        use_bias = False

        stride_bias = bias.stride(0) if use_bias else 0
        gemm_kernel[(num_sms,)](
            a,
            b,
            ring_buffer,
            bias,
            locks,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            ring_buffer.stride(0),
            ring_buffer.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=num_sms,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
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

        return ring_buffer

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        ring_buffer: torch.Tensor,
        bias: torch.Tensor,
        locks: torch.Tensor,
        rank: int,
        world_size: int,
        num_sms: int,
        BLK_M: int,
        BLK_N: int,
        BLK_K: int,
        gsize_m: int,
        num_stages: int,
        heap_bases_ptr: torch.Tensor = None,
        arch: str = "gfx942",
    ):
        matmul._call(
            a=a,
            b=b,
            ring_buffer=ring_buffer,
            bias=bias,
            locks=locks,
            rank=rank,
            world_size=world_size,
            num_sms=num_sms,
            BLK_M=BLK_M,
            BLK_N=BLK_N,
            BLK_K=BLK_K,
            gsize_m=gsize_m,
            num_stages=num_stages,
            heap_bases_ptr=heap_bases_ptr,
            arch=arch,
        )
        return ring_buffer
