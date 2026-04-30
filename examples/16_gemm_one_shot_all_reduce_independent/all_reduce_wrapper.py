# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch

from gemm_one_shot_all_reduce_independent import persistent_all_reduce


class all_reduce_kernel:
    """Wrapper class for the persistent_all_reduce kernel."""

    @staticmethod
    def run(
        local_data: torch.Tensor,
        global_result: torch.Tensor,
        M: int,
        N: int,
        stride_local_m: int,
        stride_local_n: int,
        stride_global_m: int,
        stride_global_n: int,
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_N: int,
        GROUP_SIZE_M: int,
        COMM_SMS: int,
        NUM_XCDS: int,
        heap_bases: torch.Tensor,
        cur_rank: int,
        world_size: int,
        DISTRIBUTION: int,
    ):
        """Run persistent_all_reduce kernel."""
        persistent_all_reduce[(COMM_SMS,)](
            local_data,
            global_result,
            M,
            N,
            stride_local_m,
            stride_local_n,
            stride_global_m,
            stride_global_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_XCDS,
            heap_bases,
            cur_rank,
            world_size,
            DISTRIBUTION,
        )
