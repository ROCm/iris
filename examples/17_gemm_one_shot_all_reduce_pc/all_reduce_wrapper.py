# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import iris

from gemm_one_shot_all_reduce_pc import persistent_all_reduce
from examples.common.utils import is_triton_interpret_set


class all_reduce_kernel:
    """Wrapper class to track register and spill counts for persistent_all_reduce kernel."""

    _debug = True
    _registers = None
    _spills = None

    @staticmethod
    def set_debug(debug: bool):
        all_reduce_kernel._debug = debug

    @staticmethod
    def get_registers():
        if all_reduce_kernel._debug:
            return all_reduce_kernel._registers
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")

    @staticmethod
    def get_spills():
        if all_reduce_kernel._debug:
            return all_reduce_kernel._spills
        else:
            raise RuntimeError("Debug mode is not enabled. Call set_debug(True) first.")

    @staticmethod
    def run(
        local_C: torch.Tensor,
        C_global: torch.Tensor,
        locks: torch.Tensor,
        tile_ready: torch.Tensor,
        M: int,
        N: int,
        stride_cm_local: int,
        stride_cn_local: int,
        stride_cm_global: int,
        stride_cn_global: int,
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_N: int,
        GROUP_SIZE_M: int,
        COMM_SMS: int,
        NUM_XCDS: int,
        heap_bases: torch.Tensor,
        cur_rank: int,
        world_size: int,
        DISTRIBUTION: int,
        COLLECT_TIMESTAMPS: bool = False,
        mm_begin_timestamp_ptr: torch.Tensor = None,
        mm_end_timestamp_ptr: torch.Tensor = None,
    ):
        """Run persistent_all_reduce kernel and capture register/spill counts."""
        kk = persistent_all_reduce[(COMM_SMS,)](
            local_C,
            C_global,
            locks,
            tile_ready,
            M,
            N,
            stride_cm_local,
            stride_cn_local,
            stride_cm_global,
            stride_cn_global,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_M,
            COMM_SMS,
            NUM_XCDS,
            heap_bases,
            cur_rank,
            world_size,
            DISTRIBUTION,
            COLLECT_TIMESTAMPS,
            mm_begin_timestamp_ptr,
            mm_end_timestamp_ptr,
        )

        all_reduce_kernel._registers = kk.n_regs
        all_reduce_kernel._spills = kk.n_spills
