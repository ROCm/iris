#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Unified sweep benchmark script for matmul and all-gather operations.

Runs benchmarks across all permutations of M, N, K dimensions.
Supports both operation types via --operation argument.

Usage:
    python sweep_bench.py --operation matmul_all_gather
    python sweep_bench.py --operation all_gather_matmul
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


# Project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dimension configurations to test.
# Each entry contains M_local (per-rank M), N, K, and an optional label.
DIMENSION_CONFIGS = [
    # {"m_local": 2048, "n": 2048, "k": 16384, "label": "M2048_N2048_K16384"},
    # {"m_local": 2048, "n": 16384, "k": 2048, "label": "M2048_N16384_K2048"},
    # {"m_local": 2048, "n": 16384, "k": 16384, "label": "M2048_N16384_K16384"},
    # {"m_local": 2048, "n": 16384, "k": 65536, "label": "M2048_N16384_K65536"},
    # {"m_local": 2048, "n": 131072, "k": 16384, "label": "M2048_N131072_K16384"},
    # {"m_local": 16384, "n": 2048, "k": 2048, "label": "M16384_N2048_K2048"},
    # {"m_local": 16384, "n": 2048, "k": 16384, "label": "M16384_N2048_K16384"},
    # {"m_local": 16384, "n": 2048, "k": 131072, "label": "M16384_N2048_K131072"},
    # {"m_local": 16384, "n": 16384, "k": 2048, "label": "M16384_N16384_K2048"},
    # {"m_local": 131072, "n": 2048, "k": 16384, "label": "M131072_N2048_K16384"},
    # From Paper: Design Space Exploration of DMA based finer-grain compute communication overlap
    # {"m_local": 16384, "n": 16384, "k": 131072, "label": "g1"}, #llama 3 405B
    # {"m_local": 131072, "n": 16384, "k": 16384, "label": "g2"},
    # {"m_local": 53248, "n": 16384, "k": 131072, "label": "g3"},
    # {"m_local": 16384, "n": 53248, "k": 16384, "label": "g4"},
    # {"m_local": 8192, "n": 8192, "k": 262144, "label": "g5"}, #llama 2
    # {"m_local": 262144, "n": 8192, "k": 8192, "label": "g6"},
    # {"m_local": 28672, "n": 8192, "k": 262144, "label": "g7"},
    # {"m_local": 262144, "n": 28672, "k": 8192, "label": "g8"},
    # {"m_local": 196608, "n": 18432, "k": 16384, "label": "g9"},
    # {"m_local": 196608, "n": 106496, "k": 16384, "label": "g10"}, # timeout
    # {"m_local": 1048576, "n": 10240, "k": 8192, "label": "g11"},
    # {"m_local": 1048576, "n": 57344, "k": 8192, "label": "g12"}, # run out of memory
    # {"m_local": 1607680, "n": 57344, "k": 8192, "label": "g13"}, # run out of memory
    # {"m_local": 147456, "n": 28672, "k": 4096, "label": "g14"},
    # {"m_local": 327680, "n": 28672, "k": 4096, "label": "g15"},
    # {"m_local": 229376, "n": 28672, "k": 4096, "label": "g16"},
    # {"m_local": 4096, "n": 14336, "k": 4096, "label": "mixtral_gate"},
    # {"m_local": 4096, "n": 11008, "k": 4096, "label": "llama7b_gate"},
    # {"m_local": 4096, "n": 4096, "k": 4096, "label": "pow2_4k"},
    # {"m_local": 1024, "n": 3584, "k": 8192, "label": "M1024_N3584_K8192"},
    # {"m_local": 4096, "n": 3584, "k": 8192, "label": "M4096_N3584_K8192"},
    # {"m_local": 16384, "n": 3584, "k": 8192, "label": "M16384_N3584_K8192"},
    # Modern LLM shapes for K-sharding TP (DeepSeek-V3, Llama 3/3.1, Llama 4)
    # These are for row-parallel matmuls where K dimension is sharded
    # DeepSeek-V3: hidden_size=7168, intermediate_size=18432
    {"m_local": 16384, "n": 7168, "k": 7168, "label": "deepseek_v3_attn_out_16k"},
    {"m_local": 16384, "n": 7168, "k": 18432, "label": "deepseek_v3_mlp_down_16k"},
    # {"m_local": 32768, "n": 7168, "k": 7168, "label": "deepseek_v3_attn_out_32k"},
    # {"m_local": 32768, "n": 7168, "k": 18432, "label": "deepseek_v3_mlp_down_32k"},
    # {"m_local": 65536, "n": 7168, "k": 7168, "label": "deepseek_v3_attn_out_64k"},
    # {"m_local": 65536, "n": 7168, "k": 18432, "label": "deepseek_v3_mlp_down_64k"},
    # Llama 4 Scout: hidden_size=8192 (est.), intermediate_size=22016 (est.)
    # Note: These dimensions are estimated based on Llama lineage and 17B active params
    # {"m_local": 16384, "n": 8192, "k": 8192, "label": "llama4_scout_attn_out_16k"},
    # {"m_local": 16384, "n": 8192, "k": 22016, "label": "llama4_scout_mlp_down_16k"},
    # {"m_local": 32768, "n": 8192, "k": 8192, "label": "llama4_scout_attn_out_32k"},
    # {"m_local": 32768, "n": 8192, "k": 22016, "label": "llama4_scout_mlp_down_32k"},
    # {"m_local": 65536, "n": 8192, "k": 8192, "label": "llama4_scout_attn_out_64k"},
    # {"m_local": 65536, "n": 8192, "k": 22016, "label": "llama4_scout_mlp_down_64k"},
    # Llama 3/3.1 8B: hidden_size=4096, intermediate_size=14336
    {"m_local": 16384, "n": 4096, "k": 4096, "label": "llama3_8b_attn_out_16k"},
    {"m_local": 16384, "n": 4096, "k": 14336, "label": "llama3_8b_mlp_down_16k"},

    # {"m_local": 32768, "n": 4096, "k": 4096, "label": "llama3_8b_attn_out_32k"},
    # {"m_local": 32768, "n": 4096, "k": 14336, "label": "llama3_8b_mlp_down_32k"},
    # {"m_local": 65536, "n": 4096, "k": 4096, "label": "llama3_8b_attn_out_64k"},
    # {"m_local": 65536, "n": 4096, "k": 14336, "label": "llama3_8b_mlp_down_64k"},
    # Llama 3/3.1 70B: hidden_size=8192, intermediate_size=28672 (same as Llama 2 70B)
    {"m_local": 16384, "n": 8192, "k": 8192, "label": "llama3_70b_attn_out_16k"},
    {"m_local": 16384, "n": 8192, "k": 28672, "label": "llama3_70b_mlp_down_16k"},

    # {"m_local": 32768, "n": 8192, "k": 8192, "label": "llama3_70b_attn_out_32k"},
    # {"m_local": 32768, "n": 8192, "k": 28672, "label": "llama3_70b_mlp_down_32k"},
    # {"m_local": 65536, "n": 8192, "k": 8192, "label": "llama3_70b_attn_out_64k"},
    # {"m_local": 65536, "n": 8192, "k": 28672, "label": "llama3_70b_mlp_down_64k"},
    # Note: Llama 3.1 405B shapes already covered by g1-g4 (hidden=16384, intermediate=53248)
    {"m_local": 16384, "n": 16384, "k": 16384, "label": "llama3_405b_attn_out_16k"},
    {"m_local": 16384, "n": 16384, "k": 53248, "label": "llama3_405b_mlp_down_16k"},
    # NOTE: The shapes below are for SEQUENCE PARALLEL (M-sharding), which is
    # a DIFFERENT operation than all_gather_matmul (which does K-sharding for TP).
    # These shapes are kept for reference but won't work with current all_gather_matmul.
    # TODO: Implement separate M-sharding operation for sequence parallel.
    # # DeepSeek-V3 sequence parallel (TP4, hidden_dim=7168, context=128K)
    # {"m_local": 32768, "n": 7168, "k": 18432, "label": "deepseek_v3_sp4_mlp_up"},
    # {"m_local": 32768, "n": 18432, "k": 7168, "label": "deepseek_v3_sp4_mlp_down"},
    # {"m_local": 32768, "n": 7168, "k": 7168, "label": "deepseek_v3_sp4_qkv"},
    # {"m_local": 65536, "n": 7168, "k": 18432, "label": "deepseek_v3_sp4_mlp_up_long"},
    # # Llama 4 Scout sequence parallel (TP8, hidden_dim=8192, context=256K)
    # {"m_local": 32768, "n": 8192, "k": 22016, "label": "llama4_scout_sp8_mlp_up"},
    # {"m_local": 32768, "n": 22016, "k": 8192, "label": "llama4_scout_sp8_mlp_down"},
    # {"m_local": 32768, "n": 8192, "k": 8192, "label": "llama4_scout_sp8_qkv"},
    # # Llama 4 Maverick long context (TP8, hidden_dim=8192, context=512K)
    # {"m_local": 64768, "n": 8192, "k": 22016, "label": "llama4_maverick_sp8_mlp_up_long"},
    # {"m_local": 64768, "n": 22016, "k": 8192, "label": "llama4_maverick_sp8_mlp_down_long"},
    # # DeepSeek-V4 ultra-long context (TP8, hidden_dim~10240, context=1M tokens)
    # {"m_local": 131072, "n": 10240, "k": 28672, "label": "deepseek_v4_sp8_mlp_up_1m"},
    # {"m_local": 131072, "n": 28672, "k": 10240, "label": "deepseek_v4_sp8_mlp_down_1m"},
    # {"m_local": 131072, "n": 10240, "k": 10240, "label": "deepseek_v4_sp8_qkv_1m"},
    # # Llama 4 Behemoth extreme context (TP16, hidden_dim~12288, context=10M tokens)
    # {"m_local": 655360, "n": 12288, "k": 32768, "label": "llama4_behemoth_sp16_mlp_up_10m"},
    # {"m_local": 655360, "n": 32768, "k": 12288, "label": "llama4_behemoth_sp16_mlp_down_10m"},
    # {"m_local": 65536, "n": 12288, "k": 32768, "label": "llama4_behemoth_sp16_mlp_up_1m"},
    # {"m_local": 65536, "n": 32768, "k": 12288, "label": "llama4_behemoth_sp16_mlp_down_1m"},
]

# Benchmark configurations per operation type
BENCHMARK_CONFIGS = {
    "matmul_all_gather": {
        "pytorchbaseline": {
            "script": "benchmark/ops/bench_matmul_all_gather.py",
            "benchmark_filter": "^pytorch_matmul_all_gather$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "tritonblas_rcclbaseline": {
            "script": "benchmark/ops/bench_matmul_all_gather.py",
            "benchmark_filter": "^tritonblas_matmul_all_gather$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "baseline": {
            "script": "benchmark/ops/bench_matmul_all_gather.py",
            "benchmark_filter": "^matmul_all_gather$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "host_copy_engine": {
            "script": "benchmark/ops/bench_matmul_all_gather_copy_engine.py",
            "benchmark_filter": "^matmul_all_gather_copy_engine_host$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "device_copy_engine": {
            "script": "benchmark/ops/bench_matmul_all_gather_copy_engine.py",
            "benchmark_filter": "^matmul_all_gather_copy_engine_device$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "matmul_only": {
            "script": "benchmark/ops/bench_matmul.py",
            "benchmark_filter": "^matmul_only_local$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
        "pytorchmatmul_only": {
            "script": "benchmark/ops/bench_matmul.py",
            "benchmark_filter": "^pytorch_matmul_only_local$",
            "axes": {"m": "M_local", "n": "N", "k": "K"},
        },
    },
    "all_gather_matmul": {
        "baseline": {
            "script": "benchmark/ops/bench_all_gather_matmul.py",
            "benchmark_filter": "^all_gather_matmul$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "pytorchbaseline": {
            "script": "benchmark/ops/bench_all_gather_matmul.py",
            "benchmark_filter": "^rccl_all_gather_matmul$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "tritonblas_rcclbaseline": {
            "script": "benchmark/ops/bench_all_gather_matmul.py",
            "benchmark_filter": "^tritonblas_rccl_all_gather_matmul$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "hbm_buffer": {
            "script": "benchmark/ops/bench_all_gather_matmul.py",
            "benchmark_filter": "^all_gather_matmul_hbm_buffer$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "copy_engine_host": {
            "script": "benchmark/ops/bench_all_gather_matmul_copy_engine.py",
            "benchmark_filter": "^all_gather_matmul_copy_engine_host$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "copy_engine_host_hip_memcpy": {
            "script": "benchmark/ops/bench_all_gather_matmul_copy_engine.py",
            "benchmark_filter": "^all_gather_matmul_copy_engine_host_hip_memcpy$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "copy_engine_device": {
            "script": "benchmark/ops/bench_all_gather_matmul_copy_engine.py",
            "benchmark_filter": "^all_gather_matmul_copy_engine_device$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "matmul_only": {
            "script": "benchmark/ops/bench_matmul.py",
            "benchmark_filter": "^matmul_only$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "pytorchmatmul_only": {
            "script": "benchmark/ops/bench_matmul.py",
            "benchmark_filter": "^pytorch_matmul_only$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
    },
    "matmul_all_reduce": {
        "pytorchbaseline": {
            "script": "benchmark/ops/bench_matmul_all_reduce.py",
            "benchmark_filter": "^pytorch_matmul_all_reduce$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "tritonblas_rcclbaseline": {
            "script": "benchmark/ops/bench_matmul_all_reduce.py",
            "benchmark_filter": "^tritonblas_rccl_matmul_all_reduce$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        "one_shot": {
            "script": "benchmark/ops/bench_matmul_all_reduce.py",
            "benchmark_filter": "^matmul_all_reduce$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
            "variant": "one_shot",
        },
        "two_shot": {
            "script": "benchmark/ops/bench_matmul_all_reduce.py",
            "benchmark_filter": "^matmul_all_reduce$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
            "variant": "two_shot",
        },
        # "spinlock": {
        #     "script": "benchmark/ops/bench_matmul_all_reduce.py",
        #     "benchmark_filter": "^matmul_all_reduce$",
        #     "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
        #     "variant": "spinlock",
        # },
        # "copy_engine_one_shot": {
        #     "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
        #     "benchmark_filter": "^matmul_all_reduce_copy_engine$",
        #     "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
        #     "variant": "one_shot",
        # },
        # "copy_engine_two_shot": {
        #     "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
        #     "benchmark_filter": "^matmul_all_reduce_copy_engine$",
        #     "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
        #     "variant": "two_shot",
        # },
        # "copy_engine_host_hip_memcpy_one_shot": {
        #     "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
        #     "benchmark_filter": "^matmul_all_reduce_copy_engine_host_hip_memcpy$",
        #     "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
        #     "variant": "one_shot",
        # },
        # "copy_engine_host_hip_memcpy_two_shot": {
        #     "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
        #     "benchmark_filter": "^matmul_all_reduce_copy_engine_host_hip_memcpy$",
        #     "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant"},
        #     "variant": "two_shot",
        # },
        "copy_engine_device_one_shot": {
            "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
            "benchmark_filter": "^matmul_all_reduce_copy_engine$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant", "prepost": "prepost"},
            "variant": "one_shot",
            "prepost": False,
        },
        "copy_engine_device_two_shot": {
            "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
            "benchmark_filter": "^matmul_all_reduce_copy_engine$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant", "prepost": "prepost"},
            "variant": "two_shot",
            "prepost": False,
        },
        "copy_engine_device_one_shot_prepost": {
            "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
            "benchmark_filter": "^matmul_all_reduce_copy_engine$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant", "prepost": "prepost"},
            "variant": "one_shot",
            "prepost": True,
        },
        "copy_engine_device_two_shot_prepost": {
            "script": "benchmark/ops/bench_matmul_all_reduce_copy_engine.py",
            "benchmark_filter": "^matmul_all_reduce_copy_engine$",
            "axes": {"m": "M", "n": "N", "k": "K", "variant": "variant", "prepost": "prepost"},
            "variant": "two_shot",
            "prepost": True,
        },
        "copy_engine_partitioned_gemm": {
            "script": "benchmark/ops/bench_matmul.py",
            "benchmark_filter": "^matmul_copy_engine_partitioned_gemm$",
            "axes": {"m": "M", "n": "N", "k": "K"},
        },
        # "matmul_only": {
        #     "script": "benchmark/ops/bench_matmul.py",
        #     "benchmark_filter": "^matmul_only$",
        #     "axes": {"m": "M", "n": "N", "k": "K"},
        # },
        # "matmul_work_stealing": {
        #     "script": "benchmark/ops/bench_matmul.py",
        #     "benchmark_filter": "^matmul_work_stealing$",
        #     "axes": {"m": "M", "n": "N", "k": "K"},
        # },
        # "matmul_streamk": {
        #     "script": "benchmark/ops/bench_matmul.py",
        #     "benchmark_filter": "^matmul_streamk$",
        #     "axes": {"m": "M", "n": "N", "k": "K"},
        # },
        # "pytorchmatmul_only": {
        #     "script": "benchmark/ops/bench_matmul.py",
        #     "benchmark_filter": "^pytorch_matmul_only$",
        #     "axes": {"m": "M", "n": "N", "k": "K"},
        # },
    },
}

TIMEOUT_SECONDS = 150
NUM_GPUS = 8
DEFAULT_HEAP_SIZE = 1 << 34  # 16 GB
HEAP_SIZE_SAFETY_FACTOR = 1.25  # Multiply calculated size by this factor for safety


def log(msg: str):
    """Log to stderr to keep stdout clean for JSON."""
    print(msg, file=sys.stderr, flush=True)


def _dimension_values(config: Dict[str, Any]) -> tuple[int, int, int]:
    return int(config["m_local"]), int(config["n"]), int(config["k"])


def _dimension_label(config: Dict[str, Any]) -> str:
    return str(config.get("label") or f"M{config['m_local']}_N{config['n']}_K{config['k']}")


def _calculate_heap_size(
    m: int, n: int, k: int, operation: str, num_gpus: int, safety_factor: float = HEAP_SIZE_SAFETY_FACTOR
) -> int:
    """Calculate required heap size based on matrix dimensions and operation type.

    The heap is used for symmetric memory buffers across GPUs for all allocations via ctx.zeros(), ctx.randn(), etc.

    For all_gather_matmul:
        - Input A_sharded per rank: (M × K_local) where K_local = K / num_gpus
        - Gathered A_full: (M × K) - allocated in heap
        - Matrix B: (K × N) - allocated in heap
        - Output C: (M × N) - allocated in heap
        - Communication workspace and intermediate buffers

    For matmul_all_gather:
        - Input A: (M_local × K) where M_local is per-rank
        - Matrix B: (K × N) - allocated in heap
        - Output C_local: (M_local × N) per rank - allocated in heap
        - Gathered C_full: (M_total × N) where M_total = M_local * num_gpus - allocated in heap
        - Communication workspace

    Returns:
        Heap size in bytes
    """
    element_size = 2  # fp16 = 2 bytes

    if operation == "all_gather_matmul":
        # All allocations from bench_all_gather_matmul.py:
        # - A_sharded: M × K_local (per rank, before gather)
        # - Gathered buffer: M × K (allocated differently per variant)
        #   * RCCL baseline: a_gathered_parts list + a_gathered concat (both M × K total)
        #   * HBM buffer: workspace.aux_buffer (single M × K buffer)
        # - B: K × N
        # - C: M × N (output)
        k_local = k // num_gpus

        a_sharded_size = m * k_local * element_size
        # RCCL allocates both a_gathered_parts AND a_gathered, but they're equal size
        # a_gathered_parts is allocated as world_size separate buffers of K_local each
        # a_gathered is M × K contiguous
        # Both are M × K total, so worst case is 2 × (M × K)
        a_gather_buffer_size = m * k * element_size * 2  # Conservative: both parts + gathered
        b_size = k * n * element_size
        c_size = m * n * element_size

        total_size = a_sharded_size + a_gather_buffer_size + b_size + c_size

    elif operation == "matmul_all_gather":
        # All allocations from bench_matmul_all_gather.py:
        # NOTE: m represents TOTAL M across all GPUs, so M_local = m / num_gpus
        # - A: M_local × K (per rank)
        # - B: K × N
        # - C_local: M_local × N (for pytorch/tritonblas variants)
        # - C: M × N (FULL output on EACH GPU where M = total across all ranks)

        # n: hidden_size
        # k: intermediate_size

        # Figure out if attention
        if k == n:
            m_local = m // num_gpus
            n_local = n // num_gpus
            k_local = n

        else:
            m_local = m
            n_local = k // num_gpus
            k_local = n

        a_size = m_local * k_local * element_size
        b_size = k_local * n_local * element_size
        c_local_size = m_local * n_local * element_size  # For pytorch/tritonblas variants
        c_size = m * n_local * element_size  # Full gathered output (M total)

        # Conservative estimate: assume both C_local and C are allocated
        total_size = a_size + b_size + c_local_size + c_size

    elif operation == "matmul_all_reduce":
        # All allocations from bench_matmul_all_reduce.py:
        # - A: M × K
        # - B: K × N
        # - C: M × N (output)
        # - a_inbox: world_size × M × N in the worst one_shot case
        # - locks/completion_signals: small readiness counters

        a_size = m * k * element_size
        b_size = k * n * element_size
        c_size = m * n * element_size

        total_size = a_size + b_size + c_size

        inbox_size = num_gpus * m * n * element_size
        readiness_size = (1 + num_gpus) * 4
        total_size += inbox_size + readiness_size

    else:
        total_size = DEFAULT_HEAP_SIZE

    # Apply safety factor and ensure minimum size
    required_size = int(total_size * safety_factor)
    heap_size = max(DEFAULT_HEAP_SIZE, required_size)

    # Cap heap size at GPU memory limit to avoid OOM
    # MI300X has ~192 GB per GPU
    max_reasonable_heap = 180 << 30  # 180 GB to leave room for PyTorch overhead

    # Check if even the base requirement (without safety factor) exceeds GPU memory
    if total_size > max_reasonable_heap:
        log(f"  WARNING: Required buffers ({total_size / (1 << 30):.2f} GB) exceed available GPU memory (~180 GB)")
        log("           This shape is too large for the available hardware - benchmark WILL FAIL")
        log(f"           Breakdown: M={m}, N={n}, K={k}, num_gpus={num_gpus}")
        # Still return the capped size so the sweep continues, but mark for expected failure

    if heap_size > max_reasonable_heap:
        log(f"  INFO: Calculated heap size ({heap_size / (1 << 30):.2f} GB) exceeds GPU memory")
        log(f"        Capping at {max_reasonable_heap / (1 << 30):.0f} GB")
        heap_size = max_reasonable_heap

    return heap_size


def _run_bench_benchmark(
    benchmark_name: str,
    script: str,
    m: int,
    n: int,
    k: int,
    benchmark_filter: str,
    axes: Dict[str, str],
    heap_size: int,
    operation: str,
    variant: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{benchmark_name}_M{m}_N{n}_K{k}.json",
        dir=str(PROJECT_ROOT),
        delete=False,
    ) as tmp_file:
        benchmark_out = tmp_file.name

    # For matmul_all_gather, m represents total M, but the benchmark expects M_local
    # So we divide by NUM_GPUS to get the per-rank dimension
    if operation == "matmul_all_gather" and axes["m"] == "M_local":
        # m_value = m // NUM_GPUS
        # if B is square (n == k) the is attention
        if k == n:
            m_value = m // NUM_GPUS
            n_value = n // NUM_GPUS
            k_value = n

        else:
            m_value = m
            n_value = k // NUM_GPUS
            k_value = n
    else:
        m_value = m
        n_value = n
        k_value = k

    cmd = [
        sys.executable,
        script,
        "--benchmark_format=json",
        f"--benchmark_out={benchmark_out}",
        f"--benchmark_filter={benchmark_filter}",
        f"--axis_num_ranks={NUM_GPUS}",
        f"--axis_{axes['m']}={m_value}",
        f"--axis_{axes['n']}={n_value}",
        f"--axis_{axes['k']}={k_value}",
        "--axis_dtype=fp16",
        f"--heap_size={heap_size}",
    ]

    # Add variant parameter if present (for matmul_all_reduce)
    if variant is not None and "variant" in axes:
        cmd.append(f"--axis_{axes['variant']}={variant}")

    log(f"  Running {benchmark_name}: M={m_value}, N={n_value}, K={k_value}")
    log(f"    Heap size: {heap_size / (1 << 30):.2f} GB")
    log(f"    Command: {' '.join(cmd)}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            preexec_fn=os.setsid,
        )
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

        if result.returncode != 0:
            log("    ✗ Failed: Non-zero return code")
            log(f"    Return code: {result.returncode}")
            error_log_file = PROJECT_ROOT / f"benchmark_error_{operation}_{benchmark_name}_M{m}_N{n}_K{k}.log"
            with open(error_log_file, "w") as f:
                f.write(f"Operation: {operation}\n")
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Dimensions: M={m_value}, N={n_value}, K={k_value}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("=" * 80 + "\n")
                f.write("STDOUT:\n")
                f.write("=" * 80 + "\n")
                f.write(result.stdout)
                f.write("\n" + "=" * 80 + "\n")
                f.write("STDERR:\n")
                f.write("=" * 80 + "\n")
                f.write(result.stderr)
            log(f"    Full output saved to: {error_log_file}")
            lines = (result.stdout + result.stderr).strip().split("\n")
            log("    Last output lines:")
            for line in lines[-5:]:
                log(f"      {line}")
            return None

        with open(benchmark_out, "r") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            log("    ✗ Failed: bench JSON output was empty")
            return None

        record = next((r for r in records if not r.get("skipped")), None)
        if record is None:
            skip_reason = records[0].get("skip_reason", "")
            log(f"    ✗ Failed: benchmark was skipped ({skip_reason})")
            return {"status": "SKIPPED", "skip_reason": skip_reason}

        params = record.get("params", {})
        counters = record.get("counters", {})
        data = {
            "world_size": record.get("world_size"),
            "operation": record.get("benchmark"),
            "m": int(params.get(axes["m"], m_value)),
            "n": int(params.get(axes["n"], n_value)),
            "k": int(params.get(axes["k"], k_value)),
            "datatype": params.get("dtype", "float16"),
            "total_ms": record.get("gpu_time_ms"),
            "gpu_time_ms": record.get("gpu_time_ms"),
            "all_times_ms": record.get("all_times_ms", []),
            "bandwidth_gbps": record.get("bandwidth_gbps"),
            "tflops": record.get("tflops"),
        }
        data.update(counters)
        log("    ✓ Success: Loaded bench JSON results")
        return data

    except subprocess.TimeoutExpired as timeout_err:
        log(f"    ✗ Timeout after {TIMEOUT_SECONDS}s - killing process group")
        partial_stdout = timeout_err.stdout.decode("utf-8", errors="replace") if timeout_err.stdout else ""
        partial_stderr = timeout_err.stderr.decode("utf-8", errors="replace") if timeout_err.stderr else ""
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log("    Process didn't terminate, force killing...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
        error_log_file = PROJECT_ROOT / f"benchmark_timeout_{operation}_{benchmark_name}_M{m}_N{n}_K{k}.log"
        with open(error_log_file, "w") as f:
            f.write(f"Operation: {operation}\n")
            f.write(f"Benchmark: {benchmark_name}\n")
            f.write(f"Dimensions: M={m}, N={n}, K={k}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Status: TIMEOUT after {TIMEOUT_SECONDS}s\n\n")
            f.write("=" * 80 + "\n")
            f.write("PARTIAL STDOUT (before timeout):\n")
            f.write("=" * 80 + "\n")
            f.write(partial_stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write("PARTIAL STDERR (before timeout):\n")
            f.write("=" * 80 + "\n")
            f.write(partial_stderr)
        log(f"    Timeout logged to: {error_log_file}")
        return None
    except json.JSONDecodeError as e:
        log(f"    ✗ Error: Failed to parse bench JSON: {e}")
        return None
    except Exception as e:
        log(f"    ✗ Error: {e}")
        return None
    finally:
        try:
            os.remove(benchmark_out)
        except OSError:
            pass


def run_benchmark(
    benchmark_name: str,
    bench_config: Dict[str, Any],
    m: int,
    n: int,
    k: int,
    heap_size: int,
    operation: str,
) -> Optional[Dict[str, Any]]:
    return _run_bench_benchmark(
        benchmark_name,
        bench_config["script"],
        m,
        n,
        k,
        bench_config["benchmark_filter"],
        bench_config.get("axes", {"m": "M", "n": "N", "k": "K"}),
        heap_size,
        operation,
        bench_config.get("variant"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep benchmarks for matmul and all-gather operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=["matmul_all_gather", "all_gather_matmul", "matmul_all_reduce"],
        help="Operation type to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: benchmark/ops/{operation}/benchmark_sweep_results.json)",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated list of shape labels to run (e.g., 'g1,g2,pow2_4k'). If not provided, runs all shapes.",
    )
    args = parser.parse_args()

    operation = args.operation
    benchmarks = BENCHMARK_CONFIGS[operation]

    # Filter shapes if --shapes argument provided
    if args.shapes:
        requested_labels = set(label.strip() for label in args.shapes.split(","))
        filtered_configs = [cfg for cfg in DIMENSION_CONFIGS if cfg["label"] in requested_labels]
        if not filtered_configs:
            log(f"Error: No shapes found matching: {args.shapes}")
            log(f"Available labels: {', '.join(cfg['label'] for cfg in DIMENSION_CONFIGS)}")
            sys.exit(1)
        dimension_configs = filtered_configs
        log(f"Running benchmarks for shapes: {', '.join(cfg['label'] for cfg in dimension_configs)}")
    else:
        dimension_configs = DIMENSION_CONFIGS
        log(f"Running benchmarks for all {len(dimension_configs)} shapes")

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = PROJECT_ROOT / f"benchmark/ops/benchmark_sweep_results_{operation}.json"

    log("=" * 80)
    log(f"{operation.upper().replace('_', '-')} Benchmark Sweep")
    log("=" * 80)
    log(f"Dimension configurations: {len(dimension_configs)}")
    for config in dimension_configs:
        m, n, k = _dimension_values(config)
        log(f"  - {_dimension_label(config)}: M={m}, N={n}, K={k}")
    log(f"Benchmarks per configuration: {len(benchmarks)}")
    log(f"Total benchmarks: {len(dimension_configs) * len(benchmarks)}")
    log(f"Timeout per benchmark: {TIMEOUT_SECONDS}s")
    log(f"GPUs: {NUM_GPUS}")
    log(f"Output file: {output_file}")
    log("=" * 80)
    log("")

    results = []

    log(f"Running {len(dimension_configs)} dimension configurations...\n")

    for idx, config in enumerate(dimension_configs, 1):
        m, n, k = _dimension_values(config)
        label = _dimension_label(config)

        heap_size = _calculate_heap_size(m, n, k, operation, NUM_GPUS, HEAP_SIZE_SAFETY_FACTOR)
        log(f"[{idx}/{len(dimension_configs)}] Testing {label}: M={m}, N={n}, K={k}")
        log(f"  Calculated heap size: {heap_size / (1 << 30):.2f} GB (factor: {HEAP_SIZE_SAFETY_FACTOR})")

        row = {"label": label, "M": m, "N": n, "K": k, "operation": operation, "benchmarks": {}}

        # Run each benchmark variant
        for bench_key, bench_config in benchmarks.items():
            result = run_benchmark(
                benchmark_name=bench_key,
                bench_config=bench_config,
                m=m,
                n=n,
                k=k,
                heap_size=heap_size,
                operation=operation,
            )

            if result is not None:
                row["benchmarks"][bench_key] = result
            else:
                row["benchmarks"][bench_key] = {"status": "FAILED"}

        results.append(row)
        log("")

    # Write JSON file
    log(f"Writing results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log(f"✓ Results saved to {output_file}\n")

    log("\n" + "=" * 80)
    log("Benchmark sweep complete!")
    log("=" * 80)


if __name__ == "__main__":
    main()
