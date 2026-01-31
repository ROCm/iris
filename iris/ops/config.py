# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Configuration for fused GEMM+CCL operations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FusedConfig:
    """
    Configuration for fused GEMM+CCL operations.

    This class holds tuning parameters for both GEMM computation and collective
    communication operations. It provides sensible defaults that work for most cases,
    but users can override specific settings for performance tuning.

    GEMM Parameters:
        block_size_m: Block size for M dimension (rows). Default: 256.
        block_size_n: Block size for N dimension (columns). Default: 64.
        block_size_k: Block size for K dimension (reduction). Default: 64.
        group_size_m: Group size for M dimension tiling. Default: 1.
        num_sms: Number of SMs to use. If None, auto-detects from device. Default: None.
        num_xcds: Number of XCDs (chiplets). Default: 1.
        chunk_size: Chunk size for chiplet transform. Default: 1.
        cache_modifier_a: Cache modifier for matrix A (".ca" for cached). Default: ".ca".
        cache_modifier_b: Cache modifier for matrix B (".ca" for cached). Default: ".ca".
        allow_tf32: Whether to allow TF32 precision. Default: True.

    CCL Parameters (for operations that need collective communication):
        all_reduce_variant: All-reduce algorithm variant. Options: "atomic", "ring",
                           "one_shot", "two_shot", "spinlock". Default: "one_shot".
        all_reduce_num_rings: Number of concurrent rings (for ring variant). Default: 1.

    Example:
        >>> # Use defaults
        >>> config = FusedConfig()
        >>>
        >>> # Custom block sizes
        >>> config = FusedConfig(block_size_m=128, block_size_n=128)
        >>>
        >>> # Use ring all-reduce
        >>> config = FusedConfig(all_reduce_variant="ring", all_reduce_num_rings=2)
    """

    # GEMM parameters
    block_size_m: int = 256
    block_size_n: int = 64
    block_size_k: int = 64
    group_size_m: int = 1
    num_sms: Optional[int] = None  # Auto-detect if None
    num_xcds: int = 1
    chunk_size: int = 1
    cache_modifier_a: str = ".ca"
    cache_modifier_b: str = ".ca"
    allow_tf32: bool = True

    # CCL-specific parameters
    all_reduce_variant: str = "two_shot"  # atomic, ring, one_shot, two_shot, spinlock
    all_reduce_num_rings: int = 1

    def validate(self, world_size: Optional[int] = None):
        """
        Validate configuration parameters.

        Args:
            world_size: Number of ranks (needed for some validations). Default: None.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.block_size_m <= 0:
            raise ValueError(f"block_size_m must be positive, got {self.block_size_m}")
        if self.block_size_n <= 0:
            raise ValueError(f"block_size_n must be positive, got {self.block_size_n}")
        if self.block_size_k <= 0:
            raise ValueError(f"block_size_k must be positive, got {self.block_size_k}")
        if self.group_size_m <= 0:
            raise ValueError(f"group_size_m must be positive, got {self.group_size_m}")
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError(f"num_sms must be positive, got {self.num_sms}")
        if self.num_xcds <= 0:
            raise ValueError(f"num_xcds must be positive, got {self.num_xcds}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Validate all_reduce_variant
        valid_variants = ["atomic", "ring", "one_shot", "two_shot", "spinlock"]
        if self.all_reduce_variant not in valid_variants:
            raise ValueError(f"all_reduce_variant must be one of {valid_variants}, got {self.all_reduce_variant}")

        # Ring variant requires block_size_n divisible by world_size
        if self.all_reduce_variant == "ring" and world_size is not None:
            if self.block_size_n % world_size != 0:
                raise ValueError(
                    f"For ring variant, block_size_n ({self.block_size_n}) must be "
                    f"divisible by world_size ({world_size})"
                )

        if self.all_reduce_num_rings <= 0:
            raise ValueError(f"all_reduce_num_rings must be positive, got {self.all_reduce_num_rings}")
