# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Configuration structures for iris-ccl collective operations.
"""

from dataclasses import dataclass
import iris


@dataclass
class Config:
    """
    Configuration parameters for iris-ccl collective operations.
    
    This configuration struct encapsulates common kernel parameters that can be
    set once and reused across multiple collective calls, similar to the 
    origami config pattern from ROCm libraries.
    
    Args:
        block_size_m: Block size for the M dimension tiling (default: 128)
                      Optimized for Gluon all-to-all with minimal rows (4)
        block_size_n: Block size for the N dimension tiling (default: 128)
                      Optimized for Gluon all-to-all with full column vectorization (2048)
        swizzle_size: Number of tiles to swizzle/group together for 
                     better memory access patterns (default: 6)
        comm_sms: Number of SMs (Streaming Multiprocessors) to use for 
                 communication kernel (default: 64)
                 Optimized for Gluon all-to-all achieving (108)
        num_xcds: Number of XCCs. If None, auto-detected from system (default: None)
        use_gluon: If True, use Gluon-based implementation (default: False)
                   Gluon provides better control over warp-level traffic shaping
        all_reduce_variant: Variant for all-reduce operation (default: "atomic")
                           Options: "atomic", "ring", "two_shot"
        all_reduce_distribution: Distribution for two-shot all-reduce (default: 0)
                               0 for striding, 1 for block distribution
    
    Example:
        >>> import iris
        >>> from iris.ccl import Config
        >>> shmem = iris.iris()
        >>> config = Config(
        ...     block_size_m=128,
        ...     block_size_n=32,
        ...     swizzle_size=8,
        ...     comm_sms=64,
        ...     use_gluon=True
        ... )
        >>> shmem.ccl.all_to_all(output_tensor, input_tensor, config=config)
        
        >>> # All-reduce with ring variant
        >>> config = Config(all_reduce_variant="ring")
        >>> shmem.ccl.all_reduce(output_tensor, input_tensor, config=config)
    """
    block_size_m: int = 128
    block_size_n: int = 128
    swizzle_size: int = 6
    comm_sms: int = 64
    num_xcds: int = None
    chunk_size: int = None
    use_gluon: bool = False
    all_reduce_variant: str = "atomic"
    all_reduce_distribution: int = 0
    
    def __post_init__(self):
        """Validate and auto-detect num_xcds if not set."""
        if self.num_xcds is None:
            self.num_xcds = iris.hip.get_num_xcc()
            
        if self.chunk_size is None:
            self.chunk_size = self.swizzle_size * self.swizzle_size
            self.chunk_size = min(self.chunk_size, self.comm_sms // self.num_xcds)
        
        if self.block_size_m <= 0:
            raise ValueError(f"block_size_m must be positive, got {self.block_size_m}")
        if self.block_size_n <= 0:
            raise ValueError(f"block_size_n must be positive, got {self.block_size_n}")
        if self.swizzle_size <= 0:
            raise ValueError(f"swizzle_size must be positive, got {self.swizzle_size}")
        if self.comm_sms <= 0:
            raise ValueError(f"comm_sms must be positive, got {self.comm_sms}")
        if self.num_xcds <= 0:
            raise ValueError(f"num_xcds must be positive, got {self.num_xcds}")
        if self.all_reduce_variant not in ["atomic", "ring", "two_shot"]:
            raise ValueError(f"all_reduce_variant must be one of: 'atomic', 'ring', 'two_shot', got {self.all_reduce_variant}")
        if self.all_reduce_distribution not in [0, 1]:
            raise ValueError(f"all_reduce_distribution must be 0 (striding) or 1 (block), got {self.all_reduce_distribution}")

