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
        block_size_m: Block size for the M dimension tiling (default: 256)
        block_size_n: Block size for the N dimension tiling (default: 64)
        swizzle_size: Number of tiles to swizzle/group together for 
                     better memory access patterns (default: 6)
        comm_sms: Number of SMs (Streaming Multiprocessors) to use for 
                 communication kernel (default: 32)
        num_xcds: Number of XCCs. If None, auto-detected from system (default: None)
    
    Example:
        >>> from iris.ccl import all_to_all, Config
        >>> config = Config(
        ...     block_size_m=128,
        ...     block_size_n=32,
        ...     swizzle_size=8,
        ...     comm_sms=64
        ... )
        >>> all_to_all(output_tensor, input_tensor, shmem, config=config)
    """
    block_size_m: int = 256
    block_size_n: int = 64
    swizzle_size: int = 6
    comm_sms: int = 32
    num_xcds: int = None
    
    def __post_init__(self):
        """Validate and auto-detect num_xcds if not set."""
        if self.num_xcds is None:
            self.num_xcds = iris.hip.get_num_xcc()
        
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

