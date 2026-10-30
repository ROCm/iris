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
        all_gather_variant: Variant for all-gather operation (default: None, which means
                           "persistent" for the CU kernels)
                           Options: "persistent", "partitioned", "tdm"
                           - "persistent": Each PID handles multiple tiles and sends to all ranks
                           - "partitioned": PIDs partitioned across ranks, eliminates inner loop
                           - "tdm": Gluon only, gfx1250 only. Moves tiles with the tensor data
                             movement engine: one HBM->LDS load and one LDS->peer store per rank
        all_reduce_variant: Variant for all-reduce operation (default: "atomic")
                           Options: "atomic", "ring", "two_shot", "one_shot", "spinlock"
        all_reduce_distribution: Distribution for two-shot all-reduce (default: 0)
                               0 for striding, 1 for block distribution
        all_reduce_num_rings: Number of concurrent rings to form in ring-based all-reduce (default: 1)
        all_reduce_ring_slice_n: Column slice size for ring reduce-scatter/all-gather
                                 (default: auto-set to block_size_n // world_size at runtime)
        reduce_scatter_variant: Variant for reduce-scatter operation (default: "two_shot")
                                Only "two_shot" is supported
        num_stages: Number of pipeline stages for the kernel (default: 1). For the TDM
                    all-gather, the number of tiles each workgroup keeps in LDS: loads run
                    num_stages - 1 tiles ahead of stores, and num_stages x block_size_m x
                    block_size_n elements must fit in LDS
        num_warps: Number of warps per workgroup (default: 4). For gluon kernels,
                   this also sets the warps of the BlockedLayout. The product of
                   the wavefront width (64 on CDNA3/4, 32 on CDNA5, read in-kernel)
                   and num_warps determines the minimum tile size
                   (block_size_m * block_size_n for flat-2D, or block_size_n for 1D).
        waves_per_eu: Waves per execution unit hint for occupancy (default: 0, auto)
        engine: What moves the data (default: "cu")
                Options: "cu", "sdma"
                - "cu": Compute-unit kernels load and store the data
                - "sdma": Copy engines move the data; needs a context created with
                  engines=("sdma",). all_gather only; use_gluon and all_gather_variant must be
                  left unset, and the tiling fields are unused

    Example:
        >>> import iris
        >>> from iris.ccl import Config
        >>> ctx = iris.iris()
        >>> config = Config(
        ...     block_size_m=128,
        ...     block_size_n=32,
        ...     swizzle_size=8,
        ...     comm_sms=64,
        ...     use_gluon=True
        ... )
        >>> ctx.ccl.all_to_all(output_tensor, input_tensor, config=config)

        >>> # All-reduce with ring variant
        >>> config = Config(all_reduce_variant="ring")
        >>> ctx.ccl.all_reduce(output_tensor, input_tensor, config=config)

        >>> # All-gather with partitioned variant
        >>> config = Config(all_gather_variant="partitioned")
        >>> ctx.ccl.all_gather(output_tensor, input_tensor, config=config)

        >>> # All-gather on the SDMA copy engines
        >>> ctx = iris.iris(engines=("sdma",))
        >>> config = Config(engine="sdma")
        >>> ctx.ccl.all_gather(output_tensor, input_tensor, config=config)
    """

    block_size_m: int = 32
    block_size_n: int = 64
    swizzle_size: int = 4
    comm_sms: int = 64
    num_xcds: int | None = None
    chunk_size: int | None = None
    use_gluon: bool = False
    all_gather_variant: str | None = None
    all_reduce_variant: str = "two_shot"
    all_reduce_distribution: int = 1
    all_reduce_num_rings: int = 1
    all_reduce_ring_slice_n: int | None = None
    reduce_scatter_variant: str = "two_shot"
    num_stages: int = 1
    num_warps: int = 4
    waves_per_eu: int = 0
    engine: str = "cu"

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
        if self.num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {self.num_stages}")
        if self.engine not in ["cu", "sdma"]:
            raise ValueError(f"engine must be one of: 'cu', 'sdma', got {self.engine}")
        if self.all_gather_variant not in [None, "persistent", "partitioned", "tdm"]:
            raise ValueError(
                f"all_gather_variant must be one of: 'persistent', 'partitioned', 'tdm', got {self.all_gather_variant}"
            )
        if self.all_gather_variant == "tdm" and not self.use_gluon:
            raise ValueError("all_gather_variant='tdm' is a Gluon kernel; set use_gluon=True")
        if self.all_reduce_variant not in ["atomic", "ring", "two_shot", "one_shot", "spinlock"]:
            raise ValueError(
                f"all_reduce_variant must be one of: 'atomic', 'ring', 'two_shot', 'one_shot', 'spinlock', got {self.all_reduce_variant}"
            )
        if self.all_reduce_distribution not in [0, 1]:
            raise ValueError(
                f"all_reduce_distribution must be 0 (striding) or 1 (block), got {self.all_reduce_distribution}"
            )
        if self.all_reduce_num_rings <= 0:
            raise ValueError(f"all_reduce_num_rings must be positive, got {self.all_reduce_num_rings}")
        if self.all_reduce_ring_slice_n is None:
            self.all_reduce_ring_slice_n = self.block_size_n
        if self.all_reduce_ring_slice_n <= 0:
            raise ValueError(f"all_reduce_ring_slice_n must be positive, got {self.all_reduce_ring_slice_n}")
        if self.block_size_n % self.all_reduce_ring_slice_n != 0:
            raise ValueError(
                f"all_reduce_ring_slice_n must divide block_size_n "
                f"(block_size_n={self.block_size_n}, slice={self.all_reduce_ring_slice_n})"
            )
        if self.all_reduce_ring_slice_n & (self.all_reduce_ring_slice_n - 1):
            raise ValueError(f"all_reduce_ring_slice_n must be a power of two, got {self.all_reduce_ring_slice_n}")

        # Validate reduce_scatter_variant
        if self.reduce_scatter_variant != "two_shot":
            raise ValueError(f"reduce_scatter_variant must be 'two_shot', got '{self.reduce_scatter_variant}'")

        if self.num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {self.num_warps}")
