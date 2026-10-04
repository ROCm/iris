# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Configuration structures for iris-ccl collective operations.
"""

from dataclasses import dataclass
import iris


def _is_gfx1250() -> bool:
    """True when the current device is gfx1250.

    The elements-per-thread guard below is a gfx1250 measurement and is not
    known to hold elsewhere -- gfx942 runs the same ratios fine. Any failure to
    identify the device returns False, so the guard stays off rather than
    rejecting a config on a part it was never measured on.
    """
    try:
        import torch

        return "gfx1250" in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return False


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
        all_gather_variant: Variant for all-gather operation (default: "persistent")
                           Options: "persistent", "partitioned"
                           - "persistent": Each PID handles multiple tiles and sends to all ranks
                           - "partitioned": PIDs partitioned across ranks, eliminates inner loop
        all_reduce_variant: Variant for all-reduce operation (default: "atomic")
                           Options: "atomic", "ring", "two_shot", "one_shot", "spinlock"
        all_reduce_distribution: Distribution for two-shot all-reduce (default: 0)
                               0 for striding, 1 for block distribution
        all_reduce_num_rings: Number of concurrent rings to form in ring-based all-reduce (default: 1)
        all_reduce_ring_slice_n: Column slice size for ring reduce-scatter/all-gather
                                 (default: auto-set to block_size_n // world_size at runtime)
        reduce_scatter_variant: Variant for reduce-scatter operation (default: "two_shot")
                                Only "two_shot" is supported
        num_stages: Number of pipeline stages for the kernel (default: 1)
        num_warps: Number of warps per workgroup (default: 4). For gluon kernels,
                   this also sets WARPS_PER_CTA in the BlockedLayout. The product
                   threads_per_warp * num_warps determines the minimum tile size
                   (block_size_m * block_size_n for flat-2D, or block_size_n for 1D).
        threads_per_warp: Threads per warp/wavefront. Defaults to the device's
            actual warp_size; both 32 and 64 occur on AMD. Must match the
                          hardware wavefront size: 64 for AMD GPUs, 32 for NVIDIA.
                          Used by gluon kernels to construct BlockedLayout for
                          vectorized memory access.
        waves_per_eu: Waves per execution unit hint for occupancy (default: 0, auto)

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
    """

    block_size_m: int = 32
    block_size_n: int = 64
    swizzle_size: int = 4
    comm_sms: int = 64
    num_xcds: int | None = None
    chunk_size: int | None = None
    use_gluon: bool = False
    all_gather_variant: str = "persistent"
    all_to_all_variant: str = "default"
    all_reduce_variant: str = "two_shot"
    all_reduce_distribution: int = 1
    all_reduce_num_rings: int = 1
    all_reduce_ring_slice_n: int | None = None
    reduce_scatter_variant: str = "two_shot"
    num_stages: int = 1
    num_warps: int = 4
    threads_per_warp: int | None = None
    waves_per_eu: int = 0

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
        if self.all_to_all_variant not in ["default", "tdm"]:
            raise ValueError(
                f"all_to_all_variant must be 'default' or 'tdm' (gluon only), got {self.all_to_all_variant}"
            )
        if self.all_gather_variant not in ["persistent", "partitioned", "tdm"]:
            raise ValueError(
                "all_gather_variant must be one of: 'persistent', 'partitioned', "
                f"'tdm' (gluon only), got {self.all_gather_variant}"
            )
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

        if self.threads_per_warp is None:
            # Do NOT assume 64 on AMD. CDNA is 64, but gfx1250 is an AMD part
            # with a 32-wide wavefront, and assuming 64 silently halves the
            # thread count a tile is spread over -- which doubles elements per
            # thread and pushes configs that are fine on gfx942 into register
            # pressure and illegal memory accesses.
            try:
                import torch

                self.threads_per_warp = torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size
            except Exception:
                self.threads_per_warp = 64
        if self.threads_per_warp not in (32, 64):
            raise ValueError(
                f"threads_per_warp must be 32 or 64, got {self.threads_per_warp}. "
                "Both occur on AMD: CDNA is 64, gfx1250 is 32."
            )
        if self.num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {self.num_warps}")

        # A tile spread over too few threads faults ON gfx1250. Measured there
        # (world=4, all_reduce two_shot, fp16): 16 elements/thread is fine and
        # 32 faults with an illegal memory access, for every combination of
        # block_size_n and num_warps landing on those ratios -- bn=256/nw=8 and
        # bn=512/nw=16 both fault, bn=256/nw=16 and bn=512/nw=32 both pass.
        #
        # SCOPED TO gfx1250 DELIBERATELY. An earlier version of this check was
        # unconditional and broke CI on gfx942, where the gluon all_gather
        # tests run 32x256 at num_warps=4 -- 32 elements/thread on a 64-wide
        # wavefront -- and have always passed. That is direct evidence the
        # threshold does not generalise, so the guard must not either. The
        # measurement only ever covered gfx1250.
        #
        # Raise rather than let it reach the GPU: the failure surfaces
        # asynchronously as "illegal memory access" at an unrelated later
        # synchronize, which is very hard to trace back to tile shape.
        #
        # The TDM engine stages a tile through LDS rather than registers, so
        # the limit does not apply to it. That exemption keys off an all_gather
        # field, so a Config carrying all_gather_variant='tdm' is only safe for
        # all_gather -- reusing it for a register-path collective would skip a
        # check that collective still needs.
        tdm_path = self.use_gluon and (self.all_gather_variant == "tdm" or self.all_to_all_variant == "tdm")
        threads = self.num_warps * self.threads_per_warp
        per_thread = (self.block_size_m * self.block_size_n) / threads
        if per_thread >= 32 and not tdm_path and _is_gfx1250():
            raise ValueError(
                f"block_size_m*block_size_n ({self.block_size_m}*{self.block_size_n}"
                f" = {self.block_size_m * self.block_size_n}) over num_warps*"
                f"threads_per_warp ({self.num_warps}*{self.threads_per_warp}"
                f" = {threads}) is {per_thread:.0f} elements/thread; >= 32 faults "
                f"with an illegal memory access. Raise num_warps to "
                f"{self.num_warps * 2} or halve block_size_n."
            )
