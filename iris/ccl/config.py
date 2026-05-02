# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Configuration structures for iris-ccl collective operations.
"""

from dataclasses import dataclass, field, fields, replace
import iris


class _AutoTuneSentinel:
    """Sentinel value indicating a Config field should be auto-tuned.

    Usage:
        >>> from iris.ccl import AUTOTUNE
        >>> config = Config(block_size_m=AUTOTUNE, comm_sms=64)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "AUTOTUNE"

    def __bool__(self):
        raise TypeError("AUTOTUNE sentinel cannot be used as a boolean")


AUTOTUNE = _AutoTuneSentinel()


@dataclass
class Config:
    """
    Configuration parameters for iris-ccl collective operations.

    This configuration struct encapsulates common kernel parameters that can be
    set once and reused across multiple collective calls, similar to the
    origami config pattern from ROCm libraries.

    Fields set to AUTOTUNE will be automatically tuned on first call and cached.

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
        threads_per_warp: Threads per warp/wavefront (default: 64). Must match the
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

        >>> # Auto-tune all fields
        >>> from iris.ccl import AUTOTUNE
        >>> ctx.ccl.all_gather(output_tensor, input_tensor)  # all fields auto-tuned

        >>> # Partial auto-tune: fix comm_sms, tune everything else
        >>> config = Config(comm_sms=64, block_size_m=AUTOTUNE)
        >>> ctx.ccl.all_gather(output_tensor, input_tensor, config=config)
    """

    block_size_m: int = field(
        default=32,
        metadata={
            "search_space": [8, 16, 32, 64, 128],
        },
    )
    block_size_n: int = field(
        default=64,
        metadata={
            "search_space": [32, 64, 128, 256],
        },
    )
    swizzle_size: int = field(
        default=4,
        metadata={
            "search_space": [2, 4, 6, 8],
        },
    )
    comm_sms: int = field(
        default=64,
        metadata={
            "search_space": [32, 48, 64, 80, 96, 108],
        },
    )
    num_xcds: int | None = None
    chunk_size: int | None = None
    use_gluon: bool = False
    all_gather_variant: str = field(
        default="persistent",
        metadata={
            "search_space": ["persistent", "partitioned"],
            "collectives": ["all_gather"],
        },
    )
    all_reduce_variant: str = field(
        default="two_shot",
        metadata={
            "search_space": ["two_shot", "one_shot", "atomic"],
            "collectives": ["all_reduce"],
        },
    )
    all_reduce_distribution: int = field(
        default=1,
        metadata={
            "search_space": [0, 1],
            "collectives": ["all_reduce", "reduce_scatter"],
        },
    )
    all_reduce_num_rings: int = 1
    all_reduce_ring_slice_n: int | None = None
    reduce_scatter_variant: str = "two_shot"
    num_stages: int = field(
        default=1,
        metadata={
            "search_space": [1, 2],
        },
    )
    num_warps: int = field(
        default=4,
        metadata={
            "search_space": [2, 4, 8],
        },
    )
    threads_per_warp: int = 64
    waves_per_eu: int = field(
        default=0,
        metadata={
            "search_space": [0, 1, 2],
        },
    )

    def __post_init__(self):
        """Validate and auto-detect num_xcds if not set."""
        # If any field is AUTOTUNE, skip validation — it will be resolved
        # and validated after autotuning fills in concrete values.
        if self.get_autotune_fields():
            return

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
        if self.all_gather_variant not in ["persistent", "partitioned"]:
            raise ValueError(
                f"all_gather_variant must be one of: 'persistent', 'partitioned', got {self.all_gather_variant}"
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

        if self.threads_per_warp not in (32, 64):
            raise ValueError(f"threads_per_warp must be 32 (NVIDIA) or 64 (AMD), got {self.threads_per_warp}")
        if self.num_warps <= 0:
            raise ValueError(f"num_warps must be positive, got {self.num_warps}")

    def get_autotune_fields(self) -> list[str]:
        """Return names of fields set to AUTOTUNE."""
        return [f.name for f in fields(self) if getattr(self, f.name) is AUTOTUNE]

    def with_resolved(self, **kwargs) -> "Config":
        """Return a copy with specified fields overridden.

        This creates a new Config with the given fields replaced. The new
        Config goes through normal __post_init__ validation, so all
        resolved values must be concrete.
        """
        return replace(self, **kwargs)

    @classmethod
    def autotune(cls, **fixed_fields) -> "Config":
        """Create a Config that auto-tunes all fields except those specified.

        Fields passed as keyword arguments are pinned to the given values.
        All other fields are marked as AUTOTUNE and will be searched during
        the first collective call.

        Args:
            **fixed_fields: Fields to pin (e.g., comm_sms=64).

        Returns:
            Config with unspecified fields set to AUTOTUNE.

        Example:
            >>> Config.autotune(comm_sms=64)        # fix comm_sms, tune the rest
            >>> Config.autotune()                    # tune everything
            >>> Config.autotune(block_size_m=32, comm_sms=64)  # fix two, tune the rest
        """
        all_field_names = {f.name for f in fields(cls)}
        invalid = set(fixed_fields) - all_field_names
        if invalid:
            raise TypeError(f"Unknown Config fields: {invalid}")

        kwargs = {f.name: AUTOTUNE for f in fields(cls) if f.name not in fixed_fields}
        kwargs.update(fixed_fields)
        # num_xcds and chunk_size are always auto-derived, never tuned
        kwargs["num_xcds"] = None
        kwargs["chunk_size"] = None
        return cls(**kwargs)

    @classmethod
    def get_tunable_fields(cls, collective: str) -> list[str]:
        """Return field names that are tunable for a given collective."""
        result = []
        for f in fields(cls):
            meta = f.metadata
            if "search_space" not in meta:
                continue
            collectives = meta.get("collectives")
            if collectives is None or collective in collectives:
                result.append(f.name)
        return result

    @classmethod
    def get_search_space(cls, collective: str) -> dict[str, list]:
        """Return ``{field_name: [candidate_values]}`` for a given collective."""
        result = {}
        for f in fields(cls):
            meta = f.metadata
            if "search_space" not in meta:
                continue
            collectives = meta.get("collectives")
            if collectives is not None and collective not in collectives:
                continue
            result[f.name] = meta["search_space"]
        return result
