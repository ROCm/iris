# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""Resolve the GPU runtime that is ALREADY mapped into this process.

torch ships its own ROCm now -- the `rocm` pip SDK installs a full runtime
under site-packages/_rocm_sdk_core/lib -- and `import torch` maps that HIP and
its matching HSA before iris loads anything. If iris then resolves
libamdhip64 from a *different* ROCm (which is what a bare soname or a
hardcoded /opt/rocm path does), two installs end up in one process and the
loader fails on a versioned HSA symbol:

    /opt/rocm/lib/libamdhip64.so: undefined symbol:
    hsa_amd_interop_map_buffer_with_size, version ROCR_1

That reads like a broken ROCm install. It is really two working ones. Binding
to the copy already in the process keeps everything on a single runtime, and
is also the only choice that stays correct when the two differ by a major
version, as they do when torch is built against HIP 7.x on a ROCm 10 host.

Returns None off Linux or when nothing matches, so every caller keeps its
existing fallback chain.
"""

from typing import Optional

__all__ = ["in_process_library"]


def in_process_library(stem: str) -> Optional[str]:
    """Absolute path of a mapped shared object whose name contains `stem`."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.rsplit(" ", 1)[-1].strip()
                if stem in path and path.startswith("/"):
                    return path
    except OSError:
        pass
    return None
