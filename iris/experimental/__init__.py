# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris Experimental Features

This module contains experimental features for Iris that may not be fully stable
or may undergo breaking changes in future releases.

Current experimental features:
- iris_gluon: Gluon-based implementation using @aggregate and @gluon.jit
- iris_rdma: InfiniBand RDMA backend for multi-node communication

Usage (Gluon):
    >>> import iris.experimental.iris_gluon as iris_gl
    >>> from triton.experimental import gluon
    >>> from triton.experimental.gluon import language as gl
    >>>
    >>> # Host side
    >>> ctx = iris_gl.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>>
    >>> # Device side
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     ctx.load(buffer, 1)

Usage (RDMA):
    >>> import iris.experimental.iris_rdma as iris_rdma
    >>> import torch.distributed as dist
    >>>
    >>> # Initialize PyTorch Distributed first
    >>> dist.init_process_group(backend='nccl')
    >>>
    >>> # Host side
    >>> ctx = iris_rdma.iris(heap_size=2**30)
    >>> device_ctx = ctx.get_device_context()
    >>>
    >>> # Device side
    >>> @triton.jit
    >>> def kernel(dst_ptr, data, device_ctx, dst_rank):
    >>>     iris_rdma.put(dst_ptr, data, dst_rank, device_ctx, mask)
"""

from . import iris_gluon

# Try to import iris_rdma (optional, requires InfiniBand)
try:
    from . import iris_rdma
    _has_rdma = True
except ImportError as e:
    _has_rdma = False
    import warnings
    warnings.warn(
        f"iris_rdma not available: {e}\n"
        "InfiniBand RDMA support requires libibverbs-dev and building with CMake.",
        ImportWarning
    )

__all__ = ["iris_gluon"]
if _has_rdma:
    __all__.append("iris_rdma")
