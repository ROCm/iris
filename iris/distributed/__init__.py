# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris ProcessGroup backend for torch.distributed.

Registers "iris" as a backend so users can do:

    import iris.distributed
    dist.init_process_group(backend="iris")

All collective operations (all_reduce, all_gather, reduce_scatter,
all_to_all, barrier) are routed through iris CCL kernels on the
symmetric heap, bypassing NCCL entirely.
"""

from iris.distributed.process_group import IrisProcessGroup, _create_iris_backend

import torch.distributed as dist

dist.Backend.register_backend("iris", _create_iris_backend, devices="cuda")

__all__ = ["IrisProcessGroup"]
