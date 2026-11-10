# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from iris.host.platform import hip
from iris.host.tracing.core import Tracing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm GPUs")
def test_trace_metadata_describes_the_ranks_gpu_not_the_gpu_numbered_like_the_rank():
    # A rank on a second host has a rank at or above the local GPU count, so the rank is not a valid device index.
    num_gpus = torch.cuda.device_count()
    gpu_id = num_gpus - 1
    tracing = Tracing(SimpleNamespace(cur_rank=num_gpus + gpu_id, gpu_id=gpu_id))

    metadata = tracing._collect_system_metadata()

    assert metadata["gpu_device_name"] == torch.cuda.get_device_name(gpu_id)
    assert metadata["gpu_arch"] == hip.get_arch_string(gpu_id)
    assert metadata["gpu_cu_count"] == hip.get_cu_count(gpu_id)
    assert metadata["gpu_num_xcc"] == hip.get_num_xcc(gpu_id)
