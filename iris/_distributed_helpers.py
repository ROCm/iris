# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import numpy as np


def _infer_device():
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")
    try:
        backend = str(dist.get_backend()).lower()
    except Exception:
        backend = "gloo"
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _nccl_dtype_supported(t: torch.Tensor) -> bool:
    """Conservative whitelist for NCCL tensor dtypes."""
    supported = {
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    }
    # bfloat16 is commonly supported in recent stacks; include if available
    if hasattr(torch, "bfloat16"):
        supported.add(torch.bfloat16)
    return t.dtype in supported


def distributed_allgather(data):
    """
    All-gather operation using PyTorch distributed.

    Args:
        data: 1D numpy array to gather across all ranks

    Returns:
        2D numpy array with shape (world_size, len(data))
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")

    data = np.asarray(data)
    assert data.ndim == 1, "Only 1D arrays are supported."

    world_size = dist.get_world_size()
    device = _infer_device()
    backend = str(dist.get_backend()).lower()

    # Fast path: tensor all_gather if dtype is NCCL-supported or backend != nccl
    data_tensor = torch.from_numpy(data)
    use_tensor_collective = backend != "nccl" or _nccl_dtype_supported(data_tensor)

    if use_tensor_collective:
        data_tensor = data_tensor.to(device)
        gathered_tensors = [torch.empty_like(data_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, data_tensor)
        return torch.stack(gathered_tensors, dim=0).to("cpu").numpy()

    # Fallback for NCCL-unsupported dtypes (e.g., uint64/bool/etc.)
    obj_list = [None for _ in range(world_size)]
    # Use object collective (works across backends)
    dist.all_gather_object(obj_list, data)
    # Ensure uniform shapes and stack
    return np.stack(obj_list, axis=0)


def distributed_allgather_multidim(data):
    """
    All-gather operation for multi-dimensional tensors using PyTorch distributed.
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")

    world_size = dist.get_world_size()
    device = _infer_device()

    input_tensor = torch.as_tensor(data).to(device)

    tensor_list = [torch.empty_like(input_tensor) for _ in range(world_size)]

    dist.all_gather(tensor_list, input_tensor)

    stacked_tensor = torch.stack(tensor_list, dim=0)
    reshaped_tensor = stacked_tensor.view(world_size, -1)

    return reshaped_tensor.cpu().numpy()


def distributed_broadcast_scalar(value=None, root=0):
    """
    Broadcast a scalar value from root to all ranks.

    Args:
        value: Value to broadcast (only used on root rank)
        root: Root rank to broadcast from

    Returns:
        Broadcasted value
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")

    rank = dist.get_rank()
    device = _infer_device()
    backend = str(dist.get_backend()).lower()

    # First agree on dtype (numpy dtype object)
    if rank == root:
        if value is None:
            raise ValueError("Root must provide a value.")
        np_val = np.array(value)  # captures dtype
        dtype = np_val.dtype
    else:
        np_val = None
        dtype = None

    dtype_obj = [dtype]
    dist.broadcast_object_list(dtype_obj, src=root)
    dtype = dtype_obj[0]

    # If NCCL can't handle this dtype, just broadcast the object directly.
    if backend == "nccl":
        # Try a quick check using a tiny tensor of the dtype
        try:
            torch_dtype = torch.from_numpy(np.array(0, dtype=dtype)).dtype
            dummy = torch.empty((), dtype=torch_dtype)
            if not _nccl_dtype_supported(dummy):
                obj = [value if rank == root else None]
                dist.broadcast_object_list(obj, src=root)
                return obj[0]
        except (TypeError, ValueError):
            # Dtype not supported by torch (e.g., str, object), use object broadcast
            obj = [value if rank == root else None]
            dist.broadcast_object_list(obj, src=root)
            return obj[0]

    # Tensor path: create a 0-D tensor, broadcast on the selected device
    if rank != root:
        np_val = np.empty((), dtype=dtype)
    val_t = torch.from_numpy(np_val).to(device)
    dist.broadcast(val_t, src=root)
    return val_t.to("cpu").item()


def distributed_broadcast_tensor(value_to_broadcast=None, root=0):
    """
    Broadcast a tensor/array from root to all ranks.

    Args:
        value_to_broadcast: Tensor or array to broadcast (only used on root rank)
        root: Root rank to broadcast from

    Returns:
        Broadcasted numpy array
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")

    rank = dist.get_rank()
    device = _infer_device()
    backend = str(dist.get_backend()).lower()

    if rank == root:
        if value_to_broadcast is None:
            raise ValueError("Root must provide a value to broadcast.")
        tensor = torch.as_tensor(value_to_broadcast)
        metadata = [tensor.shape, tensor.dtype]
    else:
        metadata = [None, None]
        tensor = None

    dist.broadcast_object_list(metadata, src=root)
    shape, dtype = metadata

    if rank != root:
        tensor = torch.empty(shape, dtype=dtype)

    use_tensor_collective = backend != "nccl" or _nccl_dtype_supported(tensor)

    if use_tensor_collective:
        tensor = tensor.to(device)
        dist.broadcast(tensor, src=root)
        return tensor.to("cpu").numpy()
    else:
        if rank == root:
            obj = [np.asarray(value_to_broadcast)]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=root)
        return obj[0]


def distributed_barrier():
    """
    Synchronization barrier using PyTorch distributed.
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")
    dist.barrier()


def init_distributed():
    """
    Initialize PyTorch distributed and return communicator info.

    Returns:
        tuple: (communicator_placeholder, rank, world_size)
        Note: communicator_placeholder is None since PyTorch distributed
              uses global state rather than explicit communicator objects
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized. Call dist.init_process_group() first.")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return None, rank, world_size
