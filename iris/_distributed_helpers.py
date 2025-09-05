# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import numpy as np


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
    assert len(data.shape) == 1, "Only 1D arrays are supported."
    
    world_size = dist.get_world_size()
    
    # Convert to tensor and gather
    data_tensor = torch.from_numpy(data)
    gathered_tensors = [torch.zeros_like(data_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, data_tensor)
    
    # Convert back to numpy and reshape
    gathered_data = torch.stack(gathered_tensors).numpy()
    return gathered_data


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
    
    if rank == root:
        assert value is not None, "Root must provide a value."
        value = np.array(value)
        dtype = value.dtype
    else:
        value = None
        dtype = None
    
    # Broadcast dtype first
    dtype_obj = [dtype]
    dist.broadcast_object_list(dtype_obj, src=root)
    dtype = dtype_obj[0]
    
    # Prepare value tensor
    if rank != root:
        value = np.empty(1, dtype=dtype)
    else:
        value = np.array([value], dtype=dtype)
    
    # Broadcast the actual value
    value_tensor = torch.from_numpy(value)
    dist.broadcast(value_tensor, src=root)
    
    return value_tensor.numpy()[0]


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
    return None, rank, world_size  # None for communicator since PyTorch uses global state