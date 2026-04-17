# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


import torch
import torch.distributed as dist
import numpy as np
import triton
import triton.language as tl


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
    # Gloo doesn't support uint64, so use object collective for uint64 with gloo
    # For int64 with gloo, we can use tensor collective (gloo supports int64)
    use_tensor_collective = (backend != "nccl" or _nccl_dtype_supported(data_tensor)) and not (
        backend == "gloo" and data_tensor.dtype == torch.uint64
    )

    if use_tensor_collective:
        data_tensor = data_tensor.to(device)
        gathered_tensors = [torch.empty_like(data_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, data_tensor)
        stacked = torch.stack(gathered_tensors, dim=0)
        cpu_tensor = stacked.to("cpu")
        result = cpu_tensor.numpy()
        return result
    else:
        # Fallback for NCCL-unsupported dtypes or gloo with uint64 (e.g., uint64/bool/etc.)
        obj_list = [None for _ in range(world_size)]
        # Use object collective (works across backends)
        dist.all_gather_object(obj_list, data)
        # Ensure uniform shapes and stack
        result = np.stack(obj_list, axis=0)
        return result


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


def extract_group_info(group, rank, num_ranks):
    """
    Extract rank and stride information for a process group.

    Args:
        group: ProcessGroup or None. If None, uses the provided rank/num_ranks
            as the default (all-ranks) group.
        rank: Global rank of the current process.
        num_ranks: Total number of ranks in the default group.

    Returns:
        Tuple of (rank_in_group, rank_global, world_size, rank_start, rank_stride):
            - rank_in_group: Rank within the group (0-indexed)
            - rank_global: Global rank of this process
            - world_size: Number of ranks in the group
            - rank_start: Starting global rank of the group
            - rank_stride: Stride between consecutive ranks in the group

    Examples:
        >>> # group=None: all ranks [0,1,2,3], current global rank is 2
        >>> extract_group_info(None, 2, 4)
        (2, 2, 4, 0, 1)

        >>> # DP group: strided ranks [0,4,8,12], current global rank is 8
        >>> extract_group_info(dp_group, 8, 16)
        (2, 8, 4, 0, 4)
    """
    if group is None:
        return rank, rank, num_ranks, 0, 1

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized to use ProcessGroup. "
            "Call torch.distributed.init_process_group() first."
        )

    group_ranks = dist.get_process_group_ranks(group)
    world_size = len(group_ranks)
    rank_global = rank

    if rank_global not in group_ranks:
        raise RuntimeError(
            f"Rank {rank_global} is not part of the specified process group. Group contains ranks: {group_ranks}"
        )

    rank_in_group = group_ranks.index(rank_global)

    if len(group_ranks) > 1:
        strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
        if not all(s == strides[0] for s in strides):
            raise NotImplementedError(
                f"Non-strided process groups are not yet supported. "
                f"Group ranks: {group_ranks}. "
                f"Please use groups with uniform stride (e.g., [0,1,2,3] or [0,4,8,12])."
            )
        rank_start = group_ranks[0]
        rank_stride = strides[0]
        if rank_stride == 0:
            raise ValueError(
                f"Invalid process group: rank_stride is 0, indicating duplicate ranks. "
                f"Group ranks: {group_ranks}. "
                f"Each rank must appear exactly once in a process group."
            )
    else:
        rank_start = group_ranks[0]
        rank_stride = 1

    return rank_in_group, rank_global, world_size, rank_start, rank_stride


def distributed_barrier(group=None):
    """
    Synchronization barrier using PyTorch distributed.

    Args:
        group (ProcessGroup, optional): The process group to synchronize.
            If None, uses the default process group (all ranks).
    """
    if not dist.is_initialized():
        raise RuntimeError("PyTorch distributed is not initialized")
    dist.barrier(group=group)


@triton.jit
def _translate_ptr(ptr, from_rank, to_rank, heap_bases):
    """Translate a pointer from one rank's address space to another's."""
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    offset = tl.cast(ptr, tl.uint64) - from_base
    translated_ptr = tl.cast(tl.cast(to_base, tl.pointer_type(tl.int8)) + offset, ptr.dtype)
    return translated_ptr


@triton.jit
def _device_barrier_kernel(
    flags_ptr,
    iris_rank,
    world_size: tl.constexpr,
    rank_start,
    rank_stride,
    heap_bases,
    MAX_SPINS: tl.constexpr = 1_000_000_000,
):
    """
    Device-side barrier using atomic operations on the symmetric heap.
    CUDA graph capturable.

    Stateless w.r.t. host-side epoch tracking: there is no CPU-side epoch
    counter. Each rank's flag on the heap serves as its own epoch counter,
    managed entirely by the GPU via atomic_add. A persistent per-group flags
    tensor is cached in ``_device_barrier_state``.

    Launched with grid=(1,). A single CTA:
    1. Atomically increments its own flag (atomic_add, release)
    2. Serially polls each remote rank's flag for the same value (acquire)
    """
    # Increment own flag and determine target
    own_flag_ptr = flags_ptr + iris_rank
    own_translated = _translate_ptr(own_flag_ptr, iris_rank, iris_rank, heap_bases)
    old = tl.atomic_add(own_translated, 1, sem="release", scope="sys")
    target = old + 1

    # Poll each remote rank serially
    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            remote_flag_ptr = flags_ptr + remote_rank
            remote_translated = _translate_ptr(remote_flag_ptr, iris_rank, remote_rank, heap_bases)
            spin_count = 0
            while (
                tl.atomic_cas(
                    remote_translated,
                    target,
                    target,
                    sem="acquire",
                    scope="sys",
                )
                < target
            ):
                spin_count += 1
                tl.device_assert(spin_count < MAX_SPINS, "device_barrier: timeout")


def distributed_device_barrier(flags, group, rank, num_ranks, heap_bases):
    """
    Device-side barrier using atomic operations on the symmetric heap.
    CUDA graph capturable.

    Unlike ``distributed_barrier`` which uses host-side ``torch.distributed.barrier()``,
    this launches a single-CTA Triton kernel that synchronizes via
    device-side atomics, making it safe to use during CUDA graph capture.

    Stateless w.r.t. host-side epoch tracking: each rank's flag on the
    symmetric heap serves as its own epoch counter, managed entirely by
    the GPU via atomic_add. A persistent per-group flags tensor is cached
    in ``_device_barrier_state``.

    Args:
        flags: int32 tensor on symmetric heap, one element per rank.
        group: ProcessGroup or None. If None, uses all ranks.
        rank: Global rank of this process.
        num_ranks: Total number of ranks in the default group.
        heap_bases: Tensor of heap base addresses for all ranks.
    """
    _, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, rank, num_ranks)
    _device_barrier_kernel[(1,)](
        flags,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        heap_bases,
    )


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
