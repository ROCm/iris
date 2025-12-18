# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris RDMA: Experimental InfiniBand RDMA Backend for Multi-Node Communication

This module provides InfiniBand RDMA support for multi-node communication in Iris.
Unlike the main Iris which uses HIP IPC for intra-node GPU communication, this backend
enables inter-node communication via RDMA over InfiniBand.

Key Features:
- InfiniBand Queue Pair (QP) setup and management
- Symmetric heap with RDMA memory registration
- RDMA put/get operations in Triton kernels
- PyTorch Distributed integration for bootstrapping

Example:
    >>> import iris.experimental.iris_rdma as iris_rdma
    >>> import torch.distributed as dist
    >>> 
    >>> # Initialize PyTorch Distributed
    >>> dist.init_process_group(backend='nccl')
    >>> 
    >>> # Create RDMA context
    >>> ctx = iris_rdma.iris(heap_size=2**30)  # 1GB heap
    >>> device_ctx = ctx.get_device_context()  # For passing to Triton kernels
    >>> 
    >>> @triton.jit
    >>> def kernel(dst_ptr, data, device_ctx, dst_rank, BLOCK_SIZE: tl.constexpr):
    >>>     pid = tl.program_id(0)
    >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    >>>     
    >>>     # RDMA put to remote rank
    >>>     iris_rdma.put(dst_ptr + offsets, data, dst_rank, device_ctx)
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import numpy as np
import sys
import os

# Import the C++ backend module
try:
    from . import _iris_rdma_backend as backend
except ImportError:
    raise ImportError(
        "Iris RDMA backend not available. "
        "Make sure the module is built with InfiniBand support. "
        "Set IRIS_RDMA_DEBUG=1 for more information."
    )

# Import logging
from ..logging import logger


class IrisRDMA:
    """
    Main Iris RDMA class for multi-node RDMA operations.
    
    This class provides a unified interface for RDMA-based communication
    across multiple nodes using InfiniBand.
    
    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)
        process_group: PyTorch distributed process group (default: WORLD)
        device_name (str): InfiniBand device name (default: auto-detect)
    
    Example:
        >>> ctx = iris_rdma.iris(heap_size=2**31)  # 2GB heap
        >>> print(f"Rank {ctx.rank} of {ctx.world_size}")
        >>> buffer = ctx.zeros(1024, dtype=torch.float32)
    """
    
    def __init__(self, heap_size=1 << 30, process_group=None, queue_size=512):
        # Check if distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "PyTorch distributed must be initialized. "
                "Call torch.distributed.init_process_group() first."
            )
        
        if process_group is None:
            process_group = dist.group.WORLD
        
        # Get rank and world size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device_id = self.rank % torch.cuda.device_count()
        self.device = f"cuda:{self.device_id}"
        
        torch.cuda.set_device(self.device_id)
        
        # Create torch_bootstrap
        self._bootstrap = backend.torch_bootstrap(process_group)
        
        # Allocate symmetric heap (CPU pinned memory for now)
        # TODO: Support GPU memory with GPUDirect RDMA
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        
        # Create GPU memory pool
        self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)
        
        self._manager = backend.rdma_proxy(self._bootstrap, self.memory_pool, queue_size)
        self._manager.start_proxy_thread()
        
        self._backend = self._manager
        
        logger.info(f"[Rank {self.rank}] Using rdma_proxy with queue (size={queue_size})")
        
        self.remote_heap_bases = []
        for i in range(self.world_size):
            self.remote_heap_bases.append(self._manager.get_remote_heap_base(i))
        
        logger.info(f"[Rank {self.rank}] Iris RDMA initialized: heap_size={heap_size}, "
                   f"heap_base={self._manager.get_heap_base():#x}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, '_manager') and self._manager is not None:
            self._manager.stop_proxy_thread()
    
    def get_heap_base(self):
        """Get local heap base address"""
        return self._manager.get_heap_base()
    
    def get_queue_ptr(self):
        """Get queue pointer for Triton kernels"""
        return self._manager.get_queue_ptr()
    
    def get_device_context(self):
        """
        Get device context tensor for passing to Triton kernels.
        
        The context tensor encodes:
        - [0]: current rank
        - [1]: world size
        - [2]: queue pointer (for enqueueing RDMA operations)
        - [3:]: heap base addresses for all ranks
        
        Returns:
            torch.Tensor: Device context tensor (on GPU)
        
        Example:
            >>> ctx = iris_rdma.iris()
            >>> device_ctx = ctx.get_device_context()
            >>> # Pass device_ctx to Triton kernel
        """
        # Create context tensor: [rank, world_size, queue_ptr, heap_base_0, heap_base_1, ...]
        context_size = 3 + self.world_size
        context = torch.zeros(context_size, dtype=torch.int64, device=self.device)
        
        context[0] = self.rank
        context[1] = self.world_size
        context[2] = self.get_queue_ptr()
        
        for i in range(self.world_size):
            context[3 + i] = self.remote_heap_bases[i]
        
        return context
    
    def zeros(self, *size, dtype=torch.float32, device=None):
        """
        Allocate and initialize a tensor with zeros in the symmetric heap.
        
        Args:
            *size: Tensor dimensions
            dtype: Data type (default: torch.float32)
            device: Device placement (default: GPU for direct kernel access)
        
        Returns:
            torch.Tensor: Allocated tensor (on GPU by default)
        
        Example:
            >>> buffer = ctx.zeros(1024, 1024, dtype=torch.float32)
        """
        if device is None:
            device = self.device  # Use GPU by default (for GPUDirect)
        
        # Calculate size in bytes
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = int(np.prod(size))
        size_bytes = numel * elem_size
        
        # Align allocation
        aligned_offset = (self.heap_offset + self.alignment - 1) // self.alignment * self.alignment
        
        if aligned_offset + size_bytes > self.heap_size:
            raise RuntimeError(f"Heap exhausted: requested {size_bytes} bytes, "
                             f"available {self.heap_size - aligned_offset}")
        
        # Create tensor view into memory pool
        byte_offset = aligned_offset
        byte_end = byte_offset + size_bytes
        
        # Get the memory slice and view as the requested dtype
        memory_slice = self.memory_pool[byte_offset:byte_end]
        tensor = memory_slice.view(dtype).reshape(size)
        
        # Zero initialize
        tensor.zero_()
        
        # Update offset
        self.heap_offset = byte_end
        
        logger.debug(f"[Rank {self.rank}] Allocated tensor: size={size}, "
                    f"offset={byte_offset:#x}, ptr={tensor.data_ptr():#x}")
        
        return tensor
    
    def barrier(self):
        """
        Synchronize all ranks and drain RDMA queue.
        
        Waits for:
        1. All enqueued RDMA operations to complete (queue drains)
        2. All ranks to reach this barrier
        
        Example:
            >>> ctx.barrier()  # Wait for all ranks and RDMA completion
        """
        
        # First, synchronize with all GPUs
        torch.cuda.synchronize()
        
        # Then, synchronize with all ranks
        dist.barrier()
        
        # Finally, wait for queue to drain (all work processed)
        self.wait_queue_drain()
    
    def wait_queue_drain(self, timeout=30.0):
        """
        Wait for the CPU proxy thread to process all enqueued work items.
        
        Spins until queue is empty (head == tail), meaning all work has been
        processed and popped by the CPU proxy thread.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If queue doesn't drain within timeout
        """
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            # Check if queue is empty (head == tail)
            if self._manager.is_queue_empty():
                return
            
            # Small sleep to avoid burning CPU
            time.sleep(0.0001)  # 100 microseconds
        
        raise TimeoutError(f"Queue did not drain within {timeout}s")
    
    def rdma_put(self, dst_rank, local_addr, remote_addr, size):
        """
        Perform RDMA write (put) to remote rank.
        
        Args:
            dst_rank: Destination rank
            local_addr: Local buffer address (int or tensor.data_ptr())
            remote_addr: Remote buffer address (int)
            size: Size in bytes
        
        Returns:
            int: 0 on success, non-zero on error
        
        Example:
            >>> src = ctx.zeros(1024, dtype=torch.float32)
            >>> dst_addr = ctx.remote_heap_bases[1]  # Remote rank 1's heap
            >>> ctx.rdma_put(1, src.data_ptr(), dst_addr, src.numel() * 4)
        """
        if isinstance(local_addr, torch.Tensor):
            local_addr = local_addr.data_ptr()
        
        return self._backend.rdma_write(dst_rank, local_addr, remote_addr, size)
    
    def rdma_get(self, dst_rank, local_addr, remote_addr, size):
        """
        Perform RDMA read (get) from remote rank.
        
        Args:
            dst_rank: Source rank (destination of the QP)
            local_addr: Local buffer address (int or tensor.data_ptr())
            remote_addr: Remote buffer address (int)
            size: Size in bytes
        
        Returns:
            int: 0 on success, non-zero on error
        
        Example:
            >>> dst = ctx.zeros(1024, dtype=torch.float32)
            >>> src_addr = ctx.remote_heap_bases[1]  # Remote rank 1's heap
            >>> ctx.rdma_get(1, dst.data_ptr(), src_addr, dst.numel() * 4)
        """
        if isinstance(local_addr, torch.Tensor):
            local_addr = local_addr.data_ptr()
        
        return self._backend.rdma_read(dst_rank, local_addr, remote_addr, size)
    
    def poll_completion(self, dst_rank, max_completions=1):
        """
        Poll completion queue for RDMA operations.
        
        Args:
            dst_rank: Destination rank (to poll specific CQ)
            max_completions: Maximum number of completions to poll
        
        Returns:
            int: Number of completions polled (negative on error)
        
        Example:
            >>> ctx.rdma_put(1, src.data_ptr(), remote_addr, size)
            >>> while ctx.poll_completion(1) == 0:
            >>>     pass  # Wait for completion
        """
        return self._backend.poll_cq(dst_rank, max_completions)
    
    def __repr__(self):
        return f"<IrisRDMA rank={self.rank} world_size={self.world_size}>"


def iris(heap_size=1 << 30, process_group=None, queue_size=512):
    """
    Factory function to create Iris RDMA context.
    
    Args:
        heap_size (int): Size of the symmetric heap in bytes
        process_group: PyTorch distributed process group
        queue_size (int): Queue size for GPU->CPU RDMA operations
    
    Returns:
        IrisRDMA: RDMA context object
    
    Example:
        >>> import iris.experimental.iris_rdma as iris_rdma
        >>> ctx = iris_rdma.iris(heap_size=2**30)
    """
    return IrisRDMA(heap_size, process_group, queue_size)


#############################################################################
# Triton Device-Side APIs
#############################################################################

@triton.jit
def _translate(ptr, from_rank, to_rank, heap_bases):
    """
    Translate a pointer from one rank's address space to another.
    
    This implements the symmetric heap model where each rank has a heap at
    a different base address, but offsets are preserved across ranks.
    
    Args:
        ptr: Pointer in from_rank's address space
        from_rank: Source rank ID
        to_rank: Target rank ID
        heap_bases: Pointer to array of heap base addresses
    
    Returns:
        Translated pointer in to_rank's address space
    """
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    
    # Convert to int to compute difference
    ptr_int = ptr.to(tl.uint64)
    
    # Find the offset from from_rank heap
    offset = ptr_int - from_base
    
    # Byte cast for byte offset addition
    to_base_byte = to_base.to(tl.pointer_type(tl.int8))
    
    # Find the offset into the to_rank heap
    translated_ptr_byte = to_base_byte + offset
    
    # Cast back to original pointer type
    translated_ptr = translated_ptr_byte.to(ptr.dtype)
    
    return translated_ptr


@triton.jit
def _wait_for_completion(queue_ptr, queue_pos):
    """
    Wait for CPU to process a queue item.
    
    Spins until tail pointer advances past our queue position,
    indicating the CPU has processed and popped our item.
    
    Args:
        queue_ptr: Queue context pointer
        queue_pos: Queue position to wait for (returned from _enqueue_rdma_op)
    """
    state_ptr = queue_ptr.to(tl.pointer_type(tl.uint64))
    
    # Load tail pointer (offset 2 in QueueState)
    # Use volatile and cache modifier to prevent caching
    tail_ptr = tl.load(state_ptr + 2, cache_modifier=".cv", volatile=True)
    tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
    current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    # Spin until CPU advances tail past our position
    while queue_pos >= current_tail:
        tail_ptr = tl.load(state_ptr + 2, cache_modifier=".cv", volatile=True)
        tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
        current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')


@triton.jit
def _enqueue_rdma_op(dst_ptr, src_ptr, to_rank: tl.constexpr, op_code: tl.constexpr, queue_ptr, mask):
    """
    Internal: Enqueue an RDMA operation to the queue.
    
    Args:
        dst_ptr: Destination pointer on remote rank
        src_ptr: Source pointer (local address where data is stored in registered heap)
        to_rank: Target rank ID
        op_code: Operation type (1=PUT, 2=GET)
        queue_ptr: Queue pointer from device context
        mask: Triton mask for valid elements
    """
    # Queue structure (from queue.hpp):
    # struct QueueState {
    #   WorkItem* items;      // offset 0
    #   uint64_t* head;       // offset 8
    #   uint64_t* tail;       // offset 16
    #   uint64_t* tailCache;  // offset 24
    #   int32_t size;         // offset 32
    # };
    
    state_ptr = queue_ptr.to(tl.pointer_type(tl.uint64))
    
    # Load QueueState fields
    items_ptr = tl.load(state_ptr + 0)
    head_ptr = tl.load(state_ptr + 1)
    tail_ptr = tl.load(state_ptr + 2)
    
    # Load size (at offset 32 bytes = 4 * uint64)
    size_ptr = queue_ptr.to(tl.pointer_type(tl.int32))
    size = tl.load(size_ptr + 8)
    
    # Atomic increment head to reserve slot
    head_ptr_typed = head_ptr.to(tl.pointer_type(tl.uint64))
    prev_head = tl.atomic_add(head_ptr_typed, 1, sem='relaxed', scope='sys')
    
    # Wait for slot to be free: spin if prev_head >= size + *tail
    size_u64 = size.to(tl.uint64)
    tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
    current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    while prev_head >= size_u64 + current_tail:
        current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    # Calculate slot position
    slot_idx = prev_head % size_u64
    
    # WorkItem structure (32 bytes):
    # struct WorkItem {
    #   uint64_t dst_ptr;      // offset 0
    #   uint64_t src_ptr;      // offset 8
    #   uint32_t size_bytes;   // offset 16 - WRITE LAST as ready flag
    #   uint16_t rank;         // offset 20
    #   uint8_t  op_type;      // offset 22
    #   uint8_t  reserved;     // offset 23
    # };
    WORK_ITEM_SIZE_BYTES = 32
    
    slot_offset_bytes = slot_idx * WORK_ITEM_SIZE_BYTES
    
    # Get pointer to this work item
    items_ptr_u64 = items_ptr.to(tl.pointer_type(tl.uint64))
    slot_ptr_u64 = items_ptr_u64 + (slot_offset_bytes // 8).to(tl.int32)
    
    # Extract destination address (min of pointer block)
    dst_ptr_u64 = dst_ptr.to(tl.uint64)
    dst_ptr_val = tl.min(dst_ptr_u64, axis=0)
    
    # Extract source address (min of pointer block where data is stored)
    src_ptr_u64 = src_ptr.to(tl.uint64)
    src_ptr_val = tl.min(src_ptr_u64, axis=0)
    max_src_ptr = tl.max(src_ptr_u64, axis=0)
    
    # Infer element size from pointer type
    # src_ptr is a block of pointers with a specific element type (e.g., pointer<float32>)
    # The pointer dtype tells us the element type, which has a known size
    # Map Triton dtypes to their byte sizes
    ptr_dtype = src_ptr.dtype.element_ty  # Get the element type that the pointer points to
    
    # Get element size in bytes from the dtype
    # tl.float16 -> 2, tl.float32 -> 4, tl.float64 -> 8, etc.
    if ptr_dtype == tl.float16 or ptr_dtype == tl.bfloat16:
        element_size_bytes = 2
    elif ptr_dtype == tl.float32 or ptr_dtype == tl.int32 or ptr_dtype == tl.uint32:
        element_size_bytes = 4
    elif ptr_dtype == tl.float64 or ptr_dtype == tl.int64 or ptr_dtype == tl.uint64:
        element_size_bytes = 8
    elif ptr_dtype == tl.int8 or ptr_dtype == tl.uint8:
        element_size_bytes = 1
    elif ptr_dtype == tl.int16 or ptr_dtype == tl.uint16:
        element_size_bytes = 2
    else:
        # Default to 4 bytes for unknown types
        element_size_bytes = 4
    
    # Calculate total size in bytes
    # Count number of valid elements based on mask
    mask_int = mask.to(tl.int32)
    num_elements = tl.sum(mask_int, axis=0)
    size_bytes = (num_elements * element_size_bytes).to(tl.uint32)
    
    # Write header fields (but NOT size_bytes yet - it's the ready flag)
    # Write dst_ptr (offset 0)
    tl.store(slot_ptr_u64 + 0, dst_ptr_val)
    
    # Write src_ptr (offset 8)
    tl.store(slot_ptr_u64 + 1, src_ptr_val)
    
    # Write rank + op_type (offset 20-23)
    metadata = (to_rank & 0xFFFF) | ((op_code & 0xFF) << 16)
    slot_ptr_u32 = slot_ptr_u64.to(tl.pointer_type(tl.uint32))
    tl.store(slot_ptr_u32 + 5, metadata.to(tl.uint32))
    
    # Write size_bytes LAST as ready flag (offset 16)
    size_bytes_ptr = (slot_ptr_u32 + 4).to(tl.pointer_type(tl.uint32))
    tl.atomic_xchg(size_bytes_ptr, size_bytes, sem='release', scope='sys')
    
    # Return queue position for waiting
    return prev_head


@triton.jit
def put(dst_ptr, src_ptr, dst_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA put (write) operation from Triton kernel.
    
    Uses symmetric heap model: dst_ptr is in current rank's address space,
    and will be automatically translated to remote rank's address space.
    
    IMPORTANT: Data must be stored at src_ptr BEFORE calling this function.
    This avoids race conditions between GPU writes and CPU RDMA reads.
    The CPU proxy thread will dequeue and perform the actual RDMA write.
    
    Args:
        dst_ptr: Destination pointer in CURRENT rank's address space (symmetric heap)
        src_ptr: Source pointer (local address in registered heap) where data is already stored - can be block of pointers
        dst_rank: Target rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Example:
        >>> @triton.jit
        >>> def kernel(local_buffer, device_ctx, dst_rank, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     
        >>>     data = generate_data(offsets)
        >>>     src_ptrs = local_buffer + offsets
        >>>     dst_ptrs = local_buffer + offsets  # Same offset, symmetric heap!
        >>>     
        >>>     # Store data FIRST to avoid race condition
        >>>     tl.store(src_ptrs, data, mask=mask)
        >>>     
        >>>     # Then enqueue RDMA operation
        >>>     iris_rdma.put(dst_ptrs, src_ptrs, dst_rank, device_ctx, mask)
    """
    # Extract context fields
    # Context format: [rank, world_size, queue_ptr, heap_base_0, heap_base_1, ...]
    my_rank = tl.load(device_ctx + 0)
    queue_ptr = tl.load(device_ctx + 2)
    heap_bases = device_ctx + 3
    
    # Translate dst_ptr from current rank's address space to remote rank's
    translated_dst_ptr = _translate(dst_ptr, my_rank, dst_rank, heap_bases)
    
    # Enqueue PUT operation (op_code=1) with translated address
    _enqueue_rdma_op(translated_dst_ptr, src_ptr, dst_rank, 1, queue_ptr, mask)


@triton.jit
def get(dst_ptr, src_ptr, from_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA get (read) operation from Triton kernel.
    
    Uses symmetric heap model: src_ptr is in current rank's address space,
    and will be automatically translated to remote rank's address space.
    
    Enqueues a request to read data from remote rank via RDMA and WAITS for completion.
    The CPU proxy thread will dequeue, perform the RDMA read, then pop the item.
    This function spins until the tail pointer advances, then data is ready at dst_ptr.
    
    Args:
        dst_ptr: Local destination pointer where data will be written - can be block of pointers
        src_ptr: Source pointer in CURRENT rank's address space (symmetric heap)
        from_rank: Source rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Example:
        >>> @triton.jit
        >>> def kernel(local_buffer, device_ctx, from_rank, BLOCK_SIZE: tl.constexpr):
        >>>     pid = tl.program_id(0)
        >>>     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     
        >>>     src_ptrs = local_buffer + offsets  # Same offset in symmetric heap!
        >>>     dst_ptrs = local_buffer + offsets
        >>>     # RDMA read from remote rank - blocks until complete
        >>>     iris_rdma.get(dst_ptrs, src_ptrs, from_rank, device_ctx, mask)
        >>>     
        >>>     # Data is now ready at dst_ptrs, can use it immediately
        >>>     data = tl.load(dst_ptrs, mask=mask)
    """
    # Extract context fields
    # Context format: [rank, world_size, queue_ptr, heap_base_0, heap_base_1, ...]
    my_rank = tl.load(device_ctx + 0)
    queue_ptr = tl.load(device_ctx + 2)
    heap_bases = device_ctx + 3
    
    # Translate src_ptr from current rank's address space to remote rank's
    translated_src_ptr = _translate(src_ptr, my_rank, from_rank, heap_bases)
    
    # Enqueue GET operation (op_code=2)
    # For GET: translated_src_ptr is remote source, dst_ptr is local destination
    queue_pos = _enqueue_rdma_op(translated_src_ptr, dst_ptr, from_rank, 2, queue_ptr, mask)
    
    # Wait for CPU to complete the RDMA read
    _wait_for_completion(queue_ptr, queue_pos)
    
    # Data is now ready at dst_ptr (CPU has written it there via RDMA)


@triton.jit
def atomic_add(result_ptr, dst_ptr, add_value, dst_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA atomic fetch-and-add operation from Triton kernel.
    
    Atomically adds a value to remote memory and returns the original value.
    Uses symmetric heap model: dst_ptr is in current rank's address space,
    and will be automatically translated to remote rank's address space.
    
    Args:
        result_ptr: Local pointer where the original value will be stored
        dst_ptr: Destination pointer in CURRENT rank's address space (symmetric heap)
        add_value: Value to add (must be scalar uint64 or int64)
        dst_rank: Destination rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Note: Only supports 8-byte (uint64/int64) atomic operations.
    The result_ptr will contain the original value before the add.
    """
    # Extract context fields
    my_rank = tl.load(device_ctx + 0)
    queue_ptr = tl.load(device_ctx + 2)
    heap_bases = device_ctx + 3
    
    # Translate dst_ptr from current rank's address space to remote rank's
    translated_dst_ptr = _translate(dst_ptr, my_rank, dst_rank, heap_bases)
    
    # Enqueue ATOMIC_ADD operation (op_code=4)
    # For ATOMIC_ADD: result_ptr is local result buffer, translated_dst_ptr is remote target
    queue_pos = _enqueue_atomic_op(result_ptr, translated_dst_ptr, dst_rank, 4, 
                                     add_value, 0, queue_ptr, mask)
    
    # Wait for CPU to complete the atomic operation
    _wait_for_completion(queue_ptr, queue_pos)
    
    # Result is now ready at result_ptr (original value before add)


@triton.jit
def atomic_cas(result_ptr, dst_ptr, compare_value, swap_value, dst_rank: tl.constexpr, device_ctx, mask):
    """
    RDMA atomic compare-and-swap operation from Triton kernel.
    
    Atomically compares remote memory with expected value and swaps if equal.
    Returns the original value. Uses symmetric heap model.
    
    Args:
        result_ptr: Local pointer where the original value will be stored
        dst_ptr: Destination pointer in CURRENT rank's address space (symmetric heap)
        compare_value: Expected value (must be scalar uint64 or int64)
        swap_value: New value if comparison succeeds (must be scalar uint64 or int64)
        dst_rank: Destination rank ID (must be compile-time constant)
        device_ctx: Device context from iris_rdma.get_device_context()
        mask: Triton mask for valid elements
    
    Note: Only supports 8-byte (uint64/int64) atomic operations.
    The result_ptr will contain the original value at the remote location.
    If result == compare_value, the swap succeeded.
    """
    # Extract context fields
    my_rank = tl.load(device_ctx + 0)
    queue_ptr = tl.load(device_ctx + 2)
    heap_bases = device_ctx + 3
    
    # Translate dst_ptr from current rank's address space to remote rank's
    translated_dst_ptr = _translate(dst_ptr, my_rank, dst_rank, heap_bases)
    
    # Enqueue ATOMIC_CAS operation (op_code=6)
    queue_pos = _enqueue_atomic_op(result_ptr, translated_dst_ptr, dst_rank, 6,
                                     swap_value, compare_value, queue_ptr, mask)
    
    # Wait for CPU to complete the atomic operation
    _wait_for_completion(queue_ptr, queue_pos)
    
    # Result is now ready at result_ptr (original value from remote)


@triton.jit
def _enqueue_atomic_op(result_ptr, dst_ptr, to_rank: tl.constexpr, op_code: tl.constexpr,
                        operand, compare, queue_ptr, mask):
    """
    Internal: Enqueue an atomic RDMA operation to the queue.
    
    Args:
        result_ptr: Local pointer for result
        dst_ptr: Destination pointer on remote rank (already translated)
        to_rank: Target rank ID
        op_code: Operation type (4=ATOMIC_ADD, 6=ATOMIC_CAS)
        operand: Operand value (add_value or swap_value)
        compare: Compare value (0 for ADD, compare_value for CAS)
        queue_ptr: Queue pointer from device context
        mask: Triton mask for valid elements
    """
    state_ptr = queue_ptr.to(tl.pointer_type(tl.uint64))
    
    # Load QueueState fields
    items_ptr = tl.load(state_ptr + 0)
    head_ptr = tl.load(state_ptr + 1)
    tail_ptr = tl.load(state_ptr + 2)
    
    # Load size (at offset 32 bytes = 4 * uint64)
    size_ptr = queue_ptr.to(tl.pointer_type(tl.int32))
    size = tl.load(size_ptr + 8)
    
    # Atomic increment head to reserve slot
    head_ptr_typed = head_ptr.to(tl.pointer_type(tl.uint64))
    prev_head = tl.atomic_add(head_ptr_typed, 1, sem='relaxed', scope='sys')
    
    # Wait for slot to be free
    size_u64 = size.to(tl.uint64)
    tail_ptr_typed = tail_ptr.to(tl.pointer_type(tl.uint64))
    current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    while prev_head >= size_u64 + current_tail:
        current_tail = tl.atomic_add(tail_ptr_typed, 0, sem='acquire', scope='sys')
    
    # Calculate slot position
    slot_idx = prev_head % size_u64
    
    # WorkItem structure (48 bytes total):
    # Header (32 bytes due to alignas(16)):
    # offset 0:  uint64_t dst_ptr
    # offset 8:  uint64_t src_ptr (result_ptr for atomics)
    # offset 16: uint32_t size_bytes (WRITE LAST as ready flag)
    # offset 20: uint16_t rank
    # offset 22: uint8_t op_type
    # offset 23: uint8_t reserved
    # offset 24-31: padding (alignas(16) pads header to 32 bytes)
    # Atomic fields (16 bytes):
    # offset 32: uint64_t atomic_operand
    # offset 40: uint64_t atomic_compare
    WORK_ITEM_SIZE_BYTES = 48  # Header (32 with padding) + atomic fields (16)
    
    slot_offset_bytes = slot_idx * WORK_ITEM_SIZE_BYTES
    
    # Get pointer to this work item
    items_ptr_u64 = items_ptr.to(tl.pointer_type(tl.uint64))
    slot_ptr_u64 = items_ptr_u64 + (slot_offset_bytes // 8).to(tl.int32)
    
    # Cast pointers to uint64
    dst_ptr_val = tl.cast(dst_ptr, tl.uint64)
    result_ptr_val = tl.cast(result_ptr, tl.uint64)
    operand_u64 = tl.cast(operand, tl.uint64)
    compare_u64 = tl.cast(compare, tl.uint64)
    
    # Write WorkItem fields (except size which is written last as ready flag)
    # Offset 0: dst_ptr (remote target)
    tl.store(slot_ptr_u64 + 0, dst_ptr_val)
    
    # Offset 8: src_ptr (result buffer)
    tl.store(slot_ptr_u64 + 1, result_ptr_val)
    
    # Offset 32: atomic_operand (offset 32 bytes = 4 * 8 bytes)
    tl.store(slot_ptr_u64 + 4, operand_u64)
    
    # Offset 40: atomic_compare (offset 40 bytes = 5 * 8 bytes)
    tl.store(slot_ptr_u64 + 5, compare_u64)
    
    # Offset 20 (bytes): Pack rank (16 bits) + op_type (8 bits) into 32 bits
    # Same as regular RDMA operations
    slot_ptr_u32 = slot_ptr_u64.to(tl.pointer_type(tl.uint32))
    metadata = (to_rank & 0xFFFF) | ((op_code & 0xFF) << 16)
    tl.store(slot_ptr_u32 + 5, metadata)  # offset 20 bytes = 5 * 4 bytes
    
    # Offset 16 (bytes) / 4 (uint32): size_bytes - WRITE LAST as ready flag
    # For atomics, size is always 8 bytes
    tl.store(slot_ptr_u32 + 4, tl.cast(8, tl.uint32))  # offset 16 bytes = 4 * 4 bytes
    
    return prev_head


__all__ = [
    "IrisRDMA",
    "iris",
    "put",
    "get",
    "atomic_add",
    "atomic_cas",
]

