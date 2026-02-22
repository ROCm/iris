# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris Gluon: Gluon-based Multi-GPU Communication Framework

This module provides a Gluon-based implementation of Iris that uses the
`@aggregate` decorator with Gluon's `@gluon.jit` to encapsulate the Iris backend
struct, eliminating the need to pass `heap_bases` around manually.

Key Features:
- Uses Gluon's `@gluon.jit` decorator for device-side methods
- Encapsulates `heap_bases` and rank info in `IrisDeviceCtx` aggregate
- Provides same functionality as original Iris with improved ergonomics

Example:
    >>> import iris.iris_gluon as iris_gl
    >>> ctx = iris_gl.iris(heap_size=2**30)  # 1GB heap
    >>> context_tensor = ctx.get_device_context()  # Get context tensor
    >>>
    >>> @gluon.jit
    >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
    >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
    >>>     data = ctx.load(buffer, 1)
"""

from triton.language.core import _aggregate as aggregate

# Import Gluon - if this fails, you need to update Triton to a version with Gluon support
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError as e:
    raise ImportError(
        "Gluon is not available in your Triton installation. "
        "Please update Triton to a version with Gluon support to use iris.experimental.iris_gluon. "
        "You can install the latest Triton with: pip install --upgrade triton"
    ) from e

import triton.language as tl

from iris._distributed_helpers import (
    init_distributed,
    distributed_barrier,
    distributed_broadcast_scalar,
    distributed_broadcast_tensor,
)
from iris.hip import (
    set_device,
    get_cu_count,
    count_devices,
)
from iris.symmetric_heap import SymmetricHeap
import numpy as np
import torch
import logging

# Import logging functionality from the separate logging module
from ..logging import logger

# Import shared tensor-creation helpers
from .. import tensor_creation


@aggregate
class IrisDeviceCtx:
    """
    Gluon device-side context that decodes the tensor from Iris.get_device_context().

    This aggregate encapsulates the `heap_bases` pointer and provides
    device-side methods for memory operations and atomics using Gluon.

    Attributes:
        cur_rank: Current rank ID
        num_ranks: Total number of ranks
        heap_bases: Pointer to array of heap base addresses for all ranks
    """

    cur_rank: gl.tensor
    num_ranks: gl.tensor
    heap_bases: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cur_rank, num_ranks, heap_bases):
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        self.heap_bases = heap_bases

    @staticmethod
    @gluon.jit
    def initialize(context_tensor):
        """
        Initialize `IrisDeviceCtx` from the encoded tensor.

        The context tensor has the format: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

        Args:
            context_tensor: Pointer to encoded context data

        Returns:
            `IrisDeviceCtx`: Initialized device context
        """
        # Decode the tensor: [cur_rank, num_ranks, heap_base_0, heap_base_1, ...]
        cur_rank = gl.load(context_tensor + 0)
        num_ranks = gl.load(context_tensor + 1)

        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2  # Offset pointer to start at heap bases

        return IrisDeviceCtx(cur_rank, num_ranks, heap_bases)

    @gluon.jit
    def _translate(self, ptr, from_rank, to_rank):
        """
        Internal function to translate a pointer from one rank's address space to another.

        Args:
            ptr: Pointer in the `from_rank`'s address space
            from_rank: Source rank ID
            to_rank: Target rank ID

        Returns:
            Translated pointer in the `to_rank`'s address space
        """
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)
        # convert to int to compute difference
        ptr_int = tl.cast(ptr, gl.uint64)
        # Find the offset from from_rank heap
        offset = ptr_int - from_base
        # Byte cast for byte offset addition
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))
        # Find the offset into the to_rank heap
        translated_ptr_byte = to_base_byte + offset
        # Cast to_base back to pointer type
        translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)

        # Optimization to vectorize the load/store - similar to iris.py
        # This enables the compiler to generate dwordx4 or wider loads
        # Note: Gluon uses scalar multiples, not 2D tuples like Triton
        # ptr = gl.max_contiguous(gl.multiple_of(ptr, 64), 64)
        # translated_ptr = gl.max_contiguous(gl.multiple_of(translated_ptr, 64), 64)

        return translated_ptr

    @gluon.jit
    def load(self, pointer, from_rank, mask=None):
        """
        Loads a value from the specified rank's memory location to the current rank.

        Args:
            pointer: Pointer in the `from_rank`'s address space
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional loading

        Returns:
            The loaded value from the target memory location

        Example:
            >>> # Load from rank 1 to current rank
            >>> data = ctx.load(buffer + offsets, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, from_rank)
        result = gl.load(translated_ptr, mask=mask)
        return result

    @gluon.jit
    def store(self, pointer, value, to_rank, mask=None):
        """
        Writes data from the current rank to the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            value: The value to store
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional storing

        Example:
            >>> # Store from current rank to rank 1
            >>> ctx.store(buffer + offsets, values, 1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        gl.store(translated_ptr, value, mask=mask)

    @gluon.jit
    def get(self, from_ptr, to_ptr, from_rank, mask=None):
        """
        Copies data from the specified rank's memory to the current rank's local memory.

        Args:
            from_ptr: Pointer to remote memory in `from_rank`'s address space
            to_ptr: Pointer to local memory in current rank
            from_rank: The rank ID from which to read the data
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from rank 1 to current rank's local memory
            >>> ctx.get(remote_ptr + offsets, local_ptr + offsets, 1, mask=mask)
        """
        translated_from_ptr = self._translate(from_ptr, self.cur_rank, from_rank)
        data = gl.load(translated_from_ptr, mask=mask)
        gl.store(to_ptr, data, mask=mask)

    @gluon.jit
    def put(self, from_ptr, to_ptr, to_rank, mask=None):
        """
        Copies data from the current rank's local memory to the specified rank's memory.

        Args:
            from_ptr: Pointer to local memory in current rank
            to_ptr: Pointer to remote memory in `to_rank`'s address space
            to_rank: The rank ID to which the data will be written
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from current rank's local memory to rank 1
            >>> ctx.put(local_ptr + offsets, remote_ptr + offsets, 1, mask=mask)
        """
        translated_to_ptr = self._translate(to_ptr, self.cur_rank, to_rank)
        data = gl.load(from_ptr, mask=mask)
        gl.store(translated_to_ptr, data, mask=mask)

    @gluon.jit
    def copy(self, src_ptr, dst_ptr, from_rank, to_rank, mask=None):
        """
        Copies data from the specified rank's memory into the destination rank's memory.

        This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
        space to the `to_rank`'s address space, performing a masked load from the translated
        source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
        If `from_rank` and `to_rank` are the same, this function performs a local copy operation.
        It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

        Args:
            src_ptr: Pointer in the `from_rank`'s local memory from which to read data
            dst_ptr: Pointer in the `to_rank`'s local memory where the data will be written
            from_rank: The rank ID that owns `src_ptr` (source rank)
            to_rank: The rank ID that will receive the data (destination rank)
            mask: Optional mask for conditional operations

        Example:
            >>> # Copy from rank 1 to rank 0 (current rank must be either 1 or 0)
            >>> ctx.copy(remote_ptr + offsets, local_ptr + offsets, 1, 0, mask=mask)
        """
        cur_base = gl.load(self.heap_bases + self.cur_rank)
        from_base = gl.load(self.heap_bases + from_rank)
        to_base = gl.load(self.heap_bases + to_rank)

        src_ptr_int = tl.cast(src_ptr, gl.uint64)
        src_offset = src_ptr_int - cur_base

        dst_ptr_int = tl.cast(dst_ptr, gl.uint64)
        dst_offset = dst_ptr_int - cur_base

        from_base_byte = tl.cast(from_base, gl.pointer_type(gl.int8))
        to_base_byte = tl.cast(to_base, gl.pointer_type(gl.int8))

        translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
        translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

        data = gl.load(translated_src, mask=mask)
        gl.store(translated_dst, data, mask=mask)

    @gluon.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic add at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to add
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically add to rank 1's memory
            >>> old_val = ctx.atomic_add(buffer, 5, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Atomically subtracts data from the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            val: The value to subtract
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically subtract from rank 1's memory
            >>> old_val = ctx.atomic_sub(buffer, 3, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None):
        """
        Atomically compares and exchanges the specified rank's memory location.

        Args:
            pointer: Pointer in the current rank's address space
            cmp: The expected value to compare
            val: The new value to write if comparison succeeds
            to_rank: The rank ID to which the atomic operation will be performed
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Compare-and-swap on rank 1's memory
            >>> old_val = ctx.atomic_cas(flag + pid, 0, 1, 1, sem="release", scope="sys")
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic exchange at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to exchange
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Exchange value with rank 1's memory
            >>> old_val = ctx.atomic_xchg(buffer, 99, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic xor at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to xor
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically XOR with rank 1's memory
            >>> old_val = ctx.atomic_xor(buffer, 0xFF, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic and at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to and
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically AND with rank 1's memory
            >>> old_val = ctx.atomic_and(buffer, 0x0F, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic or at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to or
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically OR with rank 1's memory
            >>> old_val = ctx.atomic_or(buffer, 0xF0, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic min at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute minimum with rank 1's memory
            >>> old_val = ctx.atomic_min(buffer, 10, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @gluon.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None):
        """
        Performs an atomic max at the specified rank's memory location.

        Args:
            pointer: The memory location in the current rank's address space
            val: The value to compare and potentially store
            to_rank: The rank ID to which the atomic operation will be performed
            mask: Optional mask for conditional operations
            sem: Memory semantics (acquire, release, acq_rel, relaxed)
            scope: Scope of synchronization (gpu, cta, sys)

        Returns:
            The value at the memory location before the atomic operation

        Example:
            >>> # Atomically compute maximum with rank 1's memory
            >>> old_val = ctx.atomic_max(buffer, 100, 1)
        """
        translated_ptr = self._translate(pointer, self.cur_rank, to_rank)
        return gl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


class IrisGluon:
    """
    Gluon-based Iris class for multi-GPU communication and memory management.

    This class provides the same functionality as the original Iris class but
    uses Gluon's `@aggregate` decorator to encapsulate the backend state.

    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)

    Example:
        >>> ctx = iris_gluon.iris(heap_size=2**31)  # 2GB heap
        >>> backend = ctx.get_backend()  # Get Gluon aggregate
        >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)
    """

    def __init__(self, heap_size=1 << 30):
        # Initialize distributed environment
        comm, cur_rank, num_ranks = init_distributed()
        num_gpus = count_devices()

        gpu_id = cur_rank % num_gpus
        set_device(gpu_id)

        self.comm = comm
        self.num_ranks = num_ranks
        self.cur_rank = cur_rank
        self.gpu_id = gpu_id
        self.heap_size = heap_size
        self.device = f"cuda:{gpu_id}"

        # Initialize symmetric heap
        self.heap = SymmetricHeap(heap_size, gpu_id, cur_rank, num_ranks)
        self.heap_bases = self.heap.get_heap_bases()

        for i in range(num_ranks):
            self.debug(f"GPU {i}: Heap base {hex(int(self.heap_bases[i].item()))}")

        distributed_barrier()

        # Initialize CCL interface
        self.ccl = self.CCL(self)

    class CCL:
        """
        Collective Communication Library (CCL) interface for IrisGluon.

        Provides collective operations that can be called as methods on the IrisGluon instance.
        Example usage:
            >>> shmem = iris_gluon.iris()
            >>> shmem.ccl.all_to_all(output_tensor, input_tensor)
        """

        def __init__(self, iris_instance):
            """
            Initialize CCL with a reference to the parent IrisGluon instance.

            Args:
                iris_instance: The parent IrisGluon instance
            """
            self._iris = iris_instance

        def all_to_all(self, output_tensor, input_tensor, group=None, async_op=False, config=None):
            """
            All-to-all collective operation.

            Each rank sends a tensor chunk to each other rank and receives
            a tensor chunk from each other rank. Input/output tensors should have
            shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

            Args:
                output_tensor: Output tensor of shape (M, N * world_size)
                input_tensor: Input tensor of shape (M, N * world_size)
                group: ProcessGroup or None. If None, uses all ranks in shmem context.
                       Default: None.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Set config.use_gluon=True to use Gluon implementation with traffic shaping.

            Example:
                >>> shmem = iris_gluon.iris()
                >>> shmem.ccl.all_to_all(output_tensor, input_tensor)

                >>> # Custom configuration with Gluon traffic shaping
                >>> from iris.ccl import Config
                >>> config = Config(use_gluon=True, block_size_m=128, block_size_n=32)
                >>> shmem.ccl.all_to_all(output_tensor, input_tensor, config=config)
            """
            from iris.ccl.all_to_all import all_to_all as _all_to_all

            _all_to_all(output_tensor, input_tensor, self._iris, group=group, async_op=async_op, config=config)

        def all_gather(self, output_tensor, input_tensor, group=None, async_op=False, config=None):
            """
            All-gather collective operation.

            Each rank sends its input tensor to all ranks, and all ranks receive
            and concatenate all input tensors along dimension 0 (rows), matching
            torch.distributed.all_gather_into_tensor behavior.

            Args:
                output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
                input_tensor: Input tensor of shape (M, N) - local rank's data to send
                group: ProcessGroup or None. If None, uses all ranks in shmem context.
                       Default: None.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.

            Example:
                >>> shmem = iris_gluon.iris()
                >>> # Input: (M, N), Output: (world_size * M, N)
                >>> shmem.ccl.all_gather(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(block_size_m=128, block_size_n=32)
                >>> shmem.ccl.all_gather(output_tensor, input_tensor, config=config)
            """
            from iris.ccl.all_gather import all_gather as _all_gather

            _all_gather(output_tensor, input_tensor, self._iris, group=group, async_op=async_op, config=config)

        def reduce_scatter(self, output_tensor, input_tensor, op=None, group=None, async_op=False, config=None):
            """
            Reduce-scatter collective operation.

            Each rank reduces its assigned tiles from all ranks' inputs and stores
            the result only to its own output tensor. This is similar to all-reduce
            but without broadcasting the result to all ranks.

            Args:
                output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
                input_tensor: Input tensor of shape (M, N) - local rank's partial data
                op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
                    Default: ReduceOp.SUM.
                group: ProcessGroup or None. If None, uses all ranks in shmem context.
                       Default: None.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Only supports reduce_scatter_variant="two_shot".

            Example:
                >>> shmem = iris_gluon.iris()
                >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
                >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
            """
            from iris.ccl.reduce_scatter import reduce_scatter as _reduce_scatter
            from iris.ccl import ReduceOp

            # Default to SUM if not specified
            if op is None:
                op = ReduceOp.SUM

            _reduce_scatter(
                output_tensor, input_tensor, self._iris, op=op, group=group, async_op=async_op, config=config
            )

    def _log_with_rank(self, level, message):
        """Helper method to log with rank information injected into the record."""
        extra = {"iris_rank": self.cur_rank, "iris_num_ranks": self.num_ranks}
        logger.log(level, message, extra=extra)

    def debug(self, message):
        """Log a debug message with rank information."""
        self._log_with_rank(logging.DEBUG, message)

    def info(self, message):
        """Log an info message with rank information."""
        self._log_with_rank(logging.INFO, message)

    def warning(self, message):
        """Log a warning message with rank information."""
        self._log_with_rank(logging.WARNING, message)

    def error(self, message):
        """Log an error message with rank information."""
        self._log_with_rank(logging.ERROR, message)

    def get_device_context(self):
        """
        Get the device context tensor for Gluon kernels.

        Returns a tensor encoding: `[cur_rank, num_ranks, heap_base_0, heap_base_1, ...]`

        Returns:
            torch.Tensor: Encoded context data as int64 tensor on device

        Example:
            >>> ctx = iris_gluon.iris()
            >>> context_tensor = ctx.get_device_context()
            >>>
            >>> @gluon.jit
            >>> def kernel(IrisDeviceCtx: gl.constexpr, context_tensor):
            >>>     ctx = IrisDeviceCtx.initialize(context_tensor)
            >>>     data = ctx.load(buffer, 1)
        """
        # Convert heap_bases to a list for concatenation
        heap_bases_list = self.heap_bases.tolist()

        # Create context tensor: [cur_rank, num_ranks, heap_base_0, heap_base_1, ...]
        context_data = [self.cur_rank, self.num_ranks] + heap_bases_list
        context_tensor = torch.tensor(context_data, dtype=torch.int64, device=self.device)

        return context_tensor

    def get_backend(self):
        """
        Legacy method for backward compatibility.
        Use get_device_context() for Gluon kernels.

        Returns:
            torch.Tensor: Device context tensor
        """
        return self.get_device_context()

    def get_heap_bases(self):
        """
        Return the tensor of symmetric heap base addresses for all ranks.

        Returns:
            torch.Tensor: A 1D tensor of uint64 heap base addresses
        """
        return self.heap_bases

    def barrier(self, group=None):
        """
        Synchronize ranks within the specified group using a distributed barrier.

        Args:
            group (ProcessGroup, optional): The process group to synchronize.
                If None, uses the default process group (all ranks).
        """
        distributed_barrier(group=group)

    def get_device(self):
        """
        Get the underlying device where the Iris symmetric heap resides.

        Returns:
            torch.device: The CUDA device of Iris-managed memory
        """
        return self.heap.get_device()

    def get_cu_count(self):
        """
        Get the number of compute units (CUs) for the current GPU.

        Returns:
            int: Number of compute units on this rank's GPU
        """
        return get_cu_count(self.gpu_id)

    def get_rank(self):
        """
        Get the current rank ID.

        Returns:
            int: The current rank ID
        """
        return self.cur_rank

    def get_num_ranks(self):
        """
        Get the total number of ranks.

        Returns:
            int: The total number of ranks in the distributed system
        """
        return self.num_ranks

    def broadcast(self, data, src_rank=0):
        """
        Broadcast data from source rank to all ranks.

        Args:
            data: Data to broadcast (scalar or tensor)
            src_rank: Source rank for broadcast (default: 0)

        Returns:
            The broadcasted data
        """
        # Check if the value on src_rank is a tensor or array-like
        if self.cur_rank == src_rank and data is not None:
            # Explicitly exclude strings and non-numeric types
            if isinstance(data, (str, dict, bool)):
                is_tensor = False
            elif isinstance(data, torch.Tensor):
                is_tensor = True
            elif isinstance(data, np.ndarray):
                is_tensor = True
            elif isinstance(data, (list, tuple)):
                # Try to convert list/tuple to tensor to check if it's numeric
                try:
                    torch.as_tensor(data)
                    is_tensor = True
                except (TypeError, ValueError):
                    is_tensor = False
            else:
                # For other types, try to convert and check
                try:
                    test_array = np.asarray(data)
                    # Check if it's a numeric dtype that torch can handle
                    if np.issubdtype(test_array.dtype, np.number):
                        torch.as_tensor(test_array)
                        is_tensor = True
                    else:
                        is_tensor = False
                except (TypeError, ValueError):
                    is_tensor = False
        else:
            is_tensor = False

        # Broadcast the type decision to all ranks
        is_tensor = distributed_broadcast_scalar(is_tensor, src_rank)

        if is_tensor:
            return distributed_broadcast_tensor(data, root=src_rank)
        else:
            return distributed_broadcast_scalar(data, src_rank)

    def zeros(
        self,
        *size,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    ):
        """
        Create a tensor filled with zeros on the symmetric heap.

        Args:
            size: Shape of the tensor
            dtype: Data type (default: torch.float32)
            device: Device (must match Iris device)
            layout: Layout (default: torch.strided)
            requires_grad: Whether to track gradients

        Returns:
            torch.Tensor: Zero-initialized tensor on the symmetric heap
        """
        return tensor_creation.zeros(
            self.heap,
            self.get_device(),
            size,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def ones(
        self,
        *size,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    ):
        """
        Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.

        Example:
            >>> ctx = iris_gluon.iris(1 << 20)
            >>> tensor = ctx.ones(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([1., 1., 1.], device='cuda:0')
        """
        return tensor_creation.ones(
            self.heap,
            self.get_device(),
            size,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def full(
        self,
        size,
        fill_value,
        *,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    ):
        """
        Creates a tensor of size size filled with fill_value. The tensor's dtype is inferred from fill_value.
        The tensor is allocated on the Iris symmetric heap.

        Args:
            size (int...): a list, tuple, or torch.Size of integers defining the shape of the output tensor.
            fill_value (Scalar): the value to fill the output tensor with.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.

        Example:
            >>> ctx = iris_gluon.iris(1 << 20)
            >>> tensor = ctx.full((2, 3), 3.14)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([3.1400, 3.1400, 3.1400], device='cuda:0')
        """
        return tensor_creation.full(
            self.heap,
            self.get_device(),
            size,
            fill_value,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def zeros_like(
        self,
        input,
        *,
        dtype=None,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=torch.preserve_format,
    ):
        """
        Returns a tensor filled with the scalar value 0, with the same size as input, allocated on the Iris symmetric heap.

        Args:
            input (Tensor): the size of input will determine size of the output tensor.

        Keyword Arguments:
            dtype (torch.dtype, optional): the desired data type of returned Tensor.
                Default: if None, defaults to the dtype of input.
            layout (torch.layout, optional): the desired layout of returned tensor.
                Default: if None, defaults to the layout of input. Note: Iris tensors are always contiguous (strided).
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, defaults to the device of input. Must be compatible with this Iris instance.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                Default: torch.preserve_format.

        Example:
            >>> ctx = iris_gluon.iris(1 << 20)
            >>> input_tensor = ctx.ones(2, 3)
            >>> zeros_tensor = ctx.zeros_like(input_tensor)
            >>> print(zeros_tensor.shape)  # torch.Size([2, 3])
        """
        return tensor_creation.zeros_like(
            self.heap,
            self.get_device(),
            input,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    def is_symmetric(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        This method checks whether a tensor resides in the symmetric heap, making it
        accessible for RMA operations across ranks. Use this to validate tensors before
        performing distributed operations.

        Args:
            tensor (torch.Tensor): PyTorch tensor to check

        Returns:
            bool: True if tensor is on the symmetric heap, False otherwise

        Example:
            >>> import iris.experimental.iris_gluon as iris_gl
            >>> ctx = iris_gl.iris(heap_size=2**30)
            >>> # Create a symmetric tensor
            >>> symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
            >>> ctx.is_symmetric(symmetric_tensor)  # True
            >>>
            >>> # Create an external tensor (not on symmetric heap)
            >>> external_tensor = torch.zeros(1000, dtype=torch.float32, device='cuda')
            >>> ctx.is_symmetric(external_tensor)   # False
        """
        return self.heap.is_symmetric(tensor)


def iris(heap_size=1 << 30):
    """
    Create and return a Gluon-based Iris instance with the specified heap size.
    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.
    Returns:
        IrisGluon: An initialized Gluon-based Iris instance
    Example:
        >>> import iris.iris_gluon as iris_gl
        >>> ctx = iris_gl.iris(2**30)  # 1GB heap
        >>> backend = ctx.get_backend()
        >>> tensor = ctx.zeros(1024, 1024)
    """
    return IrisGluon(heap_size)
