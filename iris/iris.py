# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Iris: Multi-GPU Communication and Memory Management Framework

Iris is a high-performance framework that enables seamless multi-GPU programming in Triton,
enabling fine-grained communication and compute overlap natively in Triton
across multiple GPUs with SHMEM-like Remote Memory Access (RMA) capabilities.

Key Features:
- Symmetric heap management across multiple GPUs
- High-performance atomic operations (add, cas, xchg, xor, and, or, min, max)
- Efficient load/store operations with rank-to-rank communication
- Memory allocation and deallocation utilities
- Built-in logging with rank information
- PyTorch distributed integration for distributed computing
- DeviceContext: Object-oriented API for device-side operations (gluon-style)

Example (Traditional Functional API):
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)  # 1GB heap
    >>> tensor = ctx.zeros(1024, 1024, dtype=torch.float32)
    >>>
    >>> @triton.jit
    >>> def kernel(buffer, heap_bases, rank, world_size):
    >>>     data = iris.load(buffer, rank, remote_rank, heap_bases)

Example (Object-Oriented DeviceContext API):
    >>> import iris
    >>> from iris import DeviceContext
    >>> ctx = iris.iris(heap_size=2**30)
    >>> context_tensor = ctx.get_device_context()
    >>>
    >>> @triton.jit
    >>> def kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr):
    >>>     device_ctx = DeviceContext.initialize(context_tensor, rank, world_size)
    >>>     data = device_ctx.load(buffer, from_rank=remote_rank)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

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

import anvil
from iris.symmetric_heap import SymmetricHeap
import numpy as np
import torch
import logging

# Import logging functionality from the separate logging module
from .logging import logger

# Import tracing functionality
from .tracing import Tracing, TraceEvent, DeviceTracing  # noqa: F401  re-export for iris.TraceEvent

# Import shared tensor-creation helpers
from . import tensor_creation
from .util import is_simulation_env


class Iris:
    """
    Main Iris class for multi-GPU communication and memory management.

    This class provides a unified interface for distributed GPU operations including
    memory allocation, atomic operations, and inter-rank communication.

    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)
        allocator_type (str): Type of allocator to use. Options: "torch" (default), "vmem"

    Example:
        >>> ctx = iris.iris(heap_size=2**31)  # 2GB heap with torch allocator
        >>> print(f"Rank {ctx.cur_rank} of {ctx.num_ranks}") # Rank 0 of 1
        >>> tensor = ctx.zeros(1000, 1000, dtype=torch.float32)

        >>> # Use VMem allocator for memory oversubscription
        >>> ctx = iris.iris(heap_size=2**31, allocator_type="vmem")
    """

    def __init__(self, heap_size=1 << 30, allocator_type="torch"):
        if is_simulation_env():
            allocator_type = "torch"

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

        # Initialize symmetric heap with specified allocator
        self.heap = SymmetricHeap(heap_size, gpu_id, cur_rank, num_ranks, allocator_type)
        self.device = f"cuda:{gpu_id}"
        self.heap_bases = self.heap.get_heap_bases()

        # Pre-fetch heap_bases to CPU for host-side address translation
        # This avoids needing to copy from GPU during SDMA operations
        self.heap_bases_cpu = self.heap_bases.cpu().numpy()

        if is_simulation_env():
            import json

            heap_bases_list = [int(self.heap_bases[r].item()) for r in range(self.num_ranks)]
            out_path = f"iris_rank_{self.cur_rank}_heap_bases.json"
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "rank": self.cur_rank,
                        "num_ranks": self.num_ranks,
                        "heap_bases": [hex(b) for b in heap_bases_list],
                    },
                    f,
                    indent=2,
                )

        distributed_barrier()

        # initialize copy engines
        self.copy_engines = anvil.AnvilLib.get_instance()
        self.copy_engines.init()

        # connect to all peers (including local)
        # TODO only connect local ranks
        # TODO get size
        context_size = 6
        self.copy_engines_device_ctx = torch.zeros((num_ranks, context_size), dtype=torch.uint64, device=self.device)

        for rank in range(num_ranks):
            # Device-initiated queues
            self.copy_engines.connect(cur_rank, rank, 1, allocate_on_host=False)
            # Host-initiated queues
            self.copy_engines.connect(cur_rank, rank, 1, allocate_on_host=True)

            queue = self.copy_engines.get_sdma_queue(cur_rank, rank, 0)
            handle = queue.device_ctx()
            self.info(f"---- Queue {rank} ------------")
            self.info(f"queue_buf {handle.queue_buf:#x} at {id(handle.queue_buf):#x}")
            self.info(f"rptr {handle.rptr:#x} at {id(handle.rptr):#x}")
            self.info(f"wptr {handle.wptr:#x} at {id(handle.wptr):#x}")
            self.info(f"doorbell {handle.doorbell:#x} at {id(handle.doorbell):#x}")
            self.info(f"cached_write_ptr {handle.cached_wptr:#x} at {id(handle.cached_wptr):#x}")
            self.info(f"committed_write_ptr {handle.committed_wptr:#x} at {id(handle.committed_wptr):#x}")

            self.copy_engines_device_ctx[rank][0] = handle.queue_buf
            self.copy_engines_device_ctx[rank][1] = handle.rptr
            self.copy_engines_device_ctx[rank][2] = handle.wptr
            self.copy_engines_device_ctx[rank][3] = handle.doorbell
            self.copy_engines_device_ctx[rank][4] = handle.cached_wptr
            self.copy_engines_device_ctx[rank][5] = handle.committed_wptr
        # Initialize CCL interface
        self.ccl = self.CCL(self)

        # Lazy initialization for ops interface
        self._ops = None

        # Initialize tracing
        self.tracing = Tracing(self)

    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, "heap") and hasattr(self.heap, "allocator"):
                if hasattr(self.heap.allocator, "close"):
                    self.heap.allocator.close()
        except Exception:
            pass  # Best effort cleanup in destructor (GC context)

    def _log_with_rank(self, level, message):
        """Helper method to log with rank information injected into the record."""
        if logger.isEnabledFor(level):
            record = logging.LogRecord(
                name=logger.name, level=level, pathname="", lineno=0, msg=message, args=(), exc_info=None
            )
            # Inject rank information into the record
            record.iris_rank = self.cur_rank
            record.iris_num_ranks = self.num_ranks
            logger.handle(record)

    def debug(self, message):
        """
        Log a debug message with rank information.

        Args:
            message (str): Human-readable message to log at debug level.

        Notes:
            The log record is enriched with ``iris_rank`` and ``iris_num_ranks`` so
            formatters can display the originating rank and world size.

        Example:
            >>> ctx = iris.iris()
            >>> iris.set_logger_level(iris.DEBUG)
            >>> ctx.debug("Allocating buffers")  # [Iris] [0/1] Allocating buffers
        """
        self._log_with_rank(logging.DEBUG, message)

    def info(self, message):
        """
        Log an info message with rank information.

        Args:
            message (str): Human-readable message to log at info level.

        Example:
            >>> ctx = iris.iris()
            >>> ctx.info("Starting iteration 0")  # [Iris] [0/1] Starting iteration 0
        """
        self._log_with_rank(logging.INFO, message)

    def warning(self, message):
        """
        Log a warning message with rank information.

        Args:
            message (str): Human-readable message to log at warning level.

        Example:
            >>> ctx = iris.iris()
            >>> ctx.warning("Memory usage is high")  # [Iris] [0/1] Memory usage is high
        """
        self._log_with_rank(logging.WARNING, message)

    def error(self, message):
        """
        Log an error message with rank information.

        Args:
            message (str): Human-readable message to log at error level.

        Example:
            >>> ctx = iris.iris()
            >>> ctx.error("Failed to allocate memory")  # [Iris] [0/1] Failed to allocate memory
        """
        self._log_with_rank(logging.ERROR, message)

    @property
    def ops(self):
        """
        Access fused GEMM+CCL operations.

        This property provides a namespace for high-level fused operations that combine
        matrix multiplication with collective communication. Operations automatically infer
        dimensions, strides, and hardware parameters from input tensors.

        Available operations:
            - matmul_all_reduce: GEMM + All-Reduce
            - all_gather_matmul: All-Gather + GEMM
            - matmul_all_gather: GEMM + All-Gather
            - matmul_reduce_scatter: GEMM + Reduce-Scatter

        Returns:
            OpsNamespace: Namespace with fused operation methods

        Raises:
            ImportError: If tritonBLAS is not available

        Example:
            >>> ctx = iris.iris()
            >>> A = ctx.randn((1024, 512), dtype=torch.float16)
            >>> B = ctx.randn((512, 2048), dtype=torch.float16)
            >>> output = ctx.zeros((1024, 2048), dtype=torch.float16)
            >>> ctx.ops.matmul_all_reduce(output, A, B, ctx)
        """
        if self._ops is None:
            from iris.ops import OpsNamespace

            self._ops = OpsNamespace(self)
        return self._ops

    def broadcast(self, value, source_rank=0):
        """
        Broadcast a value from one rank to all ranks.

        This method automatically detects the type of value and uses the appropriate
        broadcast mechanism:
        - For tensors and arrays: uses efficient PyTorch distributed tensor collectives
        - For scalars and other objects: uses object broadcast

        Args:
            value (Any): The value to broadcast. Can be a scalar, tensor, numpy array,
                or any picklable object. Only the ``source_rank`` value is used;
                other ranks should pass a placeholder (e.g., ``None``).
            source_rank (int): Rank id that holds the authoritative value.

        Returns:
            Any: The value broadcast to all ranks. Tensors and arrays are returned as
                numpy arrays; scalars and objects are returned in their original type.

        Examples:
            >>> ctx = iris.iris()
            >>> # Broadcasting a scalar
            >>> value = 42 if ctx.cur_rank == 0 else None
            >>> value = ctx.broadcast(value, source_rank=0)  # All ranks get 42
            >>>
            >>> # Broadcasting a tensor
            >>> if ctx.cur_rank == 0:
            >>>     data = torch.randn(10, 10)
            >>> else:
            >>>     data = None
            >>> data = ctx.broadcast(data, source_rank=0)  # All ranks get the same array
        """
        # Check if the value on source_rank is a tensor or array-like
        if self.cur_rank == source_rank and value is not None:
            # Explicitly exclude strings and non-numeric types
            if isinstance(value, (str, dict, bool)):
                is_tensor = False
            elif isinstance(value, torch.Tensor):
                is_tensor = True
            elif isinstance(value, np.ndarray):
                is_tensor = True
            elif isinstance(value, (list, tuple)):
                # Try to convert list/tuple to tensor to check if it's numeric
                try:
                    torch.as_tensor(value)
                    is_tensor = True
                except (TypeError, ValueError):
                    is_tensor = False
            else:
                # For other types, try to convert and check
                try:
                    test_array = np.asarray(value)
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
        is_tensor = distributed_broadcast_scalar(is_tensor, source_rank)

        if is_tensor:
            return distributed_broadcast_tensor(value, root=source_rank)
        else:
            return distributed_broadcast_scalar(value, source_rank)

    def zeros_like(
        self, input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format
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
            >>> ctx = iris.iris(1 << 20)
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

    def arange(
        self, start=0, end=None, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
    ):
        """
        Returns a 1-D tensor of size ⌈(end - start) / step⌉ with values from the interval [start, end)
        taken with common difference step beginning from start. The tensor is allocated on the symmetric heap.

        Note: When using floating-point dtypes (especially reduced precision types like bfloat16),
        the results may be affected by floating-point rounding behavior. Some values in the sequence
        might not be exactly representable in certain floating-point formats, which can lead to
        repeated values or unexpected rounding. For precise sequences, it is recommended to use
        integer dtypes instead of floating-point dtypes.

        Note that non-integer step is subject to floating point rounding errors when comparing
        against end; to avoid inconsistency, we advise subtracting a small epsilon from end in such cases.

        Args:
            start (Number, optional): the starting value for the set of points. Default: 0.
            end (Number): the ending value for the set of points
            step (Number, optional): the gap between each pair of adjacent points. Default: 1.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.get_default_dtype()).
                If dtype is not given, infer the data type from the other input arguments.
                If any of start, end, or step are floating-point, the dtype is inferred
                be the default dtype, see get_default_dtype(). Otherwise, the dtype is inferred
                to be torch.int64.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
                Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
            >>> print(tensor.shape)  # torch.Size([5])
        """
        # Handle the case where only one argument is provided (end)
        if end is None:
            end = start
            start = 0
        return tensor_creation.arange(
            self.heap,
            self.get_device(),
            start,
            end,
            step,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def zeros(self, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
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
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.zeros(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([0., 0., 0.], device='cuda:0')
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

    def randn(
        self,
        *size,
        generator=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        """
        Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
        (also called the standard normal distribution). The tensor is allocated on the Iris symmetric heap.

        .. math::
            \\text{out}_i \\sim \\mathcal{N}(0, 1)

        For complex dtypes, the tensor is i.i.d. sampled from a complex normal distribution with zero mean
        and unit variance as

        .. math::
            \\text{out}_i \\sim \\mathcal{CN}(0, 1)

        This is equivalent to separately sampling the real :math:`(\\text{Re})` and imaginary :math:`(\\text{Im})`
        part of :math:`\\text{out}_i` as

        .. math::
            \\text{Re}(\\text{out}_i) \\sim \\mathcal{N}(0, \\frac{1}{2}), \\quad \\text{Im}(\\text{out}_i) \\sim \\mathcal{N}(0, \\frac{1}{2})

        The shape of the tensor is defined by the variable argument size.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type (see torch.set_default_device()).
                device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory.
                Works only for CPU tensors. Default: False.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.randn(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([ 0.3982, -0.0059, -0.4365], device='cuda:0')
        """
        return tensor_creation.randn(
            self.heap,
            self.get_device(),
            size,
            generator=generator,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def ones(self, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
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
            >>> ctx = iris.iris(1 << 20)
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

    def as_symmetric(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Import an external PyTorch tensor into the symmetric heap.

        This creates a new tensor in the symmetric heap that shares physical memory
        with the external tensor. Any modifications to either tensor will be visible
        in both. This is useful for importing pre-allocated tensors (e.g., model weights)
        into the symmetric heap for RMA operations.

        Note: This feature requires `allocator_type='vmem'`.

        Args:
            external_tensor (torch.Tensor): External PyTorch tensor to import.
                Must be a CUDA tensor.

        Returns:
            torch.Tensor: New tensor in symmetric heap sharing memory with external tensor

        Raises:
            RuntimeError: If allocator doesn't support imports or import fails

        Example:
            >>> ctx = iris.iris(allocator_type='vmem')
            >>> # Create an external tensor
            >>> external = torch.randn(1000, 1000, device='cuda')
            >>> # Import it into symmetric heap
            >>> symmetric = ctx.as_symmetric(external)
            >>> # Verify they share memory
            >>> external[0, 0] = 999.0
            >>> assert symmetric[0, 0].item() == 999.0
            >>> # Now you can use symmetric in RMA operations
            >>> ctx.put(symmetric, peer_rank, remote_buffer)
        """
        return self.heap.as_symmetric(external_tensor)

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
            >>> ctx = iris.iris(heap_size=2**30)
            >>> # Create a symmetric tensor
            >>> symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
            >>> ctx.is_symmetric(symmetric_tensor)  # True
            >>>
            >>> # Create an external tensor (not on symmetric heap)
            >>> external_tensor = torch.zeros(1000, dtype=torch.float32, device='cuda')
            >>> ctx.is_symmetric(external_tensor)   # False
            >>>
            >>> # Import external tensor (only with vmem allocator)
            >>> ctx_vmem = iris.iris(allocator_type='vmem')
            >>> imported = ctx_vmem.as_symmetric(external_tensor)
            >>> ctx_vmem.is_symmetric(imported)      # True
        """
        return self.heap.is_symmetric(tensor)

    def full(self, size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
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
            >>> ctx = iris.iris(1 << 20)
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

    def uniform(self, size, low=0.0, high=1.0, dtype=torch.float):
        """
        Returns a tensor filled with random numbers from a uniform distribution, allocated on the Iris symmetric heap.

        Args:
            size (int or tuple of ints): the size of the output tensor.
            low (float, optional): the lower bound of the uniform distribution. Default: 0.0.
            high (float, optional): the upper bound of the uniform distribution. Default: 1.0.
            dtype (torch.dtype, optional): the desired data type of returned tensor. Default: torch.float.

        Returns:
            Tensor: A tensor filled with random numbers from a uniform distribution.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.uniform((2, 3), low=0.0, high=1.0)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([0.1234, 0.5678, 0.9012], device='cuda:0')
        """
        return tensor_creation.uniform(self.heap, self.get_device(), size, low, high, dtype)

    def empty(
        self,
        *size,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        """
        Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Note:
            If torch.use_deterministic_algorithms() and torch.utils.deterministic.fill_uninitialized_memory are both set to True,
            the output tensor is initialized to prevent any possible nondeterministic behavior from using the data as an input to an operation.
            Floating point and complex tensors are filled with NaN, and integer tensors are filled with the maximum value.

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
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory.
                Works only for CPU tensors. Default: False. Note: Iris tensors are always on GPU.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                Default: torch.contiguous_format.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.empty(2, 3)
            >>> print(tensor.shape)  # torch.Size([2, 3])
        """
        return tensor_creation.empty(
            self.heap,
            self.get_device(),
            size,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    def randint(
        self, *args, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
    ):
        """
        Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        The shape of the tensor is defined by the variable argument size.
        The tensor is allocated on the Iris symmetric heap.

        Note:
            With the global dtype default (torch.float32), this function returns a tensor with dtype torch.int64.

        Args:
            low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
            high (int): One above the highest integer to be drawn from the distribution.
            size (tuple): a tuple defining the shape of the output tensor.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): if None, this function returns a tensor with dtype torch.int64.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor. Default: if None, uses the current device.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.randint(0, 10, (2, 3))  # Random integers [0, 10)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([7, 2, 9], device='cuda:0')
        """
        # Parse arguments to determine low, high, and size
        if len(args) == 2:
            high, size = args
            low = 0
        elif len(args) == 3:
            low, high, size = args
        else:
            raise ValueError(f"randint expects 2 or 3 positional arguments, got {len(args)}")
        return tensor_creation.randint(
            self.heap,
            self.get_device(),
            low,
            high,
            size,
            generator=generator,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def linspace(self, start, end, steps, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        The tensor is allocated on the Iris symmetric heap.

        The values are:
        (start, start + (end-start)/(steps-1), ..., start + (steps-2)*(end-start)/(steps-1), end)

        Args:
            start (float or Tensor): the starting value for the set of points. If Tensor, it must be 0-dimensional.
            end (float or Tensor): the ending value for the set of points. If Tensor, it must be 0-dimensional.
            steps (int): size of the constructed tensor.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the data type to perform the computation in.
                Default: if None, uses the global default dtype when both start and end are real,
                and corresponding complex dtype when either is complex.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor. Default: if None, uses the current device.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
            >>> print(tensor) # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000], device='cuda:0')
        """
        return tensor_creation.linspace(
            self.heap,
            self.get_device(),
            start,
            end,
            steps,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def rand(
        self,
        *size,
        generator=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        """
        Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
        The tensor is allocated on the Iris symmetric heap.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            generator (torch.Generator, optional): a pseudorandom number generator for sampling.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
                Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor.
                Default: torch.strided. Note: Iris tensors always use `torch.strided` regardless of this parameter.
            device (torch.device, optional): the desired device of returned tensor.
                Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor.
                Default: False.
            pin_memory (bool, optional): If set, returned tensor would be allocated in the pinned memory.
                Works only for CPU tensors. Default: False. Note: Iris tensors are always on GPU.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> tensor = ctx.rand(2, 3)  # Random values in [0, 1)
            >>> print(tensor.shape)  # torch.Size([2, 3])
            >>> print(tensor[0])  # tensor([0.1234, 0.5678, 0.9012], device='cuda:0')
        """
        return tensor_creation.rand(
            self.heap,
            self.get_device(),
            size,
            generator=generator,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        )

    def __deallocate(self, pointer):
        pass

    def get_heap_bases(self):
        """
        Return the tensor of symmetric heap base addresses for all ranks.

        Returns:
            torch.Tensor: A 1D tensor of ``uint64`` heap base addresses of size ``num_ranks``
            on the Iris device. Pass this to device-side Triton kernels that require
            heap translation.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> heap_bases = ctx.get_heap_bases()
            >>> print(heap_bases.shape)  # torch.Size([num_ranks])
        """
        return self.heap_bases

    def get_copy_engine_ctx(self):
        return self.copy_engines_device_ctx

    def translate(self, ptr: int, from_rank: int, to_rank: int) -> int:
        """
        Translate a pointer address from one rank's address space to another.

        This is useful for host-side SDMA operations where you need to convert
        peer-mapped addresses to the target GPU's local address space.

        Args:
            ptr (int): The pointer address in from_rank's address space
            from_rank (int): Source rank (address space of ptr)
            to_rank (int): Target rank (desired address space)

        Returns:
            int: Translated pointer address in to_rank's address space

        Example:
            >>> ctx = iris.iris()
            >>> buffer = ctx.zeros(1024, dtype=torch.float32)
            >>> # Translate buffer address from rank 0 to rank 1's address space
            >>> remote_addr = ctx.translate(buffer.data_ptr(), 0, 1)
            >>> ctx.copy_engines.host_put(0, 1, 0, src_ptr, remote_addr, size)
        """
        # Use pre-cached CPU copy to avoid GPU->CPU transfer on every call
        from_base = int(self.heap_bases_cpu[from_rank])
        to_base = int(self.heap_bases_cpu[to_rank])
        offset = ptr - from_base
        return to_base + offset

    def put(
        self,
        src_tensor: torch.Tensor,
        dst_rank: int,
        dst_tensor: torch.Tensor = None,
        wait_flag: torch.Tensor = None,
        wait_value: int = None,
        signal_flag: torch.Tensor = None,
        signal_value: int = 1,
        async_op: bool = False,
        channel: int = 0,
    ):
        """
        One-sided put operation with optional wait (POLL) and signal (ATOMIC).

        Supports:
        - Simple copy: put(src, dst_rank)
        - Copy + signal: put(src, dst_rank, signal_flag=flag)
        - Wait + copy: put(src, dst_rank, wait_flag=flag, wait_value=N)
        - Wait + copy + signal: put(src, dst_rank, wait_flag=..., signal_flag=...)

        Args:
            src_tensor: Source tensor (local, must be symmetric)
            dst_rank: Destination rank
            dst_tensor: Destination tensor (symmetric). If None, uses src_tensor.
            wait_flag: Optional LOCAL flag tensor to poll before transfer (POLL packet)
            wait_value: Expected value for wait_flag
            signal_flag: Optional flag tensor to atomic-add on REMOTE rank after transfer (will be translated)
            signal_value: Value to add to signal_flag (default 1)
            async_op: If True, don't wait for completion
            channel: SDMA channel to use

        Examples:
            >>> # Simple copy
            >>> shmem.put(data, dst_rank=1)

            >>> # Copy with completion signal
            >>> shmem.put(data, dst_rank=1, signal_flag=completion_flag)

            >>> # Wait for ready signal, then copy
            >>> shmem.put(data, dst_rank=1, wait_flag=ready_flag, wait_value=1)

            >>> # Full pipeline: wait, copy, signal
            >>> shmem.put(data, dst_rank=1,
            ...          wait_flag=batch_ready, wait_value=256,
            ...          signal_flag=transfer_done, signal_value=1)
        """
        if dst_tensor is None:
            dst_tensor = src_tensor

        src_rank = self.get_rank()
        src_ptr = src_tensor.data_ptr()
        dst_ptr = self.translate(dst_tensor.data_ptr(), src_rank, dst_rank)
        size = src_tensor.numel() * src_tensor.element_size()

        # Determine which SDMA packet combination to use
        has_wait = wait_flag is not None
        has_signal = signal_flag is not None

        if has_wait and has_signal:
            # POLL + COPY + ATOMIC (two submissions)
            wait_ptr = wait_flag.data_ptr()
            signal_ptr = self.translate(signal_flag.data_ptr(), src_rank, dst_rank)

            # First: POLL + COPY
            self.copy_engines.host_wait_flag_then_put(
                src_rank, dst_rank, channel, wait_ptr, wait_value, src_ptr, dst_ptr, size
            )
            # Then: ATOMIC
            self.copy_engines.host_atomic_add(src_rank, dst_rank, channel, signal_ptr, signal_value)

        elif has_wait:
            # POLL + COPY
            wait_ptr = wait_flag.data_ptr()
            self.copy_engines.host_wait_flag_then_put(
                src_rank, dst_rank, channel, wait_ptr, wait_value, src_ptr, dst_ptr, size
            )

        elif has_signal:
            # COPY + ATOMIC (combined in one submission)
            signal_ptr = self.translate(signal_flag.data_ptr(), src_rank, dst_rank)
            self.copy_engines.host_put_signal(
                src_rank, dst_rank, channel, src_ptr, dst_ptr, size, signal_ptr, signal_value
            )

        else:
            # Simple COPY
            self.copy_engines.host_put(src_rank, dst_rank, channel, src_ptr, dst_ptr, size)

        if not async_op:
            self.copy_engines.host_quiet(src_rank, dst_rank, channel)

    def put_tile(self, tile, dst_rank: int,
                 dst_ptr: int,
                 dst_stride: int,
                 wait_flag: int = None, wait_value: int = None,
                 signal_flag: int = None, signal_value: int = 1,
                 async_op: bool = False, channel: int = 0):
        """
        2D tile transfer with optional wait/signal (sub-window copy).

        Low-level API - caller provides pre-translated pointers for performance.

        Args:
            tile: Pre-configured anvil.Tile object with data pointer and dimensions set
            dst_rank: Destination rank
            dst_ptr: Destination pointer (already translated to remote address space)
            dst_stride: Destination row stride in bytes
            wait_flag: Optional LOCAL flag pointer to poll before transfer
            wait_value: Expected value for wait_flag
            signal_flag: Optional REMOTE flag pointer to atomic-add after transfer (already translated)
            signal_value: Value to add to signal_flag
            async_op: If True, don't wait for completion
            channel: SDMA channel to use

        Examples:
            >>> import anvil
            >>> tile = anvil.Tile()
            >>> tile.pid_m = 0
            >>> tile.pid_n = 0
            >>> tile.block_m = 256
            >>> tile.block_n = 256
            >>> tile.elem_size = A.element_size()
            >>> tile.src_stride = A.stride(0) * tile.elem_size
            >>> tile.data = A.data_ptr()
            >>> dst_ptr = shmem.translate(A.data_ptr(), src_rank, dst_rank)
            >>> dst_stride = A.stride(0) * tile.elem_size
            >>> wait_ptr = flag.data_ptr()
            >>> signal_ptr = shmem.translate(flag.data_ptr(), src_rank, dst_rank)
            >>> shmem.put_tile(tile, dst_rank=1, dst_ptr=dst_ptr, dst_stride=dst_stride,
            ...               wait_flag=wait_ptr, wait_value=256, signal_flag=signal_ptr)
        """
        src_rank = self.get_rank()

        has_wait = wait_flag is not None
        has_signal = signal_flag is not None

        if has_wait and has_signal:
            # POLL + SUB_WINDOW_COPY + ATOMIC (two submissions)
            self.copy_engines.host_wait_flag_then_put_tile(
                src_rank, dst_rank, channel, wait_flag, wait_value, tile, dst_ptr, dst_stride
            )
            self.copy_engines.host_atomic_add_32(src_rank, dst_rank, channel, signal_flag, signal_value)

        elif has_wait:
            # POLL + SUB_WINDOW_COPY
            self.copy_engines.host_wait_flag_then_put_tile(
                src_rank, dst_rank, channel, wait_flag, wait_value, tile, dst_ptr, dst_stride
            )

        elif has_signal:
            # SUB_WINDOW_COPY + ATOMIC
            self.copy_engines.host_put_tile_signal(
                src_rank, dst_rank, channel, tile, dst_ptr, dst_stride, signal_flag, signal_value
            )

        else:
            # Simple SUB_WINDOW_COPY
            self.copy_engines.host_put_tile(src_rank, dst_rank, channel, tile, dst_ptr, dst_stride)

        if not async_op:
            self.copy_engines.host_quiet(src_rank, dst_rank, channel)

    def put_tiles(
        self,
        tiles,
        dst_rank: int,
        dst_ptrs,
        dst_strides,
        wait_flag: int = None,
        wait_value: int = None,
        signal_flag: int = None,
        signal_value: int = 1,
        async_op: bool = False,
        channel: int = 0,
    ):
        """
        Batched 2D tile transfer with optional shared wait/signal.

        Args:
            tiles: Sequence of pre-configured anvil.Tile objects
            dst_rank: Destination rank
            dst_ptrs: Sequence of translated destination pointers
            dst_strides: Sequence of destination row strides in bytes
            wait_flag: Optional LOCAL flag pointer to poll before all transfers
            wait_value: Expected value for wait_flag
            signal_flag: Optional REMOTE flag pointer to atomic-add after all transfers
            signal_value: Value to add to signal_flag
            async_op: If True, don't wait for completion
            channel: SDMA channel to use
        """
        src_rank = self.get_rank()

        if len(tiles) != len(dst_ptrs) or len(tiles) != len(dst_strides):
            raise ValueError("tiles, dst_ptrs, and dst_strides must have the same length")

        has_wait = wait_flag is not None
        has_signal = signal_flag is not None

        if has_wait:
            self.copy_engines.host_wait_flag_then_put_tiles(
                src_rank, dst_rank, channel, wait_flag, wait_value, tiles, dst_ptrs, dst_strides
            )
            if has_signal:
                self.copy_engines.host_atomic_add_32(src_rank, dst_rank, channel, signal_flag, signal_value)
        else:
            for tile, dst_ptr, dst_stride in zip(tiles, dst_ptrs, dst_strides):
                self.put_tile(
                    tile,
                    dst_rank=dst_rank,
                    dst_ptr=dst_ptr,
                    dst_stride=dst_stride,
                    signal_flag=None,
                    async_op=True,
                    channel=channel,
                )
            if has_signal:
                self.copy_engines.host_atomic_add_32(src_rank, dst_rank, channel, signal_flag, signal_value)

        if not async_op:
            self.copy_engines.host_quiet(src_rank, dst_rank, channel)

    def quiet(self, dst_rank: int = None, channel: int = 0):
        """
        Wait for all outstanding SDMA operations to complete.

        Args:
            dst_rank: If specified, wait only for ops to this rank.
                     If None, wait for ops to all ranks.
            channel: SDMA channel

        Example:
            >>> shmem.put(tensor, dst_rank=1, async_op=True)
            >>> shmem.quiet(dst_rank=1)  # Wait for completion
            >>> shmem.quiet()  # Wait for all ranks
        """
        src_rank = self.get_rank()
        if dst_rank is not None:
            self.copy_engines.host_quiet(src_rank, dst_rank, channel)
        else:
            # Quiet to all ranks
            for rank in range(self.get_num_ranks()):
                self.copy_engines.host_quiet(src_rank, rank, channel)

    def get_device_context(self):
        """
        Get the device context tensor for DeviceContext initialization.

        Returns a tensor encoding: [cur_rank, world_size, heap_base_0, heap_base_1, ...]
        If tracing is enabled, also includes: [trace_enabled, max_events, trace_counter_ptr, trace_buffer_ptrs...]

        This opaque format allows future extension without breaking the API.

        Returns:
            torch.Tensor: Encoded context data as int64 tensor on device

        Example:
            >>> import iris
            >>> from iris import DeviceContext
            >>> import triton
            >>> import triton.language as tl
            >>>
            >>> ctx = iris.iris()
            >>> context_tensor = shmem.get_device_context()
            >>>
            >>> @triton.jit
            >>> def my_kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size)
            >>>     data = ctx.load(buffer, from_rank=1)
        """
        # Convert heap_bases to a list for concatenation
        heap_bases_list = self.heap_bases.tolist()

        # Create context tensor: [cur_rank, world_size, heap_base_0, heap_base_1, ...]
        context_data = [self.cur_rank, self.num_ranks] + heap_bases_list

        # Add tracing info if enabled
        if self.tracing.enabled:
            # Explicit buffer ordering (must match DeviceContext.initialize extraction order)
            trace_buffer_ptrs = [
                self.tracing.trace_buffers["event_id"].data_ptr(),
                self.tracing.trace_buffers["pid"].data_ptr(),
                self.tracing.trace_buffers["pid_m"].data_ptr(),
                self.tracing.trace_buffers["pid_n"].data_ptr(),
                self.tracing.trace_buffers["cur_rank"].data_ptr(),
                self.tracing.trace_buffers["target_rank"].data_ptr(),
                self.tracing.trace_buffers["xcc_id"].data_ptr(),
                self.tracing.trace_buffers["cu_id"].data_ptr(),
                self.tracing.trace_buffers["timestamp"].data_ptr(),
                self.tracing.trace_buffers["address"].data_ptr(),
                self.tracing.trace_buffers["duration_cycles"].data_ptr(),
                self.tracing.trace_buffers["op_index"].data_ptr(),
                self.tracing.trace_buffers["payload_size"].data_ptr(),
            ]
            context_data += [
                1,  # trace_enabled = 1 (true)
                self.tracing.max_events,
                self.tracing.trace_counter.data_ptr(),
                self.tracing.op_index_counter.data_ptr(),
            ] + trace_buffer_ptrs
        else:
            context_data += [0]  # trace_enabled = 0 (false)

        context_tensor = torch.tensor(context_data, dtype=torch.int64, device=self.device)

        return context_tensor

    def barrier(self, stream=None, group=None, sync_copy_engine=False):
        """
        Synchronize ranks within the specified group and their CUDA devices.

        This first calls ``torch.cuda.synchronize()`` or ``stream.synchronize()`` to ensure the local GPU has
        finished all queued work, then performs a distributed barrier so that all
        ranks in the group reach the same point before proceeding.

        Args:
            stream: If stream is given: wait only for that stream before barrier. If stream is None: legacy behavior (device-wide sync).
            group (ProcessGroup, optional): The process group to synchronize.
                If None, uses the default process group (all ranks).
            sync_copy_engine (bool, optional): If True, also wait for all outstanding SDMA operations to complete.
                Default is False.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> ctx.barrier()  # Synchronize all ranks
            >>> ctx.barrier(group=my_group)  # Synchronize only ranks in my_group
            >>> ctx.barrier(sync_copy_engine=True)  # Synchronize GPU + SDMA
        """
        # Wait for all GPUs to finish work
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()

        # Wait for SDMA operations if requested
        if sync_copy_engine:
            self.quiet()

        # Distributed barrier
        distributed_barrier(group=group)

    def get_device(self):
        """
        Get the underlying device where the Iris symmetric heap resides.

        Returns:
            torch.device: The CUDA device of Iris-managed memory.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> device = ctx.get_device()
            >>> print(device)  # cuda:0
        """
        return self.heap.get_device()

    def get_cu_count(self):
        """
        Get the number of compute units (CUs) for the current GPU.

        Returns:
            int: Number of compute units on this rank's GPU.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> cu_count = ctx.get_cu_count()
            >>> print(f"GPU has {cu_count} CUs")  # GPU has 304 CUs
        """
        return get_cu_count(self.gpu_id)

    def get_device_id(self):
        """
        Get the device ID used by this Iris instance.

        In simulation mode, this may differ from the local rank if multiple
        ranks share a single GPU. This is the device ID that was set during
        Iris initialization.

        Returns:
            int: The GPU device ID used by this Iris instance.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> device_id = ctx.get_device_id()
            >>> print(f"Using GPU {device_id}")  # Using GPU 0
        """
        return self.gpu_id

    def get_rank(self):
        """
        Get this process's rank id in the distributed communicator.

        Returns:
            int: Zero-based rank id of the current process.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> rank = ctx.get_rank()
            >>> print(f"This is rank {rank}")  # This is rank 0
        """
        return self.cur_rank

    def get_num_ranks(self):
        """
        Get the total number of ranks in the distributed communicator.

        Returns:
            int: World size (number of ranks).

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> num_ranks = ctx.get_num_ranks()
            >>> print(f"Total ranks: {num_ranks}")  # Total ranks: 1
        """
        return self.num_ranks

    class CCL:
        """
        Collective Communication Library (CCL) interface for Iris.

        Provides collective operations that can be called as methods on the Iris instance.
        Example usage:
            >>> ctx = iris.iris()
            >>> ctx.ccl.all_to_all(output_tensor, input_tensor)
        """

        def __init__(self, iris_instance):
            """
            Initialize CCL with a reference to the parent Iris instance.

            Args:
                iris_instance: The parent Iris instance
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

            Example:
                >>> ctx = iris.iris()
                >>> ctx.ccl.all_to_all(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(block_size_m=128, block_size_n=32)
                >>> ctx.ccl.all_to_all(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> ctx.ccl.all_to_all(output_tensor, input_tensor, async_op=True)
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
                >>> ctx = iris.iris()
                >>> # Input: (M, N), Output: (world_size * M, N)
                >>> ctx.ccl.all_gather(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(block_size_m=128, block_size_n=32)
                >>> ctx.ccl.all_gather(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> ctx.ccl.all_gather(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_gather import all_gather as _all_gather

            _all_gather(output_tensor, input_tensor, self._iris, group=group, async_op=async_op, config=config)

        def all_reduce_preamble(self, output_tensor, input_tensor, config=None, workspace=None):
            """
            Prepare reusable workspace for all-reduce.

            Args:
                output_tensor: Output tensor that will receive the reduced data.
                input_tensor: Input tensor providing the local contribution.
                config: Optional Config describing variant parameters.
                workspace: Optional existing workspace to update/reuse.

            Returns:
                Workspace object that can be passed to ``all_reduce``.
            """
            from iris.ccl.all_reduce import all_reduce_preamble as _all_reduce_preamble

            return _all_reduce_preamble(
                output_tensor,
                input_tensor,
                self._iris,
                config=config,
                workspace=workspace,
            )

        def all_reduce(
            self, output_tensor, input_tensor, op=None, group=None, async_op=False, config=None, workspace=None
        ):
            """
            All-reduce collective operation.

            Each rank has a local input tensor, and all ranks compute the sum of all
            input tensors. The result is written to output_tensor on all ranks.

            Args:
                output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
                input_tensor: Input tensor of shape (M, N) - local rank's partial data
                op: Reduction operation to apply. Currently only ReduceOp.SUM is supported.
                    Default: ReduceOp.SUM.
                group: ProcessGroup or None. If None, uses all ranks in shmem context.
                       Default: None.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Set config.all_reduce_variant to choose variant: "atomic", "ring", or "two_shot"
                workspace: Optional workspace prepared by ``all_reduce_preamble`` to
                           reuse internal buffers across invocations.

            Example:
                >>> ctx = iris.iris()
                >>> ctx.ccl.all_reduce(output_tensor, input_tensor)

                >>> # Custom configuration with ring variant
                >>> from iris.ccl import Config
                >>> config = Config(all_reduce_variant="ring")
                >>> ctx.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Two-shot variant with block distribution
                >>> config = Config(all_reduce_variant="two_shot", all_reduce_distribution=1)
                >>> ctx.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> ctx.ccl.all_reduce(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_reduce import all_reduce as _all_reduce
            from iris.ccl import ReduceOp

            # Default to SUM if not specified
            if op is None:
                op = ReduceOp.SUM

            return _all_reduce(
                output_tensor,
                input_tensor,
                self._iris,
                op=op,
                group=group,
                async_op=async_op,
                config=config,
                workspace=workspace,
            )

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
                >>> ctx = iris.iris()
                >>> ctx.ccl.reduce_scatter(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
                >>> ctx.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
            """
            from iris.ccl.reduce_scatter import reduce_scatter as _reduce_scatter
            from iris.ccl import ReduceOp

            # Default to SUM if not specified
            if op is None:
                op = ReduceOp.SUM

            _reduce_scatter(
                output_tensor, input_tensor, self._iris, op=op, group=group, async_op=async_op, config=config
            )


@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases, hint: tl.constexpr = None):
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr_byte = to_base_byte + offset
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)
    if hint is not None:
        translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, hint), hint)
    return translated_ptr


@triton.jit
def translate_ptr(ptr, from_rank, to_rank, heap_bases, hint: tl.constexpr = None):
    """
    Public device-side pointer translation helper.

    This is a thin wrapper around the internal translation routine so Triton
    kernels importing the top-level `iris` package can access address-space
    translation without depending on a private symbol name.
    """
    return __translate(ptr, from_rank, to_rank, heap_bases, hint)


@aggregate
class DeviceContext:
    """
    Device-side context that encapsulates rank and heap_bases for ergonomic Iris operations.

    This aggregate provides an object-oriented interface for Iris device operations,
    eliminating the need to pass heap_bases to every function call.

    Usage:
        import iris
        from iris import DeviceContext

        # Host-side: Get encoded context tensor
        shmem = iris.iris()
        context_tensor = shmem.get_device_context()

        @triton.jit
        def my_kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            # Initialize device context from encoded tensor
            ctx = DeviceContext.initialize(context_tensor, rank, world_size)

            # Use object-oriented API
            data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
            ctx.store(buffer + offsets, data, to_rank=1, mask=mask)
            old_val = ctx.atomic_add(counter, 1, to_rank=1)

    Attributes:
        rank: Current rank (constexpr)
        world_size: Total number of ranks (constexpr)
        heap_bases: Heap base pointers for all ranks (tensor)
        trace_enabled: Whether tracing is enabled (constexpr)
        max_trace_events: Maximum number of trace events (constexpr)
        trace_counter: Pointer to atomic event counter (tensor)
        trace_buf_pid: Pointer to pid buffer (tensor)
        trace_buf_pid_m: Pointer to pid_m buffer (tensor)
        trace_buf_pid_n: Pointer to pid_n buffer (tensor)
        trace_buf_cur_rank: Pointer to cur_rank buffer (tensor)
        trace_buf_target_rank: Pointer to target_rank buffer (tensor)
        trace_buf_xcc_id: Pointer to xcc_id buffer (tensor)
        trace_buf_cu_id: Pointer to cu_id buffer (tensor)
        trace_buf_timestamp: Pointer to timestamp buffer (tensor)
        trace_buf_address: Pointer to address buffer (tensor)
    """

    rank: tl.constexpr
    world_size: tl.constexpr
    heap_bases: tl.tensor
    tracing: DeviceTracing

    @triton.constexpr_function
    def __init__(self, rank, world_size, heap_bases, tracing):
        """
        Internal constructor - use DeviceContext.initialize() instead.

        Args:
            rank: Current rank (constexpr)
            world_size: Total number of ranks (constexpr)
            heap_bases: Heap base pointers for all ranks (tensor)
            tracing: DeviceTracing instance
        """
        self.rank = tl.constexpr(rank)
        self.world_size = tl.constexpr(world_size)
        self.heap_bases = heap_bases
        self.tracing = tracing

    @staticmethod
    @triton.jit
    def initialize(context_tensor, rank, world_size, tracing: tl.constexpr = False):
        """
        Initialize DeviceContext from the encoded context tensor.

        The context tensor has the format:
        - [cur_rank, num_ranks, heap_base_0, ..., heap_base_N, trace_info...]
        - If tracing=True: extracts trace buffer pointers from context_tensor

        Args:
            context_tensor: Pointer to encoded context data (from Iris.get_device_context())
            rank: Current rank (must be constexpr in kernel signature)
            world_size: Total number of ranks (must be constexpr in kernel signature)
            tracing: Enable event tracing (constexpr, default: False)

        Returns:
            DeviceContext: Initialized device context

        Example:
            >>> import iris
            >>> from iris import DeviceContext
            >>>
            >>> ctx = iris.iris()
            >>> ctx.tracing.enable(max_events=1_000_000)
            >>> context_tensor = ctx.get_device_context()
            >>>
            >>> @triton.jit
            >>> def kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            >>>     # Without tracing
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size)
            >>>
            >>>     # With tracing
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size, tracing=True)
            >>>     mask = tl.full([64], True, dtype=tl.int1)  # Example mask
            >>>     ctx.tracing.record_event_start(event_id=TraceEvent().put, target_rank=1, address=ptr, pid_m=0, pid_n=0, mask=mask)
        """
        # Extract heap bases (from index 2 onwards)
        heap_bases = context_tensor + 2  # Offset pointer to start at heap bases

        if tracing:
            # Extract tracing info (starts after heap_bases)
            trace_info_idx = 2 + world_size + 1  # Skip: cur_rank, num_ranks, heap_bases, trace_enabled flag
            max_events = tl.load(context_tensor + trace_info_idx + 0)
            trace_counter_ptr = tl.load(context_tensor + trace_info_idx + 1)
            op_index_counter_ptr = tl.load(context_tensor + trace_info_idx + 2)

            # Cast counter pointers to pointer type
            trace_counter = tl.cast(trace_counter_ptr, tl.pointer_type(tl.int32))
            op_index_counter = tl.cast(op_index_counter_ptr, tl.pointer_type(tl.int32))

            # Extract trace buffer pointers (13 buffers)
            base_idx = trace_info_idx + 3  # Updated: +3 because we now have op_index_counter
            trace_buf_event_id = tl.cast(tl.load(context_tensor + base_idx + 0), tl.pointer_type(tl.int32))
            trace_buf_pid = tl.cast(tl.load(context_tensor + base_idx + 1), tl.pointer_type(tl.int32))
            trace_buf_pid_m = tl.cast(tl.load(context_tensor + base_idx + 2), tl.pointer_type(tl.int32))
            trace_buf_pid_n = tl.cast(tl.load(context_tensor + base_idx + 3), tl.pointer_type(tl.int32))
            trace_buf_cur_rank = tl.cast(tl.load(context_tensor + base_idx + 4), tl.pointer_type(tl.int32))
            trace_buf_target_rank = tl.cast(tl.load(context_tensor + base_idx + 5), tl.pointer_type(tl.int32))
            trace_buf_xcc_id = tl.cast(tl.load(context_tensor + base_idx + 6), tl.pointer_type(tl.int32))
            trace_buf_cu_id = tl.cast(tl.load(context_tensor + base_idx + 7), tl.pointer_type(tl.int32))
            trace_buf_timestamp = tl.cast(tl.load(context_tensor + base_idx + 8), tl.pointer_type(tl.int64))
            trace_buf_address = tl.cast(tl.load(context_tensor + base_idx + 9), tl.pointer_type(tl.int64))
            trace_buf_duration_cycles = tl.cast(tl.load(context_tensor + base_idx + 10), tl.pointer_type(tl.int64))
            trace_buf_op_index = tl.cast(tl.load(context_tensor + base_idx + 11), tl.pointer_type(tl.int32))
            trace_buf_payload_size = tl.cast(tl.load(context_tensor + base_idx + 12), tl.pointer_type(tl.int32))

            # Create DeviceTracing instance
            device_tracing = DeviceTracing(
                enabled=tracing,
                rank=rank,
                max_events=max_events,
                counter=trace_counter,
                op_index_counter=op_index_counter,
                buf_event_id=trace_buf_event_id,
                buf_pid=trace_buf_pid,
                buf_pid_m=trace_buf_pid_m,
                buf_pid_n=trace_buf_pid_n,
                buf_cur_rank=trace_buf_cur_rank,
                buf_target_rank=trace_buf_target_rank,
                buf_xcc_id=trace_buf_xcc_id,
                buf_cu_id=trace_buf_cu_id,
                buf_timestamp=trace_buf_timestamp,
                buf_address=trace_buf_address,
                buf_duration_cycles=trace_buf_duration_cycles,
                buf_op_index=trace_buf_op_index,
                buf_payload_size=trace_buf_payload_size,
            )

            return DeviceContext(rank, world_size, heap_bases, device_tracing)
        else:
            # When tracing disabled, use dummy pointers (never dereferenced; we return early in record_*)
            dummy_ptr_i32 = tl.cast(context_tensor, tl.pointer_type(tl.int32))
            dummy_ptr_i64 = tl.cast(context_tensor, tl.pointer_type(tl.int64))
            max_events_zero = tl.full((), 0, dtype=tl.int32)
            device_tracing = DeviceTracing(
                enabled=False,
                rank=rank,
                max_events=max_events_zero,
                counter=dummy_ptr_i32,
                op_index_counter=dummy_ptr_i32,
                buf_event_id=dummy_ptr_i32,
                buf_pid=dummy_ptr_i32,
                buf_pid_m=dummy_ptr_i32,
                buf_pid_n=dummy_ptr_i32,
                buf_cur_rank=dummy_ptr_i32,
                buf_target_rank=dummy_ptr_i32,
                buf_xcc_id=dummy_ptr_i32,
                buf_cu_id=dummy_ptr_i32,
                buf_timestamp=dummy_ptr_i64,
                buf_address=dummy_ptr_i64,
                buf_duration_cycles=dummy_ptr_i64,
                buf_op_index=dummy_ptr_i32,
                buf_payload_size=dummy_ptr_i32,
            )

            return DeviceContext(rank, world_size, heap_bases, device_tracing)

    @triton.jit
    def _translate(self, ptr, from_rank, to_rank, hint: tl.constexpr = None):
        """Internal pointer translation between rank address spaces."""
        return __translate(ptr, from_rank, to_rank, self.heap_bases, hint)

    @triton.jit
    def load(self, pointer, from_rank, mask=None, hint: tl.constexpr = None):
        """
        Loads a value from the specified rank's memory location.

        This method performs a memory read operation by translating the pointer
        from the current rank's address space to the `from_rank`'s address space and loading
        data from the target memory location. If the current rank and `from_rank` are the same,
        this performs a local load operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `from_rank`'s address space.
            from_rank (int): The rank ID from which to read the data.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address pointer[idx]. Defaults to None.
            hint (int or tuple, optional): Vectorization hint for the translated pointer. Defaults to None.

        Returns:
            Block: The loaded value from the target memory location.

        Example:
            >>> data = ctx.load(buffer + offsets, from_rank=1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.rank, from_rank, hint)
        result = tl.load(translated_ptr, mask=mask)
        return result

    @triton.jit
    def store(self, pointer, value, to_rank, mask=None, hint: tl.constexpr = None):
        """
        Writes data to the specified rank's memory location.

        This method performs a memory write operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and storing
        the provided data to the target memory location. If the current rank and `to_rank` are the same,
        this performs a local store operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space.
            value (Block): The tensor of elements to be stored.
            to_rank (int): The rank ID to which the data will be written.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.
            hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None.

        Returns:
            None

        Example:
            >>> ctx.store(buffer + offsets, values, to_rank=1, mask=mask)
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        tl.store(translated_ptr, value, mask=mask)

    @triton.jit
    def get(self, from_ptr, to_ptr, from_rank, mask=None, hint: tl.constexpr = None):
        """
        Copies data from the specified rank's memory into current rank's local memory.

        This method performs a remote load operation by translating `from_ptr` from the current
        rank's address space to the `from_rank`'s address space, loading the data, and storing
        it to `to_ptr` in the current rank's local memory. If the current rank and `from_rank`
        are the same, this performs a local copy operation.

        Args:
            from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references memory in `from_rank`.
            to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer to local memory in current rank where the data will be written.
            from_rank (int): The rank ID from which to read the data.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
            hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None.

        Returns:
            None

        Example:
            >>> ctx.get(remote_ptr + offsets, local_ptr + offsets, from_rank=1, mask=mask)
        """
        translated_from_ptr = self._translate(from_ptr, self.rank, from_rank, hint)
        data = tl.load(translated_from_ptr, mask=mask)
        tl.store(to_ptr, data, mask=mask)

    @triton.jit
    def put(self, from_ptr, to_ptr, to_rank, mask=None, hint: tl.constexpr = None):
        """
        Copies data from current rank's local memory to the specified rank's memory.

        This method performs a remote store operation by loading data from `from_ptr` in the
        current rank's local memory, translating `to_ptr` from the current rank's address space
        to the `to_rank`'s address space, and storing the data to the target memory location.
        If the current rank and `to_rank` are the same, this performs a local copy operation.

        Args:
            from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer to local memory in current rank from which to read data.
            to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references memory in `to_rank`.
            to_rank (int): The rank ID to which the data will be written.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
            hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None.

        Returns:
            None

        Example:
            >>> ctx.put(local_ptr + offsets, remote_ptr + offsets, to_rank=1, mask=mask)
        """
        translated_to_ptr = self._translate(to_ptr, self.rank, to_rank, hint)
        data = tl.load(from_ptr, mask=mask)
        tl.store(translated_to_ptr, data, mask=mask)

    @triton.jit
    def copy(self, src_ptr, dst_ptr, from_rank, to_rank, mask=None, hint: tl.constexpr = None):
        """
        Copies data from one rank's memory to another rank's memory.

        This method performs a data transfer by translating `src_ptr` from the current rank's
        address space to the `from_rank`'s address space, performing a masked load from the
        translated source, translating `dst_ptr` to the `to_rank`'s address space, and storing
        the loaded data to the target memory location. If `from_rank` and `to_rank` are the same,
        this performs a local copy operation. It is undefined behaviour if the current rank is
        neither `from_rank` nor `to_rank`.

        Args:
            src_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references `from_rank`'s local memory.
            dst_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that references `to_rank`'s local memory.
            from_rank (int): The rank ID that owns `src_ptr` (source rank).
            to_rank (int): The rank ID that will receive the data (destination rank).
            mask (Block of triton.int1, optional): If mask[idx] is false, do not load from src_ptr[idx] and do not store to dst_ptr[idx]. Defaults to None.

        Returns:
            None

        Example:
            >>> ctx.copy(src_ptr + offsets, dst_ptr + offsets, from_rank=1, to_rank=0, mask=mask)
        """
        cur_base = tl.load(self.heap_bases + self.rank)
        from_base = tl.load(self.heap_bases + from_rank)
        to_base = tl.load(self.heap_bases + to_rank)

        src_ptr_int = tl.cast(src_ptr, tl.uint64)
        src_offset = src_ptr_int - cur_base

        dst_ptr_int = tl.cast(dst_ptr, tl.uint64)
        dst_offset = dst_ptr_int - cur_base

        from_base_byte = tl.cast(from_base, tl.pointer_type(tl.int8))
        to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))

        translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
        translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

        if hint is not None:
            translated_src = tl.max_contiguous(tl.multiple_of(translated_src, hint), hint)
            translated_dst = tl.max_contiguous(tl.multiple_of(translated_dst, hint), hint)

        data = tl.load(translated_src, mask=mask)
        tl.store(translated_dst, data, mask=mask)

    @triton.jit
    def atomic_add(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic add at the specified rank's memory location.

        This method performs an atomic addition operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        adding the provided data to the `to_rank` memory location. If the current rank and
        `to_rank` are the same, this performs a local atomic addition operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.

        Example:
            >>> old_val = ctx.atomic_add(counter, 1, to_rank=1)
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_sub(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Atomically subtracts data from the specified rank's memory location.

        This method performs an atomic subtraction operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        subtracting the provided data from the `to_rank` memory location. If the current rank
        and `to_rank` are the same, this performs a local atomic subtraction operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The tensor of elements to be subtracted atomically.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_cas(self, pointer, cmp, val, to_rank, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic compare-and-swap at the specified rank's memory location.

        This method performs an atomic compare-and-swap operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        comparing the value at the memory location with `cmp`. If they match, it replaces the
        value with `val`. If the current rank and `to_rank` are the same, this performs a local
        atomic CAS operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory location in the current rank's address space that will be translated to the `to_rank`'s address space.
            cmp (Block): The expected value to compare against.
            val (Block): The new value to store if comparison succeeds.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)

    @triton.jit
    def atomic_xchg(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic exchange at the specified rank's memory location.

        This method performs an atomic exchange operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        swapping the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic exchange operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The new values to store.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_xor(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic XOR at the specified rank's memory location.

        This method performs an atomic bitwise XOR operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        XOR'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic XOR operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to XOR with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_and(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic AND at the specified rank's memory location.

        This method performs an atomic bitwise AND operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        AND'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic AND operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to AND with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_or(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic OR at the specified rank's memory location.

        This method performs an atomic bitwise OR operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        OR'ing the value at the memory location with `val`. If the current rank and `to_rank`
        are the same, this performs a local atomic OR operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to OR with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_min(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic minimum at the specified rank's memory location.

        This method performs an atomic minimum operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        updating the memory location to the minimum of its current value and `val`. If the
        current rank and `to_rank` are the same, this performs a local atomic min operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to compare with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)

    @triton.jit
    def atomic_max(self, pointer, val, to_rank, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
        """
        Performs an atomic maximum at the specified rank's memory location.

        This method performs an atomic maximum operation by translating the pointer
        from the current rank's address space to the `to_rank`'s address space and atomically
        updating the memory location to the maximum of its current value and `val`. If the
        current rank and `to_rank` are the same, this performs a local atomic max operation.

        Args:
            pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the current rank's address space that will be translated to the `to_rank`'s address space.
            val (Block): The values to compare with.
            to_rank (int): The rank ID to which the atomic operation will be performed.
            mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel", and "relaxed". Defaults to "acq_rel".
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect. Acceptable values are "gpu" (default), "cta", or "sys". The default value is "gpu".

        Returns:
            Block: The data stored at pointer before the atomic operation.
        """
        translated_ptr = self._translate(pointer, self.rank, to_rank, hint)
        return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None, hint: tl.constexpr = None):
    """
    Loads a value from the specified rank's memory location.

    This function performs a memory read operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and loading
    data from the target memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local load operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the pointer will be translated. Must be the current rank where the pointer is local.
        from_rank (int): The rank ID from which to read the data.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address pointer[idx]. Defaults to None.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        Block: The loaded value from the target memory location.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Load data from rank 1's memory into the current rank
        >>>     cur_rank = 0      # Current rank
        >>>     remote_rank = 1   # Remote rank to load from
        >>>     data = iris.load(ptr, cur_rank, remote_rank, heap_bases)
        >>>     return data
    """
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases, hint)
    result = tl.load(translated_ptr, mask=mask)
    return result


@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None, hint: tl.constexpr = None):
    """
    Writes data to the specified rank's memory location.

    This function performs a memory write operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and storing
    the provided data to the target memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local store operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        value (Block): The tensor of elements to be stored.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Store value 42 into rank 1's heap from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     value = 42
        >>>     iris.store(ptr, value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    tl.store(translated_ptr, value, mask=mask)


@triton.jit
def copy(src_ptr, dst_ptr, from_rank, to_rank, cur_rank, heap_bases, mask=None, hint: tl.constexpr = None):
    """
    Copies data from the specified rank's memory into the destination rank's memory.
    This function performs the transfer by translating `src_ptr` from the `from_rank`'s address
    space to the `to_rank`'s address space, performing a masked load from the translated
    source, and storing the loaded data to `dst_ptr` in the `to_rank` memory location.
    If `from_rank` and `to_rank` are the same, this function performs a local copy operation.
    It is undefined behaviour if neither `from_rank` nor `to_rank` is the `cur_rank`.

    Args:
        src_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s local memory from which to read data.
        dst_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `to_rank`'s local memory where the data will be written.
        from_rank (int): The rank ID that owns `src_ptr` (source rank).
        to_rank (int): The rank ID that will receive the data (destination rank).
        cur_rank (int): The rank ID issuing the copy operation. Must be either `from_rank` or `to_rank`.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load from the translated src_ptr[idx] and do not store to dst_ptr[idx]. Defaults to None.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointers. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.copy(remote_ptr, local_ptr, from_rank, to_rank, to_rank, heap_bases)
    """

    cur_base = tl.load(heap_bases + cur_rank)

    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)

    src_ptr_int = tl.cast(src_ptr, tl.uint64)
    src_offset = src_ptr_int - cur_base

    dst_ptr_int = tl.cast(dst_ptr, tl.uint64)
    dst_offset = dst_ptr_int - cur_base

    from_base_byte = tl.cast(from_base, tl.pointer_type(tl.int8))
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))

    translated_src = tl.cast(from_base_byte + src_offset, src_ptr.dtype)
    translated_dst = tl.cast(to_base_byte + dst_offset, src_ptr.dtype)

    if hint is not None:
        translated_src = tl.max_contiguous(tl.multiple_of(translated_src, hint), hint)
        translated_dst = tl.max_contiguous(tl.multiple_of(translated_dst, hint), hint)

    data = tl.load(translated_src, mask=mask)
    tl.store(translated_dst, data, mask=mask)


@triton.jit
def get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None, hint: tl.constexpr = None):
    """
    Copies data from the specified rank's memory to the current rank's local memory.

    This function performs a memory read operation by translating the `from_ptr`
    from the current rank's address space to the `from_rank`'s address space, loading data
    from the `from_rank` memory location, and storing it to the local `to_ptr`.
    If the `from_rank` is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `from_rank`'s address space. Must be the current rank where the pointer is local.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory where the data will be stored.
        from_rank (int): The `from_rank` ID from which to read the data.
        to_rank (int): The current rank ID where the data will be stored.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Use a scalar for 1-D (e.g. 16) or a tuple for N-D (e.g. (1, 16)). Defaults to None (no hint).

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.get(remote_ptr, local_ptr, from_rank, to_rank, heap_bases)
    """
    translated_from_ptr = __translate(from_ptr, from_rank, to_rank, heap_bases, hint)

    data = tl.load(translated_from_ptr, mask=mask)

    tl.store(to_ptr, data, mask=mask)


@triton.jit
def put(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    stride_tm: tl.constexpr = 0,
    stride_tn: tl.constexpr = 0,
    stride_fm: tl.constexpr = 0,
    stride_fn: tl.constexpr = 0,
    mask=None,
    hint: tl.constexpr = None,
    USE_COPY_ENGINE: tl.constexpr = False,
    IS_2D_COPY: tl.constexpr = False,
    from_base_ptr=None,
    to_base_ptr=None,
):
    """
    Copies data from the current rank's local memory to the specified rank's memory.
    This function performs a memory write operation by loading data from the current
    rank's `from_ptr`, translating the `to_ptr` from the current rank's address
    space to the `to_rank`'s address space, and storing the data to the `to_rank` memory location.

    Supports both 1D (flat/linear) and 2D (tiled) copies:
    - 1D copies: Used when stride_tm == 0 and stride_fm == 0 (default), uses linear SDMA packets
    - 2D copies: Used when strides are non-zero, uses sub-window SDMA packets for better performance

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space.
        from_rank (int): The current rank ID from which to read the data.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        copy_engine_ctx (tl.tensor): Copy engine context for SDMA operations.
        stride_tm (int, optional): Stride in M dimension for destination buffer (in elements). Default: 0 (flat copy).
        stride_tn (int, optional): Stride in N dimension for destination buffer (in elements). Default: 0.
        stride_fm (int, optional): Stride in M dimension for source buffer (in elements). Default: 0 (flat copy).
        stride_fn (int, optional): Stride in N dimension for source buffer (in elements). Default: 0.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load/copy data at that index. Defaults to None.
        USE_COPY_ENGINE (tl.constexpr, optional): Whether to use SDMA copy engine. Defaults to False (uses regular load/store).
        from_base_ptr (triton.PointerType, optional): Base pointer of the source buffer. Required for 2D copies when USE_COPY_ENGINE is True.
        to_base_ptr (triton.PointerType, optional): Base pointer of the destination buffer. Required for 2D copies when USE_COPY_ENGINE is True.

    Returns:
        None

    Examples:
        1D (flat) copy:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, heap_bases, copy_engine_ctx):
        >>>     from_rank = 0
        >>>     to_rank = 1
        >>>     offsets = tl.arange(0, 256)
        >>>     iris.put(local_ptr + offsets, remote_ptr + offsets,
        >>>              from_rank, to_rank, heap_bases, copy_engine_ctx,
        >>>              mask=offsets < 256, USE_COPY_ENGINE=True)

        2D (tiled) copy:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, heap_bases, copy_engine_ctx, base_ptr):
        >>>     from_rank = 0
        >>>     to_rank = 1
        >>>     iris.put(local_ptr, remote_ptr, from_rank, to_rank, heap_bases, copy_engine_ctx,
        >>>              stride_tm=1024, stride_fm=1024,
        >>>              mask=mask, USE_COPY_ENGINE=True,
        >>>              from_base_ptr=base_ptr, to_base_ptr=base_ptr)
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)

    if not USE_COPY_ENGINE:
        data = tl.load(from_ptr, mask=mask)

        tl.store(translated_to_ptr, data, mask=mask)
    else:
        ctx = copy_engine_ctx + (6 * to_rank)
        queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
        read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
        write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
        doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
        cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
        committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

        # dst_ptr_val = tl.min(translated_to_ptr.to(tl.uint64), axis=-1)
        dst_ptr_val0 = tl.min(translated_to_ptr.to(tl.uint64))
        # Extract source address (min of pointer block where data is stored)
        src_ptr_u64 = from_ptr.to(tl.uint64)
        # src_ptr_val = tl.min(src_ptr_u64, axis=-1)
        src_ptr_val0 = tl.min(src_ptr_u64)
        # max_src_ptr = tl.max(src_ptr_u64, axis=0)

        # Infer element size from pointer type
        # src_ptr is a block of pointers with a specific element type (e.g., pointer<float32>)
        # The pointer dtype tells us the element type, which has a known size
        # Map Triton dtypes to their byte sizes
        ptr_dtype = from_ptr.dtype.element_ty  # Get the element type that the pointer points to

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

        # Determine packet size based on copy type
        # Linear copy packet: 32 bytes for 1D, Sub-window copy packet: 80 bytes for 2D
        # IS_2D_COPY is a compile-time constant for proper branch elimination
        mask_int = mask.to(tl.int32)
        command_in_bytes_u32 = 80 if IS_2D_COPY else 32
        command_in_bytes = command_in_bytes_u32.to(tl.uint64)

        # Acquire space in the queue
        base, offset = anvil.acquire_fadd(
            queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
        )

        # Write padding NOPs if we wrapped around
        anvil.place_nop_packet(queue_ptr_u32, base, offset)

        # Place the appropriate packet type
        packet_offset_bytes = base + offset

        if not IS_2D_COPY:
            # For 1D copies, mask is 1D, so just sum all elements
            num_elements = tl.sum(mask_int, axis=0)
            size_bytes = (num_elements * element_size_bytes).to(tl.uint32)

            # Place linear copy packet for 1D/flat copies
            anvil.place_copy_packet(
                queue_ptr_u32,
                packet_offset_bytes,
                size_bytes,
                src_ptr_val0,
                dst_ptr_val0,
            )
        else:
            # For 2D copies, mask is 2D [M, N], use axis operations
            num_elements_per_stride = tl.max(tl.sum(mask_int, axis=-1))
            num_strides = tl.max(tl.sum(mask_int, axis=0))
            size_bytes = (num_elements_per_stride * element_size_bytes).to(tl.uint32)
            src_stride = (stride_fm * element_size_bytes).to(tl.uint32)
            dst_stride = (stride_tm * element_size_bytes).to(tl.uint32)

            # Place sub-window copy packet for 2D tiled copies
            # Calculate base addresses and offsets for sub-window copy
            src_base = from_base_ptr.to(tl.uint64)
            dst_base = __translate(to_base_ptr, from_rank, to_rank, heap_bases).to(tl.uint64)

            # Calculate tile offset from base
            tile_offset_bytes = src_ptr_val0 - src_base
            src_y_val = (tile_offset_bytes // src_stride).to(tl.uint32)
            src_x_val = (tile_offset_bytes % src_stride).to(tl.uint32)

            tile_offset_bytes_dst = dst_ptr_val0 - dst_base
            dst_y_val = (tile_offset_bytes_dst // dst_stride).to(tl.uint32)
            dst_x_val = (tile_offset_bytes_dst % dst_stride).to(tl.uint32)

            anvil.place_sub_window_copy_packet(
                queue_ptr_u32,
                packet_offset_bytes,
                src_base,
                dst_base,
                tile_width=size_bytes,
                tile_height=num_strides,
                src_buffer_pitch=src_stride,
                dst_buffer_pitch=dst_stride,
                src_x=src_x_val,
                src_y=src_y_val,
                dst_x=dst_x_val,
                dst_y=dst_y_val,
            )

        # Submit the command to the queue
        pending_wptr = base + offset + command_in_bytes
        anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def nontemporal_store(addr, value):
    tl.inline_asm_elementwise(
        asm="""flat_store_dwordx2 $1 $2 sc0 nt; s_waitcnt vmcnt(0)""",
        constraints=("=r,v,v"),  # =r used for dummy return to satisfy compiler requirement
        args=[addr, value],
        dtype=tl.int32,  # return not used
        is_pure=False,
        pack=1,
    )


# TODO rename or add nt
@triton.jit
def nontemporal_load(addr):
    val = tl.inline_asm_elementwise(
        asm="""flat_load_dwordx2 $0 $1 sc0 sc1; s_waitcnt vmcnt(0)""",
        constraints=("=v,v"),
        args=[addr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    return val


@triton.jit
def nontemporal_atomic_add(addr, value):
    old = tl.inline_asm_elementwise(
        asm="""flat_atomic_add_x2 $0 $1 sc0 sc1; s_waitcnt vmcnt(0)""",
        constraints=("=v,v,v"),
        args=[addr, value],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    return old


# @triton.jit
# def nontemporal_compare_exchange(addr, cmp_low, cmp_high, val_low, val_high):
#     # data_128bit = tl.cat([cmp_low, cmp_high, val_low, val_high])
#     data_128bit = tl.make_vector([cmp_low, cmp_high, val_low, val_high], type=tl.uint32)
#     old = tl.inline_asm_elementwise(
#         asm="""flat_atomic_cmpswap_x2 $0 $1 $2 sc0 nt; s_waitcnt vmcnt(0)""",
#         constraints=("=v,v,v"),
#         args=[addr, data_128bit],
#         dtype=tl.uint64,
#         is_pure=False,
#         pack=1,
#     )
#     return True # TODO if old == cmp else False


# @triton.jit
# def signal_ce(to_ptr, from_rank, to_rank, heap_bases, ce_handle, mask=None):
#     """
#     Copies data from the current rank's local memory to the specified rank's memory.
#     This function performs a memory write operation by loading data from the current
#     rank's `from_ptr`, translating the `to_ptr` from the current rank's address
#     space to the `to_rank`'s address space, and storing the data to the `to_rank` memory location.
#     If the `to_rank` is the same as the current rank, this function performs a local copy operation.

#     Args:
#         from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
#         to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
#         from_rank (int): The current rank ID from which to read the data.
#         to_rank (int): The `to_rank` ID to which the data will be written.
#         heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
#         mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

#     Returns:
#         None

#     Example:
#         >>> @triton.jit
#         >>> def kernel(local_ptr, remote_ptr, heap_bases):
#         >>>     from_rank = 0
#         >>>     to_rank = 1
#         >>>     iris.put(local_ptr, remote_ptr, from_rank, to_rank, heap_bases)
#     """

#     handle = ce_handle  # iris.get_copy_engine_handle(to_rank)
#     queue_ptr_u32 = tl.load(handle + 0).to(tl.pointer_type(tl.uint32))
#     read_ptr = tl.load(handle + 1).to(tl.pointer_type(tl.uint64))
#     write_ptr = tl.load(handle + 2).to(tl.pointer_type(tl.uint64))
#     doorbell_ptr = tl.load(handle + 3).to(tl.pointer_type(tl.uint64))
#     cached_write_ptr = tl.load(handle + 4).to(tl.pointer_type(tl.uint64))
#     committed_write_ptr = tl.load(handle + 5).to(tl.pointer_type(tl.uint64))

#     translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases)
#     dst_ptr_val = translated_to_ptr.to(tl.uint64)

#     command_in_bytes = 32
#     # Acquire space
#     base = anvil.acquire(queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes)

#     # Place command packet
#     slot_ptr_u32  = queue_ptr_u32 + (base // 4)
#     anvil.place_atomic_packet(slot_ptr_u32, dst_ptr_val)

#     # Submit command
#     anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, command_in_bytes)


@triton.jit
def atomic_add(
    pointer,
    val,
    from_rank,
    to_rank,
    heap_bases,
    mask=None,
    sem=None,
    scope=None,
    hint: tl.constexpr = None,
    copy_engine_ctx=None,
    USE_COPY_ENGINE: tl.constexpr = False,
):
    """
    Performs an atomic add at the specified rank's memory location.

    This function performs an atomic addition operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    adding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic addition operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically add 5 to rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     increment = 5
        >>>     old_val = iris.atomic_add(ptr, increment, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    if not USE_COPY_ENGINE:
        return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)
    else:
        handle = copy_engine_ctx + (6 * to_rank)
        queue_ptr_u32 = tl.load(handle + 0).to(tl.pointer_type(tl.uint32))
        read_ptr = tl.load(handle + 1).to(tl.pointer_type(tl.uint64))
        write_ptr = tl.load(handle + 2).to(tl.pointer_type(tl.uint64))
        doorbell_ptr = tl.load(handle + 3).to(tl.pointer_type(tl.uint64))
        cached_write_ptr = tl.load(handle + 4).to(tl.pointer_type(tl.uint64))
        committed_write_ptr = tl.load(handle + 5).to(tl.pointer_type(tl.uint64))

        dst_ptr_val = translated_ptr.to(tl.uint64)

        command_in_bytes = 32
        # Acquire space (returns base index and wraparound offset)
        base, offset = anvil.acquire_fadd(
            # base = anvil.acquire(
            queue_ptr_u32,
            read_ptr,
            write_ptr,
            doorbell_ptr,
            cached_write_ptr,
            committed_write_ptr,
            command_in_bytes,
        )
        # tl.device_print("offset ", offset)

        # Write padding NOPs if we wrapped around
        anvil.place_nop_packet(queue_ptr_u32, base, offset)

        # Calculate packet position (base + offset for wraparound)
        packet_offset_bytes = base + offset

        # Place command packet
        anvil.place_atomic_packet(queue_ptr_u32, packet_offset_bytes, dst_ptr_val, val)

        # Submit command
        pending_wptr = base + offset + command_in_bytes
        anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def put_signal(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    flag_ptr,
    flag_value,
    stride_tm: tl.constexpr,
    stride_fm: tl.constexpr,
    mask=None,
    hint: tl.constexpr = None,
    from_base_ptr=None,
    to_base_ptr=None,
):
    """
    Combines 2D copy (put) with atomic_add signal in one SDMA submission.

    This is equivalent to calling put() followed by atomic_add(), but batches both
    operations into a single SDMA queue submission for better performance.

    Args:
        from_ptr: Source pointer in current rank's local memory
        to_ptr: Destination pointer (will be translated to to_rank's address space)
        from_rank: Current rank ID
        to_rank: Remote rank ID to write to
        heap_bases: Array of heap base addresses for all ranks
        copy_engine_ctx: Copy engine context for SDMA operations
        flag_ptr: Pointer to flag location for signaling
        flag_value: Value to atomically add to flag (typically 1)
        stride_tm: Destination row stride in elements
        stride_fm: Source row stride in elements
        mask: 2D mask indicating which elements to copy
        hint: Vectorization hint for translated pointers
        from_base_ptr: Base pointer of source buffer (required for 2D)
        to_base_ptr: Base pointer of destination buffer (required for 2D)

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(local_A, remote_staged_a, flags, heap_bases, ctx, base_ptr):
        >>>     # Copy tile and signal completion
        >>>     m_offs = tl.arange(0, 256)[:, None]
        >>>     k_offs = tl.arange(0, 64)[None, :]
        >>>     mask = (m_offs < 256) & (k_offs < 64)
        >>>     iris.put_signal(local_A + offsets, remote_staged_a + offsets,
        >>>                     from_rank=0, to_rank=1, heap_bases=heap_bases,
        >>>                     copy_engine_ctx=ctx, flag_ptr=flags, flag_value=1,
        >>>                     stride_tm=1024, stride_fm=64, mask=mask,
        >>>                     from_base_ptr=base_ptr, to_base_ptr=base_ptr)
    """
    # Translate destination pointer
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)
    translated_flag_ptr = __translate(flag_ptr, from_rank, to_rank, heap_bases, hint)

    # Get copy engine context for target rank
    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    # Extract addresses
    dst_ptr_val = tl.min(translated_to_ptr.to(tl.uint64))
    src_ptr_u64 = from_ptr.to(tl.uint64)
    src_ptr_val = tl.min(src_ptr_u64)

    # Get element size from pointer type
    ptr_dtype = from_ptr.dtype.element_ty
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
        element_size_bytes = 4

    # Reserve space for BOTH packets: SUB_WINDOW_COPY (80 bytes) + ATOMIC (32 bytes) = 112 bytes
    command_in_bytes = 112

    # Acquire space in queue
    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )

    # Write padding NOPs if we wrapped around
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset

    # Calculate 2D copy parameters from mask
    mask_int = mask.to(tl.int32)
    num_elements_per_stride = tl.max(tl.sum(mask_int, axis=-1))
    num_strides = tl.max(tl.sum(mask_int, axis=0))
    size_bytes = (num_elements_per_stride * element_size_bytes).to(tl.uint32)
    src_stride = (stride_fm * element_size_bytes).to(tl.uint32)
    dst_stride = (stride_tm * element_size_bytes).to(tl.uint32)

    # Calculate base addresses and offsets
    src_base = from_base_ptr.to(tl.uint64)
    dst_base = __translate(to_base_ptr, from_rank, to_rank, heap_bases).to(tl.uint64)

    tile_offset_bytes = src_ptr_val - src_base
    src_y_val = (tile_offset_bytes // src_stride).to(tl.uint32)
    src_x_val = (tile_offset_bytes % src_stride).to(tl.uint32)

    tile_offset_bytes_dst = dst_ptr_val - dst_base
    dst_y_val = (tile_offset_bytes_dst // dst_stride).to(tl.uint32)
    dst_x_val = (tile_offset_bytes_dst % dst_stride).to(tl.uint32)

    # Place SUB_WINDOW_COPY packet (80 bytes)
    anvil.place_sub_window_copy_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        src_base,
        dst_base,
        tile_width=size_bytes,
        tile_height=num_strides,
        src_buffer_pitch=src_stride,
        dst_buffer_pitch=dst_stride,
        src_x=src_x_val,
        src_y=src_y_val,
        dst_x=dst_x_val,
        dst_y=dst_y_val,
    )

    # Place ATOMIC packet immediately after (32 bytes)
    atomic_offset_bytes = packet_offset_bytes + 80
    flag_dst_ptr_val = translated_flag_ptr.to(tl.uint64)
    anvil.place_atomic_packet(queue_ptr_u32, atomic_offset_bytes, flag_dst_ptr_val, flag_value)

    # Submit both packets in one doorbell ring
    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def put_signal_rect(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    flag_ptr,
    flag_value,
    width_bytes: tl.constexpr,
    height: tl.constexpr,
    src_pitch: tl.constexpr,
    dst_pitch: tl.constexpr,
    hint: tl.constexpr = None,
):
    """
    Combines 2D rectangular copy (put) with atomic_add signal in one SDMA submission.

    Unlike put_signal(), this function takes explicit dimensions instead of a mask,
    allowing arbitrarily large tiles without hitting Triton's tensor size limit.

    Args:
        from_ptr: Source base pointer (scalar) in current rank's local memory
        to_ptr: Destination base pointer (scalar) - will be translated to to_rank's address space
        from_rank: Current rank ID
        to_rank: Remote rank ID to write to
        heap_bases: Array of heap base addresses for all ranks
        copy_engine_ctx: Copy engine context for SDMA operations
        flag_ptr: Pointer to flag location for signaling
        flag_value: Value to atomically add to flag (typically 1)
        width_bytes: Width of rectangle in bytes
        height: Height of rectangle in rows
        src_pitch: Source row stride in bytes
        dst_pitch: Destination row stride in bytes
        hint: Vectorization hint for translated pointers

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(A_sharded, staged_a, flags, heap_bases, ctx):
        >>>     # Transfer 256 rows × 1024 bytes (128 elements × 4 bytes × 2 K-blocks)
        >>>     src_ptr = A_sharded + m_offset * stride_am + k_offset * stride_ak
        >>>     dst_ptr = staged_a + m_offset * stride_sa_m + k_offset * stride_sa_k
        >>>     iris.put_signal_rect(
        >>>         src_ptr, dst_ptr, 0, 1, heap_bases, ctx, flags, 1,
        >>>         width_bytes=1024, height=256,
        >>>         src_pitch=stride_am * 2, dst_pitch=stride_sa_m * 2
        >>>     )
    """
    # Translate destination pointers
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)
    translated_flag_ptr = __translate(flag_ptr, from_rank, to_rank, heap_bases, hint)

    # Get copy engine context for target rank
    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    # Extract addresses (scalar pointers)
    src_ptr_val = from_ptr.to(tl.uint64)
    dst_ptr_val = translated_to_ptr.to(tl.uint64)
    flag_dst_ptr_val = translated_flag_ptr.to(tl.uint64)

    # Reserve space for BOTH packets: SUB_WINDOW_COPY (80 bytes) + ATOMIC (32 bytes) = 112 bytes
    command_in_bytes = 112

    # Acquire space in queue
    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )

    # Write padding NOPs if we wrapped around
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset

    # Place SUB_WINDOW_COPY packet (80 bytes)
    # Using base pointers directly (no offset calculation needed)
    anvil.place_sub_window_copy_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        src_ptr_val,
        dst_ptr_val,
        tile_width=width_bytes,
        tile_height=height,
        src_buffer_pitch=src_pitch,
        dst_buffer_pitch=dst_pitch,
        src_x=0,  # Offset already baked into pointers
        src_y=0,
        dst_x=0,
        dst_y=0,
    )

    # Place ATOMIC packet immediately after (32 bytes)
    atomic_offset_bytes = packet_offset_bytes + 80
    anvil.place_atomic_packet(queue_ptr_u32, atomic_offset_bytes, flag_dst_ptr_val, flag_value)

    # Submit both packets in one doorbell ring
    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def wait_then_put_rect(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    wait_flag_ptr,
    wait_value,
    width_bytes: tl.constexpr,
    height: tl.constexpr,
    src_pitch: tl.constexpr,
    dst_pitch: tl.constexpr,
    hint: tl.constexpr = None,
):
    """
    Enqueue a POLL_REGMEM followed by a 2D SUB_WINDOW_COPY in one SDMA submission.

    This is the device-side counterpart to the host wait-then-put-tile path.
    The SDMA queue waits on a local flag and performs the copy autonomously
    after the producer has completed the corresponding batch.
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)

    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    poll_packet_bytes = 24
    copy_packet_bytes = 80
    command_in_bytes = poll_packet_bytes + copy_packet_bytes

    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset
    anvil.place_poll_regmem_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        wait_flag_ptr.to(tl.uint64),
        wait_value,
    )
    anvil.place_sub_window_copy_packet(
        queue_ptr_u32,
        packet_offset_bytes + poll_packet_bytes,
        from_ptr.to(tl.uint64),
        translated_to_ptr.to(tl.uint64),
        tile_width=width_bytes,
        tile_height=height,
        src_buffer_pitch=src_pitch,
        dst_buffer_pitch=dst_pitch,
        src_x=0,
        src_y=0,
        dst_x=0,
        dst_y=0,
    )

    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def wait_then_put_rects(
    from_base_ptr,
    to_base_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    wait_flag_ptr,
    wait_value,
    transfer_row_offsets,
    transfer_col_offsets,
    transfer_width_bytes,
    transfer_heights,
    transfer_start,
    transfer_count,
    stride_n_bytes,
    src_pitch: tl.constexpr,
    dst_pitch: tl.constexpr,
    MAX_RECTS: tl.constexpr,
    hint: tl.constexpr = None,
):
    """
    Enqueue one POLL_REGMEM followed by many 2D SUB_WINDOW_COPY packets.

    The copy list is provided as flattened metadata arrays plus a per-wave
    start/count pair so the poster can submit an entire wave with one queue
    reservation and one doorbell ring.
    """
    translated_to_base_ptr = __translate(to_base_ptr, from_rank, to_rank, heap_bases, hint)

    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    poll_packet_bytes = 24
    copy_packet_bytes = 80
    command_in_bytes = poll_packet_bytes + transfer_count * copy_packet_bytes

    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset
    anvil.place_poll_regmem_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        wait_flag_ptr.to(tl.uint64),
        wait_value,
    )

    from_base_val = from_base_ptr.to(tl.uint64)
    to_base_val = translated_to_base_ptr.to(tl.uint64)

    for i in range(MAX_RECTS):
        if i < transfer_count:
            transfer_idx = transfer_start + i
            row_offset = tl.load(transfer_row_offsets + transfer_idx)
            col_offset = tl.load(transfer_col_offsets + transfer_idx)
            width_bytes = tl.load(transfer_width_bytes + transfer_idx)
            height = tl.load(transfer_heights + transfer_idx)
            byte_offset = (row_offset.to(tl.uint64) * src_pitch) + (col_offset.to(tl.uint64) * stride_n_bytes)
            copy_offset_bytes = packet_offset_bytes + poll_packet_bytes + i * copy_packet_bytes
            anvil.place_sub_window_copy_packet(
                queue_ptr_u32,
                copy_offset_bytes,
                from_base_val + byte_offset,
                to_base_val + byte_offset,
                tile_width=width_bytes,
                tile_height=height,
                src_buffer_pitch=src_pitch,
                dst_buffer_pitch=dst_pitch,
                src_x=0,
                src_y=0,
                dst_x=0,
                dst_y=0,
            )

    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def wait_then_put_signal_rect(
    from_ptr,
    to_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    wait_flag_ptr,
    wait_value,
    signal_flag_ptr,
    signal_value,
    width_bytes: tl.constexpr,
    height: tl.constexpr,
    src_pitch: tl.constexpr,
    dst_pitch: tl.constexpr,
    hint: tl.constexpr = None,
):
    """
    Enqueue POLL_REGMEM + 2D SUB_WINDOW_COPY + ATOMIC in one SDMA submission.

    This is the device-side counterpart to host-side wait/copy/signal flows and is
    useful for marking receiver-visible completion after the final copy in a queue.
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases, hint)
    translated_signal_ptr = __translate(signal_flag_ptr, from_rank, to_rank, heap_bases, hint)

    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    poll_packet_bytes = 24
    copy_packet_bytes = 80
    atomic_packet_bytes = 32
    command_in_bytes = poll_packet_bytes + copy_packet_bytes + atomic_packet_bytes

    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset
    anvil.place_poll_regmem_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        wait_flag_ptr.to(tl.uint64),
        wait_value,
    )
    anvil.place_sub_window_copy_packet(
        queue_ptr_u32,
        packet_offset_bytes + poll_packet_bytes,
        from_ptr.to(tl.uint64),
        translated_to_ptr.to(tl.uint64),
        tile_width=width_bytes,
        tile_height=height,
        src_buffer_pitch=src_pitch,
        dst_buffer_pitch=dst_pitch,
        src_x=0,
        src_y=0,
        dst_x=0,
        dst_y=0,
    )
    anvil.place_atomic_packet(
        queue_ptr_u32,
        packet_offset_bytes + poll_packet_bytes + copy_packet_bytes,
        translated_signal_ptr.to(tl.uint64),
        signal_value,
    )

    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def wait_then_put_signal_rects(
    from_base_ptr,
    to_base_ptr,
    from_rank,
    to_rank,
    heap_bases,
    copy_engine_ctx: tl.tensor,
    wait_flag_ptr,
    wait_value,
    signal_flag_ptr,
    signal_value,
    transfer_row_offsets,
    transfer_col_offsets,
    transfer_width_bytes,
    transfer_heights,
    transfer_start,
    transfer_count,
    stride_n_bytes,
    src_pitch: tl.constexpr,
    dst_pitch: tl.constexpr,
    MAX_RECTS: tl.constexpr,
    hint: tl.constexpr = None,
):
    """
    Enqueue one POLL_REGMEM, many 2D SUB_WINDOW_COPY packets, and one ATOMIC.
    """
    translated_to_base_ptr = __translate(to_base_ptr, from_rank, to_rank, heap_bases, hint)
    translated_signal_ptr = __translate(signal_flag_ptr, from_rank, to_rank, heap_bases, hint)

    ctx = copy_engine_ctx + (6 * to_rank)
    queue_ptr_u32 = tl.load(ctx + 0).to(tl.pointer_type(tl.uint32))
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    write_ptr = tl.load(ctx + 2).to(tl.pointer_type(tl.uint64))
    doorbell_ptr = tl.load(ctx + 3).to(tl.pointer_type(tl.uint64))
    cached_write_ptr = tl.load(ctx + 4).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    poll_packet_bytes = 24
    copy_packet_bytes = 80
    atomic_packet_bytes = 32
    command_in_bytes = poll_packet_bytes + transfer_count * copy_packet_bytes + atomic_packet_bytes

    base, offset = anvil.acquire_fadd(
        queue_ptr_u32, read_ptr, write_ptr, doorbell_ptr, cached_write_ptr, committed_write_ptr, command_in_bytes
    )
    anvil.place_nop_packet(queue_ptr_u32, base, offset)

    packet_offset_bytes = base + offset
    anvil.place_poll_regmem_packet(
        queue_ptr_u32,
        packet_offset_bytes,
        wait_flag_ptr.to(tl.uint64),
        wait_value,
    )

    from_base_val = from_base_ptr.to(tl.uint64)
    to_base_val = translated_to_base_ptr.to(tl.uint64)
    for i in range(MAX_RECTS):
        if i < transfer_count:
            transfer_idx = transfer_start + i
            row_offset = tl.load(transfer_row_offsets + transfer_idx)
            col_offset = tl.load(transfer_col_offsets + transfer_idx)
            width_bytes = tl.load(transfer_width_bytes + transfer_idx)
            height = tl.load(transfer_heights + transfer_idx)
            byte_offset = (row_offset.to(tl.uint64) * src_pitch) + (col_offset.to(tl.uint64) * stride_n_bytes)
            copy_offset_bytes = packet_offset_bytes + poll_packet_bytes + i * copy_packet_bytes
            anvil.place_sub_window_copy_packet(
                queue_ptr_u32,
                copy_offset_bytes,
                from_base_val + byte_offset,
                to_base_val + byte_offset,
                tile_width=width_bytes,
                tile_height=height,
                src_buffer_pitch=src_pitch,
                dst_buffer_pitch=dst_pitch,
                src_x=0,
                src_y=0,
                dst_x=0,
                dst_y=0,
            )

    anvil.place_atomic_packet(
        queue_ptr_u32,
        packet_offset_bytes + poll_packet_bytes + transfer_count * copy_packet_bytes,
        translated_signal_ptr.to(tl.uint64),
        signal_value,
    )

    pending_wptr = base + offset + command_in_bytes
    anvil.submit(write_ptr, doorbell_ptr, committed_write_ptr, base, pending_wptr)


@triton.jit
def quiet(copy_engine_ctx: tl.tensor, to_rank):
    """
    Device-side equivalent of host_quiet for a single destination queue.

    Waits until the hardware read pointer catches up to the queue's committed
    write pointer, meaning all packets submitted to that SDMA queue have
    completed.
    """
    ctx = copy_engine_ctx + (6 * to_rank)
    read_ptr = tl.load(ctx + 1).to(tl.pointer_type(tl.uint64))
    committed_write_ptr = tl.load(ctx + 5).to(tl.pointer_type(tl.uint64))

    target_wptr = tl.load(committed_write_ptr, cache_modifier=".cv", volatile=True)
    while tl.load(read_ptr, cache_modifier=".cv", volatile=True) != target_wptr:
        pass

    # tl.debug_barrier()


@triton.jit
def atomic_sub(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Atomically subtracts data from the specified rank's memory location.

    This function performs an atomic subtraction operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
        subtracting the provided data from the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic subtraction operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block): The tensor of elements to be subtracted atomically.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The value at the memory location before the atomic subtraction.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically subtract 3 from rank 2's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 2   # Remote rank (destination)
        >>>     decrement = 3
        >>>     old_val = iris.atomic_sub(ptr, decrement, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None, hint: tl.constexpr = None):
    """
    Atomically compares and exchanges the specified rank's memory location.

    This function performs an atomic compare-and-swap operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    comparing the current value with the expected value, then writing the new value if they match.
    If the `from_rank` and `to_rank` are the same, this function performs a local atomic compare-and-swap operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        cmp (Block): The expected value to be compared with the current value at the memory location.
        val (Block): The new value to be written if the compare succeeds.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The value contained at the memory location before the atomic operation attempt.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Compare-and-swap on rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     expected = 0
        >>>     new_val = 42
        >>>     old_val = iris.atomic_cas(ptr, expected, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic exchange at the specified rank's memory location.

    This function performs an atomic exchange operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    exchanging the current value with the provided new value. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic exchange operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Exchange value with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_value = 99
        >>>     old_val = iris.atomic_xchg(ptr, new_value, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_xor(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic xor at the specified rank's memory location.

    This function performs an atomic xor operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    xoring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic xor operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically XOR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xFF
        >>>     old_val = iris.atomic_xor(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_and(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic and at the specified rank's memory location.

    This function performs an atomic and operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    anding the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic and operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically AND with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0x0F
        >>>     old_val = iris.atomic_and(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None):
    """
    Performs an atomic or at the specified rank's memory location.

    This function performs an atomic or operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    oring the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic or operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically OR with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     mask_val = 0xF0
        >>>     old_val = iris.atomic_or(ptr, mask_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_min(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic min at the specified rank's memory location.

    This function performs an atomic min operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the min on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic min operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find minimum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 10
        >>>     old_val = iris.atomic_min(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_max(
    pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None, hint: tl.constexpr = None
):
    """
    Performs an atomic max at the specified rank's memory location.

    This function performs an atomic max operation by translating the pointer
    from the `from_rank`'s address space to the `to_rank`'s address space and atomically
    performing the max on the provided data to the `to_rank` memory location. If the `from_rank` and `to_rank` are the same,
    this function performs a local atomic max operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the `from_rank`'s address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
        hint (int or tuple, optional): Vectorization hint passed to tl.multiple_of / tl.max_contiguous on the translated pointer. Defaults to None (no hint).

    Returns:
        Block: The data stored at pointer before the atomic operation.

    Example:
        >>> @triton.jit
        >>> def kernel(ptr, heap_bases):
        >>>     # Atomically find maximum with rank 1's memory from rank 0
        >>>     cur_rank = 0      # Current rank (source)
        >>>     remote_rank = 1   # Remote rank (destination)
        >>>     new_val = 100
        >>>     old_val = iris.atomic_max(ptr, new_val, cur_rank, remote_rank, heap_bases)
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases, hint)
    return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


def iris(heap_size=1 << 30, allocator_type="torch"):
    """
    Create and return an Iris instance with the specified heap size.

    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.
        allocator_type (str): Type of allocator to use. Options: "torch" (default), "vmem".
                              Can be overridden with IRIS_ALLOCATOR environment variable.

    Returns:
        Iris: An initialized Iris instance.

    Example:
        >>> import iris
        >>> iris_ctx = iris.iris(2**30)  # 1GB heap with default (torch) allocator
        >>> tensor = iris_ctx.zeros(1024, 1024)

        >>> # Use VMem allocator
        >>> iris_ctx = iris.iris(2**30, allocator_type="vmem")
        >>> tensor = iris_ctx.zeros(1024, 1024)
    """
    return Iris(heap_size, allocator_type)
