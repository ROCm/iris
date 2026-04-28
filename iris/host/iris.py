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

import os

from iris.host.distributed.helpers import (
    init_distributed,
    distributed_barrier,
    distributed_device_barrier,
    distributed_broadcast_scalar,
    distributed_broadcast_tensor,
)
from iris.host.platform.hip import (
    set_device,
    get_cu_count,
    count_devices,
)
from iris.host.memory.symmetric_heap import SymmetricHeap
import numpy as np
from typing import Any
import torch
import logging

# Import logging functionality from the separate logging module
from iris.host.logging.logging import logger

# Import tracing functionality
from iris.host.tracing.core import Tracing  # noqa: F401
from iris.host.tracing.events import TraceEvent  # noqa: F401  re-export for iris.TraceEvent
from iris.mem.triton.tracing import Tracing as DeviceTracing  # noqa: F401

# Import shared tensor-creation helpers
from iris.host.memory import tensors as tensor_creation
from iris.host.platform.utils import is_simulation_env


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

        if logger.isEnabledFor(logging.INFO):
            self._log_with_rank(
                logging.INFO,
                f"init: heap_size={heap_size / (1 << 30):.1f}GB rank={cur_rank}/{num_ranks} allocator={allocator_type}",
            )

        # Initialize symmetric heap with specified allocator
        self.heap = SymmetricHeap(heap_size, gpu_id, cur_rank, num_ranks, allocator_type)
        self.device = f"cuda:{gpu_id}"
        self.heap_bases = self.heap.get_heap_bases()

        if is_simulation_env():
            import json

            heap_bases_list = [int(self.heap_bases[r].item()) for r in range(self.num_ranks)]
            prefix = os.environ.get("IRIS_HEAP_BASES_PREFIX", "iris")
            out_path = f"{prefix}_rank_{self.cur_rank}_heap_bases.json"
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

        # Initialize CCL interface
        self.ccl = self.CCL(self)

        # Lazy initialization for ops interface
        self._ops = None

        # Device-side barrier state, keyed by process group (None = all ranks).
        self._device_barrier_state: dict[Any, torch.Tensor] = {}

        # Initialize tracing
        self.tracing = Tracing(self)

        # Pre-build the device context tensor (rebuilt when tracing is enabled)
        self._build_device_context()

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

    def broadcast(self, value, src=0):
        """
        Broadcast a value from one rank to all ranks.

        This method automatically detects the type of value and uses the appropriate
        broadcast mechanism:
        - For tensors and arrays: uses efficient PyTorch distributed tensor collectives
        - For scalars and other objects: uses object broadcast

        Matches ``torch.distributed.broadcast`` parameter naming.

        Args:
            value (Any): The value to broadcast. Can be a scalar, tensor, numpy array,
                or any picklable object. Only the ``src`` rank's value is used;
                other ranks should pass a placeholder (e.g., ``None``).
            src (int): Source rank that holds the authoritative value.

        Returns:
            Any: The value broadcast to all ranks. Tensors and arrays are returned as
                numpy arrays; scalars and objects are returned in their original type.

        Examples:
            >>> ctx = iris.iris()
            >>> # Broadcasting a scalar
            >>> value = 42 if ctx.cur_rank == 0 else None
            >>> value = ctx.broadcast(value, src=0)  # All ranks get 42
            >>>
            >>> # Broadcasting a tensor
            >>> if ctx.cur_rank == 0:
            >>>     data = torch.randn(10, 10)
            >>> else:
            >>>     data = None
            >>> data = ctx.broadcast(data, src=0)  # All ranks get the same array
        """
        # Check if the value on src rank is a tensor or array-like
        if self.cur_rank == src and value is not None:
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
        is_tensor = distributed_broadcast_scalar(is_tensor, src)

        if is_tensor:
            return distributed_broadcast_tensor(value, root=src)
        else:
            return distributed_broadcast_scalar(value, src)

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

    def _build_device_context(self):
        """
        Build and cache the device context tensor.

        Called during __init__ and again after tracing.enable() to include tracing fields.
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
            # Pad with zeros so kernels compiled with tracing=True can safely
            # decode without reading out of bounds (max_events=0 prevents writes).
            # Layout: trace_enabled(1) + max_events(1) + counters(2) + buffers(13) = 17
            context_data += [0] * 17

        self._device_context = torch.tensor(context_data, dtype=torch.int64, device=self.device)

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
            >>> context_tensor = ctx.get_device_context()
            >>>
            >>> @triton.jit
            >>> def my_kernel(context_tensor, rank: tl.constexpr, world_size: tl.constexpr, ...):
            >>>     ctx = DeviceContext.initialize(context_tensor, rank, world_size)
            >>>     data = ctx.load(buffer, from_rank=1)
        """
        return self._device_context

    def barrier(self, stream=None, group=None):
        """
        Synchronize ranks within the specified group and their CUDA devices.

        This first calls ``torch.cuda.synchronize()`` or ``stream.synchronize()`` to ensure the local GPU has
        finished all queued work, then performs a distributed barrier so that all
        ranks in the group reach the same point before proceeding.

        Args:
            stream: If stream is given: wait only for that stream before barrier. If stream is None: legacy behavior (device-wide sync).
            group (ProcessGroup, optional): The process group to synchronize.
                If None, uses the default process group (all ranks).

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> ctx.barrier()  # Synchronize all ranks
            >>> ctx.barrier(group=my_group)  # Synchronize only ranks in my_group
        """
        self._log_with_rank(logging.DEBUG, "barrier: start")
        # Wait for all GPUs to finish work
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()

        # Distributed barrier
        distributed_barrier(group=group)

    def device_barrier(self, group=None):
        """
        Device-side barrier that is CUDA graph capturable.

        Unlike ``barrier()`` which uses host-side ``torch.distributed.barrier()``,
        this uses device-side atomic operations on the symmetric heap to synchronize
        ranks. Stateless w.r.t. host-side epoch tracking: each rank's flag on
        the heap serves as its own epoch counter, managed entirely by the GPU
        via atomic_add. A persistent per-group flags tensor is cached in
        ``_device_barrier_state``.

        Args:
            group (ProcessGroup, optional): The process group to synchronize.
                If None, uses all ranks in the shmem context.

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> ctx.device_barrier()  # Synchronize all ranks on device
        """
        if group not in self._device_barrier_state:
            self._device_barrier_state[group] = self.zeros((self.num_ranks,), dtype=torch.int32)

        distributed_device_barrier(
            self._device_barrier_state[group],
            group,
            self.cur_rank,
            self.num_ranks,
            self.get_heap_bases(),
        )

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
