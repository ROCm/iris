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

Example:
    >>> import iris
    >>> ctx = iris.iris(heap_size=2**30)  # 1GB heap
    >>> tensor = ctx.zeros(1024, 1024, dtype=torch.float32)
"""

import triton
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
import math
import torch
import logging

# Import logging functionality from the separate logging module
from .logging import logger


class Iris:
    """
    Main Iris class for multi-GPU communication and memory management.

    This class provides a unified interface for distributed GPU operations including
    memory allocation, atomic operations, and inter-rank communication.

    Args:
        heap_size (int): Size of the symmetric heap in bytes. Default: 1GB (2^30)

    Example:
        >>> ctx = iris.iris(heap_size=2**31)  # 2GB heap
        >>> print(f"Rank {ctx.cur_rank} of {ctx.num_ranks}") # Rank 0 of 1
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

        # Initialize symmetric heap
        self.heap = SymmetricHeap(heap_size, gpu_id, cur_rank, num_ranks)
        self.device = f"cuda:{gpu_id}"
        self.heap_bases = self.heap.get_heap_bases()

        for i in range(num_ranks):
            self.debug(f"GPU {i}: Heap base {hex(int(self.heap_bases[i].item()))}")

        distributed_barrier()

        # Initialize CCL interface
        self.ccl = self.CCL(self)

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

    def broadcast(self, value, source_rank):
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

    def __allocate(self, num_elements, dtype):
        """Allocate memory using the symmetric heap."""
        self.debug(f"allocate: num_elements = {num_elements}, dtype = {dtype}")
        return self.heap.allocate(num_elements, dtype)

    def __parse_size(self, size):
        # Handle nested tuples/lists by flattening them recursively
        while len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        num_elements = math.prod(size)
        return size, num_elements

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
        self.debug(
            f"zeros_like: input_shape = {input.shape}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
        )

        # Use input's properties as defaults if not specified
        if dtype is None:
            dtype = input.dtype
        if layout is None:
            layout = input.layout
        if device is None:
            device = input.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Get the size from input tensor
        size = input.size()
        num_elements = input.numel()

        # Allocate new tensor with the same size
        new_tensor = self.__allocate(num_elements, dtype)
        new_tensor.zero_()

        # Reshape to match input size
        new_tensor = new_tensor.reshape(size)

        # Apply the requested memory format
        new_tensor = self.__apply_memory_format(new_tensor, size, memory_format, input)

        # Apply the requested layout
        new_tensor = self.__apply_layout(new_tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            new_tensor.requires_grad_()

        return new_tensor

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
        self.debug(f"arange: start = {start}, end = {end}, step = {step}, dtype = {dtype}, device = {device}")

        # Handle the case where only one argument is provided (end)
        if end is None:
            end = start
            start = 0

        # Validate inputs
        if step == 0:
            raise ValueError("step must be non-zero")

        # Validate step direction consistency
        if step > 0 and start >= end:
            raise ValueError(f"Invalid range: start >= end with positive step (start={start}, end={end}, step={step})")
        elif step < 0 and start <= end:
            raise ValueError(f"Invalid range: start <= end with negative step (start={start}, end={end}, step={step})")

        # Calculate the number of elements
        num_elements = math.ceil((end - start) / step)

        # Infer dtype if not provided
        if dtype is None:
            if any(isinstance(x, float) for x in [start, end, step]):
                dtype = torch.get_default_dtype()
            else:
                dtype = torch.int64

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)

        target_device = tensor.device
        arange_tensor = torch.arange(start, end, step, dtype=dtype, device=target_device)

        tensor[:] = arange_tensor

        tensor = self.__apply_layout(tensor, layout)

        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(f"zeros: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with zeros
            out.zero_()
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Fill with zeros
            tensor.zero_()
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(
            f"randn: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Generate random data and copy to out tensor
            random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
            out.copy_(random_data)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Generate random data and copy to tensor
            random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
            tensor.copy_(random_data)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(f"ones: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with ones
            out.fill_(1)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Fill with ones
            tensor.fill_(1)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(
            f"full: size = {size}, fill_value = {fill_value}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
        )

        # Infer dtype from fill_value if not provided
        if dtype is None:
            if isinstance(fill_value, (int, float)):
                if isinstance(fill_value, float):
                    dtype = torch.get_default_dtype()
                else:
                    dtype = torch.int64
            else:
                # For other types (like tensors), use their dtype
                dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with the specified value
            out.fill_(fill_value)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Fill with the specified value
            tensor.fill_(fill_value)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(f"uniform: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        size, num_elements = self.__parse_size(size)
        tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
        tensor.uniform_(low, high)
        return tensor.reshape(size)

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
        self.debug(
            f"empty: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Apply the requested memory format
        tensor = self.__apply_memory_format(tensor, size, memory_format)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(f"randint: args = {args}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

        # Parse arguments to determine low, high, and size
        # PyTorch randint signatures:
        # randint(high, size) - where high is the upper bound and size is the shape
        # randint(low, high, size) - where low and high are bounds, size is the shape
        if len(args) == 2:
            # randint(high, size)
            high, size = args
            low = 0
        elif len(args) == 3:
            # randint(low, high, size)
            low, high, size = args
        else:
            raise ValueError(f"randint expects 2 or 3 positional arguments, got {len(args)}")

        # Use default dtype if None is provided
        if dtype is None:
            dtype = torch.int64

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Generate random integers using PyTorch's randint
        # Use specified device or fall back to current device
        target_device = device if device is not None else self.device

        # Handle generator parameter
        if generator is not None:
            torch.randint(low, high, size, generator=generator, out=tensor, dtype=dtype, device=target_device)
        else:
            torch.randint(low, high, size, out=tensor, dtype=dtype, device=target_device)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(
            f"linspace: start = {start}, end = {end}, steps = {steps}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            # Check if start or end are complex numbers
            start_is_complex = isinstance(start, complex) or (hasattr(start, "dtype") and torch.is_complex(start))
            end_is_complex = isinstance(end, complex) or (hasattr(end, "dtype") and torch.is_complex(end))

            if start_is_complex or end_is_complex:
                # Infer complex dtype based on default dtype
                dtype = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128
            else:
                dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse steps and extract the integer value
        if isinstance(steps, (tuple, list)):
            if len(steps) == 1:
                # Single-element tuple/list like (5,) or [5]
                steps_int = steps[0]
                # Handle nested tuples like ((5,),)
                if isinstance(steps_int, (tuple, list)):
                    steps_int = steps_int[0]
            else:
                # Multi-element tuple/list - use __parse_size for compatibility
                size, num_elements = self.__parse_size(steps)
                steps_int = num_elements
        else:
            # steps is a single integer
            steps_int = steps

        # Ensure steps_int is an integer
        steps_int = int(steps_int)
        size = (steps_int,)
        num_elements = steps_int

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Generate linspace using PyTorch's linspace
        # Use specified device or fall back to current device
        target_device = device if device is not None else self.device
        torch.linspace(start, end, steps_int, out=tensor, dtype=dtype, device=target_device)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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
        self.debug(
            f"rand: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Use current device if none specified
        if device is None:
            device = self.device

        # Validate device compatibility with Iris
        self.__throw_if_invalid_device(device)

        # Parse size and calculate number of elements
        size, num_elements = self.__parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.__allocate(num_elements=num_elements, dtype=dtype)
            # Reshape to the desired size
            tensor = tensor.reshape(size)

        # Generate random numbers using PyTorch's rand
        # Use specified device (already validated and set above)

        # Handle generator parameter
        if generator is not None:
            torch.rand(size, generator=generator, out=tensor, dtype=dtype, device=device)
        else:
            torch.rand(size, out=tensor, dtype=dtype, device=device)

        # Apply the requested layout
        tensor = self.__apply_layout(tensor, layout)

        # Set requires_grad if specified
        if requires_grad:
            tensor.requires_grad_()

        return tensor

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

    def barrier(self, stream=None):
        """
        Synchronize all ranks and their CUDA devices.

        This first calls ``torch.cuda.synchronize()`` or ``stream.synchronize()`` to ensure the local GPU has
        finished all queued work, then performs a global distributed barrier so that all
        ranks reach the same point before proceeding.
        Args:
            stream: If stream is given: wait only for that stream before barrier. If stream is None: legacy behavior (device-wide sync).

        Example:
            >>> ctx = iris.iris(1 << 20)
            >>> ctx.barrier()  # Synchronize all ranks
        """
        # Wait for all GPUs to finish work
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()

        # Distributed barrier
        distributed_barrier()

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

    def __throw_if_invalid_output_tensor(self, tensor: torch.Tensor, num_elements: int, dtype: torch.dtype):
        if not self.__tensor_on_device(tensor):
            raise RuntimeError(
                f"The output tensor is not on the same device as the Iris instance. The Iris instance is on device {self.device} but the output tensor is on device {tensor.device}"
            )
        if not self.__on_symmetric_heap(tensor):
            raise RuntimeError(
                f"The output tensor is not on the symmetric heap. The Iris instance is on heap base {self.heap_bases[self.cur_rank]} but the output tensor is on heap base {tensor.data_ptr()}"
            )
        if tensor.numel() != num_elements:
            raise RuntimeError(f"The output tensor has {tensor.numel()} elements, but {num_elements} are required")
        if tensor.dtype != dtype:
            raise RuntimeError(f"The output tensor has dtype {tensor.dtype}, but {dtype} is required")

    def __throw_if_invalid_device(self, device):
        """
        Throw a RuntimeError if the requested device is not compatible with this Iris instance.

        Args:
            device: The requested device (can be string, torch.device, or None)

        Raises:
            RuntimeError: If the device is not compatible
        """
        if not self.__is_valid_device(device):
            raise RuntimeError(
                f"Device mismatch: requested device {device} but Iris instance is on device {self.device}. "
                f"Iris only supports tensors on its own device."
            )

    def __apply_memory_format(
        self, tensor: torch.Tensor, size: tuple, memory_format: torch.memory_format, input_tensor: torch.Tensor = None
    ):
        """
        Apply the requested memory format to a tensor by setting appropriate strides.
        This keeps the tensor on the symmetric heap while changing how PyTorch interprets the memory layout.

        Args:
            tensor: The tensor to modify
            size: The tensor's size/dimensions
            memory_format: The desired memory format
            input_tensor: The original input tensor (needed for preserve_format detection)
        """
        if memory_format == torch.contiguous_format:
            # Default format, no changes needed
            return tensor
        elif memory_format == torch.channels_last and len(size) == 4:
            # For channels_last format: preserve shape (N, C, H, W) but change strides
            # channels_last strides: [C*H*W, 1, C*W, C] for shape (N, C, H, W)
            N, C, H, W = size[0], size[1], size[2], size[3]
            # Keep the original shape (N, C, H, W) but use channels_last strides
            tensor = self.__create_tensor_with_strides(tensor, size, (C * H * W, 1, C * W, C))
            return tensor
        elif memory_format == torch.channels_last_3d and len(size) == 5:
            # For channels_last_3d format: preserve shape (N, C, D, H, W) but change strides
            # channels_last_3d strides: [C*D*H*W, 1, C*D*W, C*W, C] for shape (N, C, D, H, W)
            N, C, D, H, W = size[0], size[1], size[2], size[3], size[4]
            # Keep the original shape (N, C, D, H, W) but use channels_last_3d strides
            tensor = self.__create_tensor_with_strides(tensor, size, (C * D * H * W, 1, C * D * W, C * W, C))
            return tensor
        elif memory_format == torch.preserve_format:
            # For preserve_format, we need to detect the input tensor's memory format
            # and apply the same format to the output
            if input_tensor is not None:
                # Check the actual memory format of the input tensor
                if len(size) == 4:
                    # Check if input tensor is in channels_last format by examining strides
                    # channels_last format has strides[1] == 1 (channels dimension is contiguous)
                    input_strides = input_tensor.stride()
                    if len(input_strides) == 4 and input_strides[1] == 1:
                        # Input is in channels_last format, preserve it
                        # Use the input tensor's actual shape, not the size parameter
                        input_shape = input_tensor.shape
                        if len(input_shape) == 4:
                            # Input is already in channels_last format (N, H, W, C)
                            new_size = input_shape
                            # Use the input tensor's strides directly
                            tensor = self.__create_tensor_with_strides(tensor, new_size, input_strides)
                            return tensor
                elif len(size) == 5:
                    # Check if input tensor is in channels_last_3d format
                    input_strides = input_tensor.stride()
                    if len(input_strides) == 5 and input_strides[1] == 1:
                        # Input is in channels_last_3d format, preserve it
                        # Use the input tensor's actual shape, not the size parameter
                        input_shape = input_tensor.shape
                        if len(input_shape) == 5:
                            # Input is already in channels_last_3d format (N, D, H, W, C)
                            new_size = input_shape
                            # Use the input tensor's strides directly
                            tensor = self.__create_tensor_with_strides(tensor, new_size, input_strides)
                            return tensor
            # If no special format detected or no input tensor provided, use contiguous format
            return tensor
        else:
            # Unsupported format or dimension combination
            self.debug(
                f"Warning: Memory format {memory_format} not supported for {len(size)}D tensor, using contiguous format"
            )
            # For unsupported formats, return the tensor as-is (contiguous)
            return tensor

    def __create_tensor_with_strides(self, original_tensor: torch.Tensor, size: tuple, strides: tuple) -> torch.Tensor:
        """
        Create a new tensor with the specified strides while keeping the data on the symmetric heap.

        Args:
            original_tensor: The original tensor (source of data and heap allocation)
            size: The tensor's size/dimensions
            strides: The desired strides for the new memory format

        Returns:
            A new tensor with the specified strides, data copied from original, on the same heap
        """

        # First, create a temporary tensor with the correct strides using PyTorch
        temp_tensor = torch.empty_strided(size, strides, dtype=original_tensor.dtype, device=original_tensor.device)

        # Handle different cases based on whether size changes and what the strides indicate
        if size != original_tensor.shape:
            # Size is different - this might be a format change that requires permutation
            # Check if this is a channels_last format by comparing strides
            if len(size) == 4:
                # For channels_last: expected strides are [H*W*C, 1, W*C, C] for shape (N, H, W, C)
                N, H, W, C = size[0], size[1], size[2], size[3]
                expected_strides = (H * W * C, 1, W * C, C)
                if strides == expected_strides:
                    permuted = original_tensor.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                else:
                    # If the size differs for other reasons, do not permute; just reshape if possible
                    try:
                        permuted = original_tensor.reshape(size)
                    except Exception:
                        raise ValueError(
                            "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                        )
            elif len(size) == 5:
                # For channels_last_3d: expected strides are [D*H*W*C, 1, H*W*C, W*C, C] for shape (N, D, H, W, C)
                N, D, H, W, C = size[0], size[1], size[2], size[3], size[4]
                expected_strides = (D * H * W * C, 1, H * W * C, W * C, C)
                if strides == expected_strides:
                    permuted = original_tensor.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
                else:
                    # If the size differs for other reasons, do not permute; just reshape if possible
                    try:
                        permuted = original_tensor.reshape(size)
                    except Exception:
                        raise ValueError(
                            "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                        )
            else:
                # For other dimensions, just try to reshape
                try:
                    permuted = original_tensor.reshape(size)
                except Exception:
                    raise ValueError(
                        "Cannot safely permute or reshape tensor: size differs from original shape for unknown reason."
                    )
        else:
            # Size is the same - this is a stride-only change (like channels_last with preserved shape)
            # We need to reorder the data to match the new stride pattern
            if len(size) == 4:
                # Check if this is channels_last format with preserved shape
                N, C, H, W = size[0], size[1], size[2], size[3]
                expected_strides = (C * H * W, 1, C * W, C)
                if strides == expected_strides:
                    permuted = original_tensor
                else:
                    permuted = original_tensor
            elif len(size) == 5:
                # Check if this is channels_last_3d format with preserved shape
                N, C, D, H, W = size[0], size[1], size[2], size[3], size[4]
                expected_strides = (C * D * H * W, 1, C * D * W, C * W, C)
                if strides == expected_strides:
                    permuted = original_tensor
                else:
                    permuted = original_tensor
            else:
                permuted = original_tensor

        # Copy the permuted data to the temporary tensor
        temp_tensor.copy_(permuted)

        # Now allocate a new tensor on our symmetric heap
        num_elements = math.prod(size)
        heap_tensor = self.__allocate(num_elements, original_tensor.dtype)

        # Reshape to the desired size
        heap_tensor = heap_tensor.reshape(size)

        # Copy the data from the temporary tensor to our heap tensor
        heap_tensor.copy_(temp_tensor)

        # Clean up the temporary tensor
        del temp_tensor

        # Now we need to create a view with the correct strides
        # We can't use as_strided directly on our heap tensor, but we can
        # create a new tensor with the right strides and copy the data again
        final_tensor = torch.as_strided(heap_tensor, size, strides)

        return final_tensor

    def __apply_layout(self, tensor: torch.Tensor, layout: torch.layout) -> torch.Tensor:
        """
        Apply the requested layout to a tensor.

        Args:
            tensor: The tensor to modify
            layout: The desired layout

        Returns:
            Tensor with the requested layout
        """

        if layout == torch.strided:
            # Strided layout is the default - no changes needed
            return tensor
        else:
            # Only support strided layout for now
            raise ValueError(f"Layout {layout} not supported. Only torch.strided is currently supported.")

    def __tensor_on_device(self, tensor: torch.Tensor):
        # Get the Iris device from memory_pool.device
        iris_device = self.get_device()
        tensor_device = tensor.device

        # For CUDA devices, check if they're compatible
        if tensor_device.type == "cuda" and iris_device.type == "cuda":
            if iris_device.index is None:
                return True
            return tensor_device.index == iris_device.index

        # For non-CUDA devices, they must be exactly equal
        return tensor_device == iris_device

    def __on_symmetric_heap(self, tensor: torch.Tensor):
        """Check if a tensor is allocated on the symmetric heap."""
        return self.heap.on_symmetric_heap(tensor)

    def __is_valid_device(self, device) -> bool:
        """
        Check if the requested device is compatible with this Iris instance.

        Args:
            device: The requested device (can be string, torch.device, or None)

        Returns:
            bool: True if the device is compatible, False otherwise
        """
        if device is None:
            return True  # None means use default device

        # Convert device strings to torch.device objects for proper comparison
        requested_device = torch.device(device) if isinstance(device, str) else device
        iris_device = self.get_device()

        # Check if both are CUDA devices
        if requested_device.type == "cuda" and iris_device.type == "cuda":
            # Check if index matches or if requested is "cuda" (any index)
            if requested_device.index is None:
                return True
            else:
                return requested_device.index == iris_device.index

        # For non-CUDA devices, always return False
        return False

    class CCL:
        """
        Collective Communication Library (CCL) interface for Iris.

        Provides collective operations that can be called as methods on the Iris instance.
        Example usage:
            >>> shmem = iris.iris()
            >>> shmem.ccl.all_to_all(output_tensor, input_tensor)
        """

        def __init__(self, iris_instance):
            """
            Initialize CCL with a reference to the parent Iris instance.

            Args:
                iris_instance: The parent Iris instance
            """
            self._iris = iris_instance

        def all_to_all(self, output_tensor, input_tensor, config=None, async_op=False):
            """
            All-to-all collective operation.

            Each rank sends a tensor chunk to each other rank and receives
            a tensor chunk from each other rank. Input/output tensors should have
            shape (M, N * world_size) where each chunk of N columns corresponds to one rank.

            Args:
                output_tensor: Output tensor of shape (M, N * world_size)
                input_tensor: Input tensor of shape (M, N * world_size)
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.

            Example:
                >>> shmem = iris.iris()
                >>> shmem.ccl.all_to_all(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(block_size_m=128, block_size_n=32)
                >>> shmem.ccl.all_to_all(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> shmem.ccl.all_to_all(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_to_all import all_to_all as _all_to_all

            _all_to_all(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)

        def all_gather(self, output_tensor, input_tensor, config=None, async_op=False):
            """
            All-gather collective operation.

            Each rank sends its input tensor to all ranks, and all ranks receive
            and concatenate all input tensors along dimension 0 (rows), matching
            torch.distributed.all_gather_into_tensor behavior.

            Args:
                output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
                input_tensor: Input tensor of shape (M, N) - local rank's data to send
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.

            Example:
                >>> shmem = iris.iris()
                >>> # Input: (M, N), Output: (world_size * M, N)
                >>> shmem.ccl.all_gather(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(block_size_m=128, block_size_n=32)
                >>> shmem.ccl.all_gather(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> shmem.ccl.all_gather(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_gather import all_gather as _all_gather

            _all_gather(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)

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

        def all_reduce(self, output_tensor, input_tensor, config=None, async_op=False, workspace=None):
            """
            All-reduce collective operation.

            Each rank has a local input tensor, and all ranks compute the sum of all
            input tensors. The result is written to output_tensor on all ranks.

            Args:
                output_tensor: Output tensor of shape (M, N) - will contain sum of all inputs
                input_tensor: Input tensor of shape (M, N) - local rank's partial data
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Set config.all_reduce_variant to choose variant: "atomic", "ring", or "two_shot"
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.
                workspace: Optional workspace prepared by ``all_reduce_preamble`` to
                           reuse internal buffers across invocations.

            Example:
                >>> shmem = iris.iris()
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor)

                >>> # Custom configuration with ring variant
                >>> from iris.ccl import Config
                >>> config = Config(all_reduce_variant="ring")
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Two-shot variant with block distribution
                >>> config = Config(all_reduce_variant="two_shot", all_reduce_distribution=1)
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, config=config)

                >>> # Async operation (no barrier)
                >>> shmem.ccl.all_reduce(output_tensor, input_tensor, async_op=True)
            """
            from iris.ccl.all_reduce import all_reduce as _all_reduce

            return _all_reduce(
                output_tensor,
                input_tensor,
                self._iris,
                config=config,
                async_op=async_op,
                workspace=workspace,
            )

        def reduce_scatter(self, output_tensor, input_tensor, config=None, async_op=False):
            """
            Reduce-scatter collective operation.

            Each rank reduces its assigned tiles from all ranks' inputs and stores
            the result only to its own output tensor. This is similar to all-reduce
            but without broadcasting the result to all ranks.

            Args:
                output_tensor: Output tensor of shape (M, N) - will contain reduced tiles for this rank
                input_tensor: Input tensor of shape (M, N) - local rank's partial data
                config: Config instance with kernel parameters (default: None).
                        If None, uses default Config values.
                        Only supports reduce_scatter_variant="two_shot".
                async_op: If False, performs a barrier at the end. If True, returns immediately.
                          Default: False.

            Example:
                >>> shmem = iris.iris()
                >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor)

                >>> # Custom configuration
                >>> from iris.ccl import Config
                >>> config = Config(reduce_scatter_variant="two_shot", all_reduce_distribution=1)
                >>> shmem.ccl.reduce_scatter(output_tensor, input_tensor, config=config)
            """
            from iris.ccl.reduce_scatter import reduce_scatter as _reduce_scatter

            _reduce_scatter(output_tensor, input_tensor, self._iris, config=config, async_op=async_op)


@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases):
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    # convert to int to compute difference
    ptr_int = tl.cast(ptr, tl.uint64)
    # Find the offset from from_rank heap
    offset = ptr_int - from_base
    # Byte cast for byte offset addition
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    # Find the offset into the to_rank heap
    translated_ptr_byte = to_base_byte + offset
    # Cast to_base back to pointer type
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)

    # Optimization to vectorize the load/store
    # We can't do this in general because we don't know the shape of the tensor or block sizes
    # ptr = tl.max_contiguous(tl.multiple_of(ptr, (16, 16)), (16, 32))

    # 0 You can use this if your block sizes are multiples of 32.
    # Largest vectorized load instruction is dwordx4 (128-bits)
    # translated_ptr = tl.multiple_of(translated_ptr, (32, 32))
    # translated_ptr = tl.max_contiguous(translated_ptr, (1, 32))

    # ptr = tl.max_contiguous(tl.multiple_of(ptr, 512), 512)
    # translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, 512), 512)
    return translated_ptr


@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
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
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases)
    result = tl.load(translated_ptr, mask=mask)
    return result


@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    tl.store(translated_ptr, value, mask=mask)


@triton.jit
def copy(src_ptr, dst_ptr, from_rank, to_rank, cur_rank, heap_bases, mask=None):
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

    data = tl.load(translated_src, mask=mask)
    tl.store(translated_dst, data, mask=mask)


@triton.jit
def get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
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

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(remote_ptr, local_ptr, heap_bases):
        >>>     from_rank = 1
        >>>     to_rank = 0
        >>>     iris.get(remote_ptr, local_ptr, from_rank, to_rank, heap_bases)
    """
    translated_from_ptr = __translate(from_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(translated_from_ptr, mask=mask)

    tl.store(to_ptr, data, mask=mask)


@triton.jit
def put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
    """
    Copies data from the current rank's local memory to the specified rank's memory.
    This function performs a memory write operation by loading data from the current
    rank's `from_ptr`, translating the `to_ptr` from the current rank's address
    space to the `to_rank`'s address space, and storing the data to the `to_rank` memory location.
    If the `to_rank` is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the `to_rank`'s address space. Must be the current rank where the pointer is local.
        from_rank (int): The current rank ID from which to read the data.
        to_rank (int): The `to_rank` ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

    Returns:
        None

    Example:
        >>> @triton.jit
        >>> def kernel(local_ptr, remote_ptr, heap_bases):
        >>>     from_rank = 0
        >>>     to_rank = 1
        >>>     iris.put(local_ptr, remote_ptr, from_rank, to_rank, heap_bases)
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(from_ptr, mask=mask)

    tl.store(translated_to_ptr, data, mask=mask)


@triton.jit
def atomic_add(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_sub(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_xor(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_and(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_min(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_max(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_max(translated_ptr, val, mask=mask, sem=sem, scope=scope)


def iris(heap_size=1 << 30):
    """
    Create and return an Iris instance with the specified heap size.

    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.

    Returns:
        Iris: An initialized Iris instance.

    Example:
        >>> import iris
        >>> iris_ctx = iris.iris(2**30)  # 1GB heap
        >>> tensor = iris_ctx.zeros(1024, 1024)
    """
    return Iris(heap_size)
