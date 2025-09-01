# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from iris._mpi_helpers import (
    init_mpi,
    mpi_allgather,
    world_barrier,
    mpi_broadcast_scalar,
)
from iris.hip import (
    set_device,
    get_cu_count,
    count_devices,
    get_ipc_handle,
    open_ipc_handle,
    get_wall_clock_rate,
)
import numpy as np
import math
import torch
import ctypes
import logging

# Import logging functionality from the separate logging module
from .logging import logger


class Iris:
    def __init__(self, heap_size=1 << 30):
        # Initialize
        comm, cur_rank, num_ranks = init_mpi()
        num_gpus = count_devices()

        gpu_id = cur_rank % num_gpus
        set_device(gpu_id)

        self.comm = comm
        self.num_ranks = num_ranks
        self.cur_rank = cur_rank
        self.gpu_id = gpu_id
        self.heap_size = heap_size
        self.heap_offset = 0
        self.alignment = 1024
        self.device = f"cuda:{gpu_id}"
        self.memory_pool = torch.empty(heap_size, device=self.device, dtype=torch.int8)

        heap_base = self.memory_pool.data_ptr()
        heap_base_ptr = ctypes.c_void_p(heap_base)

        heap_bases = np.zeros(num_ranks, dtype=np.uint64)
        heap_bases[cur_rank] = heap_base
        ipc_handles = np.zeros((num_ranks, 64), dtype=np.uint8)
        ipc_handle = get_ipc_handle(heap_base_ptr, cur_rank)

        world_barrier()

        all_ipc_handles = mpi_allgather(np.frombuffer(ipc_handle, dtype=np.uint8))
        all_heap_bases = mpi_allgather(np.array([heap_bases[cur_rank]], dtype=np.uint64))

        world_barrier()

        ipc_heap_bases = np.zeros(num_ranks, dtype=np.uintp)
        for rank in range(num_ranks):
            if rank != cur_rank:
                handle = open_ipc_handle(all_ipc_handles[rank], cur_rank)
                ipc_heap_bases[rank] = int(handle)
            else:
                ipc_heap_bases[rank] = heap_bases[rank]

        for i in range(num_ranks):
            self.debug(f"GPU {i}: Heap base {hex(int(ipc_heap_bases[i]))}")

        world_barrier()
        self.heap_bases = torch.from_numpy(ipc_heap_bases).to(device=self.device, dtype=torch.uint64)

        world_barrier()

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

    def broadcast(self, value, source_rank):
        return mpi_broadcast_scalar(value, source_rank)

    def allocate(self, num_elements, dtype):
        self.debug(f"allocate: num_elements = {num_elements}, dtype = {dtype}")

        element_size = torch.tensor([], dtype=dtype).element_size()
        size_in_bytes = num_elements * element_size
        aligned_size = math.ceil(size_in_bytes / self.alignment) * self.alignment

        if self.heap_offset + aligned_size > self.heap_size:
            raise MemoryError("Heap out of memory")

        start = self.heap_offset
        self.heap_offset += aligned_size

        sub_buffer = self.memory_pool[start : start + size_in_bytes].view(dtype)
        return sub_buffer.reshape((num_elements,))

    def parse_size(self, size):
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
        new_tensor = self.allocate(num_elements, dtype)
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
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)

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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with zeros
            out.zero_()
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Generate random data and copy to out tensor
            random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
            out.copy_(random_data)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with ones
            out.fill_(1)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Fill with the specified value
            out.fill_(fill_value)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        self.debug(f"uniform: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
                # Multi-element tuple/list - use parse_size for compatibility
                size, num_elements = self.parse_size(steps)
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
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            # Create a reshaped view of the out tensor
            tensor = out.view(size)
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
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

    def deallocate(self, pointer):
        pass

    def get_heap_bases(self):
        return self.heap_bases

    def barrier(self):
        # Wait for all GPUs to finish work
        torch.cuda.synchronize()
        # MPI barrier
        world_barrier()

    def get_device(self):
        return self.memory_pool.device

    def get_cu_count(self):
        return get_cu_count(self.gpu_id)

    def get_rank(self):
        return self.cur_rank

    def get_num_ranks(self):
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
        heap_tensor = self.allocate(num_elements, original_tensor.dtype)

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
        # Special case for empty tensors - they might not have a valid data_ptr
        if tensor.numel() == 0:
            self.debug("Empty tensor detected, skipping heap check")
            return True

        # Convert CUDA pointer to integer for comparison
        tensor_ptr = int(tensor.data_ptr())
        heap_base = int(self.heap_bases[self.cur_rank])

        result = tensor_ptr >= heap_base and tensor_ptr < heap_base + self.heap_size

        return result

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
    # We can't do this in general because we don't know the shape of the tensor
    # ptr = tl.max_contiguous(tl.multiple_of(ptr, (64, 64)), (64, 64))
    # translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, (64, 64)), (64, 64))

    # ptr = tl.max_contiguous(tl.multiple_of(ptr, 512), 512)
    # translated_ptr = tl.max_contiguous(tl.multiple_of(translated_ptr, 512), 512)
    return translated_ptr


@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
    """
    Loads a value from the specified rank's memory location.

    This function performs a memory read operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and loading
    data from the target memory location. If the from_rank and to_rank are the same,
    this function performs a local load operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the pointer will be translated. Must be the current rank where the pointer is local.
        from_rank (int): The rank ID from which to read the data.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address pointer[idx]. Defaults to None.

    Returns:
        Block: The loaded value from the target memory location.
    """
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases)
    result = tl.load(translated_ptr, mask=mask)
    return result


@triton.jit
def store(pointer, value, from_rank, to_rank, heap_bases, mask=None):
    """
    Writes data to the specified rank's memory location.

    This function performs a memory write operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and storing
    the provided data to the target memory location. If the from_rank and to_rank are the same,
    this function performs a local store operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        value (Block): The tensor of elements to be stored.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not store the data at address pointer[idx]. Defaults to None.

    Returns:
        None
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    tl.store(translated_ptr, value, mask=mask)


@triton.jit
def get(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
    """
    Copies data from the specified rank's memory to the current rank's local memory.

    This function performs a memory read operation by translating the from_ptr
    from the current rank's address space to the from_rank's address space, loading data
    from the from_rank memory location, and storing it to the local to_ptr.
    If the from_rank is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the from_rank's address space. Must be the current rank where the pointer is local.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory where the data will be stored.
        from_rank (int): The from_rank ID from which to read the data.
        to_rank (int): The current rank ID where the data will be stored.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

    Returns:
        None
    """
    translated_from_ptr = __translate(from_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(translated_from_ptr, mask=mask)

    tl.store(to_ptr, data, mask=mask)


@triton.jit
def put(from_ptr, to_ptr, from_rank, to_rank, heap_bases, mask=None):
    """
    Copies data from the current rank's local memory to the specified rank's memory.
    This function performs a memory write operation by loading data from the current
    rank's from_ptr, translating the to_ptr from the current rank's address
    space to the to_rank's address space, and storing the data to the to_rank memory location.
    If the to_rank is the same as the current rank, this function performs a local copy operation.

    Args:
        from_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's local memory from which to read data.
        to_ptr (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the current rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        from_rank (int): The current rank ID from which to read the data.
        to_rank (int): The to_rank ID to which the data will be written.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not load the data at address from_ptr[idx] and do not store to to_ptr[idx]. Defaults to None.

    Returns:
        None
    """
    translated_to_ptr = __translate(to_ptr, from_rank, to_rank, heap_bases)

    data = tl.load(from_ptr, mask=mask)

    tl.store(translated_to_ptr, data, mask=mask)


@triton.jit
def atomic_add(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic add at the specified rank's memory location.

    This function performs an atomic addition operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    adding the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic addition operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_add(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_sub(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Atomically subtracts data from the specified rank's memory location.

    This function performs an atomic subtraction operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    subtracting the provided data from the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic subtraction operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block): The tensor of elements to be subtracted atomically.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value at the memory location before the atomic subtraction.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_sub(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, val, from_rank, to_rank, heap_bases, sem=None, scope=None):
    """
    Atomically compares and exchanges the specified rank's memory location.

    This function performs an atomic compare-and-swap operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    comparing the current value with the expected value, then writing the new value if they match.
    If the from_rank and to_rank are the same, this function performs a local atomic compare-and-swap operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        cmp (Block): The expected value to be compared with the current value at the memory location.
        val (Block): The new value to be written if the compare succeeds.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value contained at the memory location before the atomic operation attempt.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_cas(translated_ptr, cmp, val, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic exchange at the specified rank's memory location.

    This function performs an atomic exchange operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    exchanging the current value with the provided new value. If the from_rank and to_rank are the same,
    this function performs a local atomic exchange operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xchg(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_xor(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic xor at the specified rank's memory location.

    This function performs an atomic xor operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    xoring the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic xor operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xor(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_and(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic and at the specified rank's memory location.

    This function performs an atomic and operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    anding the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic and operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_and(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_or(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic or at the specified rank's memory location.

    This function performs an atomic or operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    oring the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic or operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_or(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_min(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic min at the specified rank's memory location.

    This function performs an atomic min operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    performing the min on the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic min operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_min(translated_ptr, val, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_max(pointer, val, from_rank, to_rank, heap_bases, mask=None, sem=None, scope=None):
    """
    Performs an atomic max at the specified rank's memory location.

    This function performs an atomic max operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    performing the max on the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic max operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        val (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
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
    """
    return Iris(heap_size)
