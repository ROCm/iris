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

STATS = True
LOGGING = True
DEBUG = False


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
            self.log_debug(f"GPU {i}: Heap base {hex(int(ipc_heap_bases[i]))}")

        world_barrier()
        self.heap_bases = torch.from_numpy(ipc_heap_bases).to(device=self.device, dtype=torch.uint64)

        world_barrier()

    def log(self, message):
        if LOGGING:
            print(f"[Iris] [{self.cur_rank}/{self.num_ranks}] {message}")

    def log_debug(self, message):
        if DEBUG:
            print(f"[Iris] [{self.cur_rank}/{self.num_ranks}] {message}")

    def log_stats(self, message):
        if STATS:
            print(f"[Iris] [{self.cur_rank}/{self.num_ranks}] {message}")

    def broadcast(self, value, source_rank):
        return mpi_broadcast_scalar(value, source_rank)

    def allocate(self, num_elements, dtype):
        self.log_debug(f"allocate: num_elements = {num_elements}, dtype = {dtype}")

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
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        num_elements = math.prod(size)
        return size, num_elements

    def zeros_like(self, tensor):
        dtype = tensor.dtype
        num_elements = tensor.numel()
        new_tensor = self.allocate(num_elements, dtype)
        new_tensor.zero_()
        return new_tensor

    def arange(
        self, start=0, end=None, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False
    ):
        """
        Returns a 1-D tensor of size ⌈(end - start) / step⌉ with values from the interval [start, end)
        taken with common difference step beginning from start.

        Args:
            start (Number, optional): the starting value for the set of points. Default: 0.
            end (Number): the ending value for the set of points
            step (Number, optional): the gap between each pair of adjacent points. Default: 1.
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor.
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
        """
        self.log_debug(f"arange: start = {start}, end = {end}, step = {step}, dtype = {dtype}")

        # Handle the case where only one argument is provided (end)
        if end is None:
            end = start
            start = 0
        # Calculate the number of elements
        num_elements = math.ceil((end - start) / step)
        # Infer dtype if not provided
        if dtype is None:
            if any(isinstance(x, float) for x in [start, end, step]):
                dtype = torch.get_default_dtype()
            else:
                dtype = torch.int64
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor[:] = torch.arange(start, end, step, dtype=dtype, device="cuda")
        if requires_grad:
            tensor.requires_grad_()
        return tensor

    def zeros(self, *size, dtype=torch.int, device=None, requires_grad=False, **kwargs):
        self.log_debug(f"zeros: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor.zero_()
        if requires_grad:
            tensor.requires_grad_()
        return tensor.reshape(size)

    def randn(
        self,
        *size,
        generator=None,
        dtype=torch.float,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        self.log_debug(
            f"randn: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}, pin_memory = {pin_memory}"
        )
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        random_data = torch.randn(num_elements, generator=generator, dtype=dtype, device=device, layout=layout)
        tensor.copy_(random_data)
        if requires_grad:
            tensor.requires_grad_()
        return tensor.reshape(size)

    def ones(self, *size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        """
        Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.

        Args:
            *size (int...): a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

        Keyword Arguments:
            out (Tensor, optional): the output tensor.
            dtype (torch.dtype, optional): the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()).
            layout (torch.layout, optional): the desired layout of returned Tensor. Default: torch.strided.
            device (torch.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type.
            requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
        """
        self.log_debug(f"ones: size = {size}, dtype = {dtype}, device = {device}, requires_grad = {requires_grad}")

        # Handle the case where size is provided as a single tuple/list
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]

        # Use global default dtype if None is provided
        if dtype is None:
            dtype = torch.get_default_dtype()
        size, num_elements = self.parse_size(size)

        # If out is provided, use it; otherwise allocate new tensor
        if out is not None:
            self.__throw_if_invalid_output_tensor(out, num_elements, dtype)
            tensor = out
        else:
            tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor.fill_(1)
        if requires_grad:
            tensor.requires_grad_()
        return tensor.reshape(size)

    def full(self, size, fill_value, dtype=torch.int):
        self.log_debug(f"full: size = {size}, fill_value = {fill_value}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor.fill_(fill_value)
        return tensor.reshape(size)

    def uniform(self, size, low=0.0, high=1.0, dtype=torch.float):
        self.log_debug(f"uniform: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor.uniform_(low, high)
        return tensor.reshape(size)

    def empty(self, size, dtype=torch.float):
        self.log_debug(f"empty: size = {size}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        return tensor.reshape(size)

    def randint(self, size, low, high, dtype=torch.int):
        self.log_debug(f"randint: size = {size}, low = {low}, high = {high}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor[:] = torch.randint(low, high, size, device="cuda", dtype=dtype)
        return tensor.reshape(size)

    def linspace(self, start, end, steps, dtype=torch.float):
        self.log_debug(f"linspace: start = {start}, end = {end}, steps = {steps}, dtype = {dtype}")
        size, num_elements = self.parse_size(steps)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        torch.linspace(start, end, size, out=tensor, dtype=dtype, device="cuda")
        return tensor.reshape(size)

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

    def __tensor_on_device(self, tensor: torch.Tensor):
        return tensor.device == self.device

    def __on_symmetric_heap(self, tensor: torch.Tensor):
        return (
            tensor.data_ptr() >= self.heap_bases[self.cur_rank]
            and tensor.data_ptr() < self.heap_bases[self.cur_rank] + self.heap_size
        )


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
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
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
    translated_from_ptr = __translate(from_ptr, to_rank, from_rank, heap_bases)

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
def atomic_add(pointer, value, from_rank, to_rank, heap_bases, mask=None, semantics=None, scope=None):
    """
    Performs an atomic add at the specified rank's memory location.

    This function performs an atomic addition operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    adding the provided data to the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic addition operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        value (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        semantics (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_add(translated_ptr, value, mask=mask, sem=semantics, scope=scope)


@triton.jit
def atomic_sub(pointer, value, from_rank, to_rank, heap_bases, mask=None, semantics=None, scope=None):
    """
    Atomically subtracts data from the specified rank's memory location.

    This function performs an atomic subtraction operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    subtracting the provided data from the to_rank memory location. If the from_rank and to_rank are the same,
    this function performs a local atomic subtraction operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        value (Block): The tensor of elements to be subtracted atomically.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        semantics (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value at the memory location before the atomic subtraction.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_sub(translated_ptr, value, mask=mask, sem=semantics, scope=scope)


@triton.jit
def atomic_cas(pointer, cmp, value, from_rank, to_rank, heap_bases, semantics=None, scope=None):
    """
    Atomically compares and exchanges the specified rank's memory location.

    This function performs an atomic compare-and-swap operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    comparing the current value with the expected value, then writing the new value if they match.
    If the from_rank and to_rank are the same, this function performs a local atomic compare-and-swap operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): Pointer in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        cmp (Block): The expected value to be compared with the current value at the memory location.
        value (Block): The new value to be written if the compare succeeds.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        semantics (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". Defaults to "acq_rel".
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). Defaults to "gpu".

    Returns:
        Block: The value contained at the memory location before the atomic operation attempt.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_cas(translated_ptr, cmp, value, sem=semantics, scope=scope)


@triton.jit
def atomic_xchg(pointer, value, from_rank, to_rank, heap_bases, mask=None, semantics=None, scope=None):
    """
    Performs an atomic exchange at the specified rank's memory location.

    This function performs an atomic exchange operation by translating the pointer
    from the from_rank's address space to the to_rank's address space and atomically
    exchanging the current value with the provided new value. If the from_rank and to_rank are the same,
    this function performs a local atomic exchange operation.

    Args:
        pointer (triton.PointerType, or block of dtype=triton.PointerType): The memory locations in the from_rank's address space that will be translated to the to_rank's address space. Must be the current rank where the pointer is local.
        value (Block of dtype=pointer.dtype.element_ty): The values with which to perform the atomic operation.
        from_rank (int): The rank ID from which the pointer originates. Must be the current rank where the pointer is local.
        to_rank (int): The rank ID to which the atomic operation will be performed.
        heap_bases (triton.PointerType): Array containing the heap base addresses for all ranks.
        mask (Block of triton.int1, optional): If mask[idx] is false, do not perform the atomic operation at address pointer[idx]. Defaults to None.
        semantics (str, optional): Specifies the memory semantics for the operation. Acceptable values are "acquire", "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to using "acq_rel" semantics.
        scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".

    Returns:
        Block: The data stored at pointer before the atomic operation.
    """
    translated_ptr = __translate(pointer, from_rank, to_rank, heap_bases)
    return tl.atomic_xchg(translated_ptr, value, mask=mask, sem=semantics, scope=scope)


def iris(heap_size=1 << 30):
    """
    Create and return an Iris instance with the specified heap size.

    Args:
        heap_size (int): Size of the heap in bytes. Defaults to 1GB.

    Returns:
        Iris: An initialized Iris instance.
    """
    return Iris(heap_size)
