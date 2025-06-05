import triton
import triton.language as tl

from iris._mpi_helpers import (
    init_mpi,
    mpi_allgather,
    world_barrier,
    mpi_broadcast_scalar,
)
from iris._hip import (
    set_device,
    get_device,
    get_cu_count,
    count_devices,
    malloc_fine_grained,
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

    def broadcast(self, source_rank, value):
        return mpi_broadcast_scalar(source_rank, value)

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

    def arange(self, *size, dtype=torch.int, device=None):
        self.log_debug(f"arange: size = {size}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor[:] = torch.arange(num_elements, device="cuda", dtype=dtype)
        return tensor.reshape(size)

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

    def ones(self, *size, dtype=torch.int):
        self.log_debug(f"ones: size = {size}, dtype = {dtype}")
        size, num_elements = self.parse_size(size)
        tensor = self.allocate(num_elements=num_elements, dtype=dtype)
        tensor.fill_(1)
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

    def wall_clock_rate(self, rank):
        return get_wall_clock_rate(rank)


@triton.jit
def translate(src_ptr, cur_rank, target_rank, heap_bases, debug=False):
    src_base = tl.load(heap_bases + cur_rank)
    dst_base = tl.load(heap_bases + target_rank)
    # convert to int to compute difference
    src_ptr_int = tl.cast(src_ptr, tl.uint64)
    # Find the offset from current rank heap
    offset = src_ptr_int - src_base
    # Byte cast for byte offset addition
    dst_base_byte = tl.cast(dst_base, tl.pointer_type(tl.int8))
    # Find the offset into the destination heap
    dst_ptr_byte = dst_base_byte + offset
    # Cast dst_base back to pointer type
    dst_ptr = tl.cast(dst_ptr_byte, src_ptr.dtype)

    pid = tl.program_id(axis=0)
    if debug and pid == 0:
        print("src_ptr", src_ptr)
        print("src_base", src_base)
        print("dst_base", dst_base)
        print("offset", offset)
        print("dst_ptr", dst_ptr)

    return dst_ptr


@triton.jit
def get(src_ptr, cur_rank, target_rank, heap_bases, mask=None):
    """
    Loads a value from the specified memory location and rank.

    Args:
        src_ptr (int): The source pointer.
        cur_rank (int): The current rank.
        target_rank (int): The target rank.
        heap_bases (int): The heap bases.
        mask (Optional[tl.tensor], optional): A boolean tensor used to guard memory accesses.

    Returns:
        Any: The loaded value.
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases)
    result = tl.load(dst_ptr, mask=mask)
    return result


@triton.jit
def put(src_ptr, data, cur_rank, target_rank, heap_bases, mask=None):
    """
    Writes data to the specified memory location and rank.

    Args:
        src_ptr (int): The source pointer.
        data (Any): The value to be written.
        cur_rank (int): The current rank.
        target_rank (int): The target rank.
        heap_bases (int): The heap bases.
        mask (Optional[tl.tensor], optional): A boolean tensor used to guard memory accesses. Defaults to None.

    Returns:
        None
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases, False)

    tl.store(dst_ptr, data, mask=mask)


@triton.jit
def atomic_add(src_ptr, data, cur_rank, target_rank, heap_bases, mask=None, sem=None, scope=None):
    """
        Atomically adds data to the specified memory location and rank.

        Args:
            src_ptr (int): The source pointer.
            data (Any): The value to be added.
            cur_rank (int): The current rank.
            target_rank (int): The target rank.
            heap_bases (int): The heap bases.
            mask (Optional[tl.tensor], optional): A boolean tensor used to guard memory accesses. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are “acquire”, “release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and
    “relaxed”. Defaults to “acq_rel”.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are “gpu” (default), “cta”
    (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). Defaults to “gpu”.

        Returns:
            Any: The result before the atomic addition.
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases, False)
    return tl.atomic_add(dst_ptr, data, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_sub(src_ptr, data, cur_rank, target_rank, heap_bases, mask=None, sem=None, scope=None):
    """
        Atomically subtracts data from the specified memory location and rank.

        Args:
            src_ptr (int): The source pointer.
            data (Any): The value to be subtracted.
            cur_rank (int): The current rank.
            target_rank (int): The target rank.
            heap_bases (int): The heap bases.
            mask (Optional[tl.tensor], optional): A boolean tensor used to guard memory accesses. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are “acquire”, “release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and
    “relaxed”. Defaults to “acq_rel”.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are “gpu” (default), “cta”
    (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). Defaults to “gpu”.

        Returns:
            Any: The value before the atomic subtraction.
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases, False)
    return tl.atomic_sub(dst_ptr, data, mask=mask, sem=sem, scope=scope)


@triton.jit
def atomic_cas(src_ptr, compare, value, cur_rank, target_rank, heap_bases, sem=None, scope=None):
    """
        Atomically compares and exchanges the memory location at src_ptr.

        Args:
            src_ptr (int): The source pointer.
            compare (Any): The expected value to be compared with the current value.
            value (Any): The new value to be written if the compare succeeds.
            cur_rank (int): The current rank.
            target_rank (int): The target rank.
            heap_bases (int): The heap bases.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are “acquire”, “release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and
    “relaxed”. Defaults to “acq_rel”.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are “gpu” (default), “cta”
    (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). Defaults to “gpu”.

        Returns:
            Any: The value contained at the pointer before the atomic operation attempt.
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases, False)
    return tl.atomic_cas(dst_ptr, compare, value, sem=sem, scope=scope)


@triton.jit
def atomic_xchg(src_ptr, value, cur_rank, target_rank, heap_bases, mask=None, sem=None, scope=None):
    """
        Atomically exchanges the memory location at src_ptr with the given value.

        Args:
            src_ptr (int): The source pointer.
            value (Any): The new value to be written.
            cur_rank (int): The current rank.
            target_rank (int): The target rank.
            heap_bases (int): The heap bases.
            mask (Optional[tl.tensor], optional): A boolean tensor used to guard memory accesses. Defaults to None.
            sem (str, optional): Specifies the memory semantics for the operation. Acceptable values are “acquire”, “release”, “acq_rel” (stands for “ACQUIRE_RELEASE”), and
    “relaxed”. Defaults to “acq_rel”.
            scope (str, optional): Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are “gpu” (default), “cta”
    (cooperative thread array, thread block), or “sys” (stands for “SYSTEM”). Defaults to “gpu”.

        Returns:
            Any: The old value at src_ptr before the exchange.
    """
    dst_ptr = translate(src_ptr, cur_rank, target_rank, heap_bases, False)
    return tl.atomic_xchg(dst_ptr, value, mask=mask, sem=sem, scope=scope)
