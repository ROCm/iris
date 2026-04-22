# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import numpy as np
import torch
import subprocess
import os

# Auto-detect backend
_is_amd_backend = True
try:
    rt_path = "libamdhip64.so"
    gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)
except OSError:
    try:
        rt_path = "libcudart.so"
        gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)
        _is_amd_backend = False
    except OSError:
        rt_path = "libamdhip64.so"
        gpu_runtime = ctypes.cdll.LoadLibrary(rt_path)


def gpu_try(err):
    if err != 0:
        if _is_amd_backend:
            gpu_runtime.hipGetErrorString.restype = ctypes.c_char_p
            error_string = gpu_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
            raise RuntimeError(f"HIP error code {err}: {error_string}")
        else:
            gpu_runtime.cudaGetErrorString.restype = ctypes.c_char_p
            error_string = gpu_runtime.cudaGetErrorString(ctypes.c_int(err)).decode("utf-8")
            raise RuntimeError(f"CUDA error code {err}: {error_string}")


def get_ipc_handle_size():
    """Return the IPC handle size for the current backend."""
    return 64 if _is_amd_backend else 128


class gpuIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * get_ipc_handle_size())]


def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    handle_size = get_ipc_handle_size()

    if _is_amd_backend:
        hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
        gpu_runtime.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            gpuIpcMemHandle_t,
            ctypes.c_uint,
        ]
    else:
        gpu_runtime.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            gpuIpcMemHandle_t,
            ctypes.c_uint,
        ]
        cudaIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)

    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != handle_size:
            raise ValueError(f"ipc_handle_data must be a {handle_size}-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * handle_size).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError(f"ipc_handle_data must be a numpy.ndarray of dtype uint8 with {handle_size} elements")

    raw_memory = ctypes.create_string_buffer(handle_size)
    ctypes.memset(raw_memory, 0x00, handle_size)
    ipc_handle_struct = gpuIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, handle_size)

    if _is_amd_backend:
        gpu_try(
            gpu_runtime.hipIpcOpenMemHandle(
                ctypes.byref(ptr),
                ipc_handle_struct,
                hipIpcMemLazyEnablePeerAccess,
            )
        )
    else:
        gpu_try(
            gpu_runtime.cudaIpcOpenMemHandle(
                ctypes.byref(ptr),
                ipc_handle_struct,
                cudaIpcMemLazyEnablePeerAccess,
            )
        )

    return ptr.value


def get_ipc_handle(ptr, rank):
    ipc_handle = gpuIpcMemHandle_t()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    else:
        gpu_try(gpu_runtime.cudaIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    device_count = ctypes.c_int()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
    else:
        gpu_try(gpu_runtime.cudaGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipSetDevice(gpu_id))
    else:
        gpu_try(gpu_runtime.cudaSetDevice(gpu_id))


def get_device_id():
    device_id = ctypes.c_int()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipGetDevice(ctypes.byref(device_id)))
    else:
        gpu_try(gpu_runtime.cudaGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    cu_count = ctypes.c_int()

    if _is_amd_backend:
        hipDeviceAttributeMultiprocessorCount = 63
        gpu_try(
            gpu_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), hipDeviceAttributeMultiprocessorCount, device_id)
        )
    else:
        cudaDevAttrMultiProcessorCount = 16
        gpu_try(gpu_runtime.cudaDeviceGetAttribute(ctypes.byref(cu_count), cudaDevAttrMultiProcessorCount, device_id))

    return cu_count.value


def get_rocm_version():
    if not _is_amd_backend:
        # Not applicable for CUDA
        return (-1, -1)

    major, minor = -1, -1

    # Try hipconfig --path first
    try:
        result = subprocess.run(["hipconfig", "--path"], capture_output=True, text=True, check=True)
        rocm_path = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Then look for $ROCM_PATH environment variable
        rocm_path = os.environ.get("ROCM_PATH")
        if not rocm_path:
            # Finally, try default location
            rocm_path = "/opt/rocm"

    # Try to read version from .info/version file
    try:
        version_file_path = os.path.join(rocm_path, ".info", "version")
        with open(version_file_path, "r") as version_file:
            version = version_file.readline().strip()
            major = int(version.split(".")[0])
            minor = int(version.split(".")[1])
    except (FileNotFoundError, IOError, ValueError, IndexError):
        # If we can't read the version file, return -1, -1
        pass

    return (major, minor)


def get_wall_clock_rate(device_id):
    wall_clock_rate = ctypes.c_int()

    if _is_amd_backend:
        hipDeviceAttributeWallClockRate = 10017
        status = gpu_runtime.hipDeviceGetAttribute(
            ctypes.byref(wall_clock_rate), hipDeviceAttributeWallClockRate, device_id
        )
    else:
        cudaDevAttrClockRate = 13
        status = gpu_runtime.cudaDeviceGetAttribute(ctypes.byref(wall_clock_rate), cudaDevAttrClockRate, device_id)

    gpu_try(status)
    return wall_clock_rate.value


def get_arch_string(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    if _is_amd_backend:
        arch_full = torch.cuda.get_device_properties(device_id).gcnArchName
        arch_name = arch_full.split(":")[0]
        return arch_name
    else:
        # For CUDA, return compute capability
        props = torch.cuda.get_device_properties(device_id)
        return f"sm_{props.major}{props.minor}"


def get_num_xcc(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    if not _is_amd_backend:
        # XCC is AMD-specific, return 1 for CUDA
        return 1

    rocm_major, _ = get_rocm_version()
    if rocm_major < 7:
        return 8
    hipDeviceAttributeNumberOfXccs = 10018
    xcc_count = ctypes.c_int()
    gpu_try(gpu_runtime.hipDeviceGetAttribute(ctypes.byref(xcc_count), hipDeviceAttributeNumberOfXccs, device_id))
    return xcc_count.value


def malloc_fine_grained(size):
    ptr = ctypes.c_void_p()

    if _is_amd_backend:
        hipDeviceMallocFinegrained = 0x1
        gpu_try(gpu_runtime.hipExtMallocWithFlags(ctypes.byref(ptr), size, hipDeviceMallocFinegrained))
    else:
        # CUDA doesn't have direct equivalent, use regular malloc
        gpu_try(gpu_runtime.cudaMalloc(ctypes.byref(ptr), size))

    return ptr


def hip_malloc(size):
    ptr = ctypes.c_void_p()
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipMalloc(ctypes.byref(ptr), size))
    else:
        gpu_try(gpu_runtime.cudaMalloc(ctypes.byref(ptr), size))
    return ptr


def hip_free(ptr):
    if _is_amd_backend:
        gpu_try(gpu_runtime.hipFree(ptr))
    else:
        gpu_try(gpu_runtime.cudaFree(ptr))


def export_dmabuf_handle(ptr, size):
    """
    Export a DMA-BUF file descriptor for a memory range.

    Args:
        ptr: Integer or ctypes pointer to GPU memory
        size: Size of the memory range in bytes

    Returns:
        tuple: (fd, base_ptr, base_size) where:
            - fd: File descriptor (integer) for the DMA-BUF handle
            - base_ptr: Base address of the allocation containing ptr
            - base_size: Size of the base allocation

        The base_ptr and base_size are needed because hipMemGetHandleForAddressRange
        exports the entire allocation buffer, not just the requested range. When
        importing, you'll need these to calculate the correct offset.

    Raises:
        RuntimeError: If export fails or backend doesn't support it
    """
    if not _is_amd_backend:
        raise RuntimeError("DMA-BUF export only supported on AMD/HIP backend")

    ptr_int = ptr if isinstance(ptr, int) else ptr.value
    ptr_arg = ctypes.c_void_p(ptr_int)

    base_ptr = ctypes.c_void_p()
    base_size = ctypes.c_size_t()

    gpu_runtime.hipMemGetAddressRange.restype = ctypes.c_int
    gpu_runtime.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # pbase
        ctypes.POINTER(ctypes.c_size_t),  # psize
        ctypes.c_void_p,  # dptr
    ]

    err = gpu_runtime.hipMemGetAddressRange(ctypes.byref(base_ptr), ctypes.byref(base_size), ptr_arg)
    if err != 0:
        gpu_try(err)

    fd = ctypes.c_int(-1)

    gpu_runtime.hipMemGetHandleForAddressRange.restype = ctypes.c_int
    gpu_runtime.hipMemGetHandleForAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # handle (DMA-BUF fd)
        ctypes.c_void_p,  # devPtr
        ctypes.c_size_t,  # size
        ctypes.c_int,  # handleType
        ctypes.c_ulonglong,  # flags
    ]

    err = gpu_runtime.hipMemGetHandleForAddressRange(ctypes.byref(fd), ptr_arg, size, 1, 0)

    if err != 0:
        gpu_try(err)  # Will raise with error message

    return (fd.value, base_ptr.value, base_size.value)


def import_dmabuf_handle(fd, size, original_ptr=None, base_ptr=None):
    """
    Import a DMA-BUF file descriptor and map it to a GPU address.

    Args:
        fd: DMA-BUF file descriptor
        size: Size of the memory range to map (typically the base_size from export)
        original_ptr: Optional. The original pointer that was exported.
        base_ptr: Optional. The base address of the allocation (from export).

        If both original_ptr and base_ptr are provided, the function will calculate
        the offset (original_ptr - base_ptr) and return the correctly offset pointer
        in the mapped address space. This is needed when exporting PyTorch tensors
        from the caching allocator, as hipMemGetHandleForAddressRange exports the
        entire allocator buffer, not just the specific tensor's memory.

        If only one parameter is provided (but not both), offset correction is skipped
        and mapped_base is returned directly.

    Returns:
        tuple: (mapped_ptr, ext_mem_handle) where:
            - mapped_ptr: GPU address (integer). If original_ptr and base_ptr are provided,
              returns the offset-corrected address.
            - ext_mem_handle: External memory handle that must be destroyed with
              destroy_external_memory() when done.

    Raises:
        RuntimeError: If import fails or backend doesn't support it
    """
    if not _is_amd_backend:
        raise RuntimeError("DMA-BUF import only supported on AMD/HIP backend")

    # hipExternalMemory_t is an opaque handle (pointer)
    hipExternalMemory_t = ctypes.c_void_p

    # Create external memory handle descriptor
    class hipExternalMemoryHandleDesc(ctypes.Structure):
        class HandleUnion(ctypes.Union):
            _fields_ = [
                ("fd", ctypes.c_int),
                ("win32", ctypes.c_void_p * 2),  # handle + name (16 bytes on 64-bit)
            ]

        _fields_ = [
            ("type", ctypes.c_int),  # hipExternalMemoryHandleType
            ("_pad", ctypes.c_int),  # Padding for 8-byte alignment
            ("handle", HandleUnion),
            ("size", ctypes.c_ulonglong),
            ("flags", ctypes.c_uint),
            ("_pad2", ctypes.c_uint),  # Padding
            ("reserved", ctypes.c_uint * 16),
        ]

    # Create buffer descriptor
    class hipExternalMemoryBufferDesc(ctypes.Structure):
        _fields_ = [
            ("offset", ctypes.c_ulonglong),
            ("size", ctypes.c_ulonglong),
            ("flags", ctypes.c_uint),
            ("reserved", ctypes.c_uint * 16),
        ]

    # Setup handle descriptor (hipExternalMemoryHandleTypeOpaqueFd = 1)
    mem_handle_desc = hipExternalMemoryHandleDesc()
    mem_handle_desc.type = 1  # hipExternalMemoryHandleTypeOpaqueFd
    mem_handle_desc.handle.fd = fd
    mem_handle_desc.size = size
    mem_handle_desc.flags = 0

    # Import external memory
    ext_mem = hipExternalMemory_t()

    # Set argument types for hipImportExternalMemory
    gpu_runtime.hipImportExternalMemory.argtypes = [
        ctypes.POINTER(hipExternalMemory_t),
        ctypes.POINTER(hipExternalMemoryHandleDesc),
    ]
    gpu_runtime.hipImportExternalMemory.restype = ctypes.c_int

    err = gpu_runtime.hipImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(mem_handle_desc))
    if err != 0:
        gpu_try(err)

    # Map buffer
    buffer_desc = hipExternalMemoryBufferDesc()
    buffer_desc.offset = 0
    buffer_desc.size = size
    buffer_desc.flags = 0

    dev_ptr = ctypes.c_void_p()
    err = gpu_runtime.hipExternalMemoryGetMappedBuffer(ctypes.byref(dev_ptr), ext_mem, ctypes.byref(buffer_desc))
    if err != 0:
        gpu_try(err)

    mapped_base = dev_ptr.value

    if original_ptr is not None and base_ptr is not None:
        original_ptr_int = original_ptr if isinstance(original_ptr, int) else original_ptr.value
        base_ptr_int = base_ptr if isinstance(base_ptr, int) else base_ptr.value
        offset = original_ptr_int - base_ptr_int
        if offset < 0:
            raise ValueError(f"Invalid offset: original_ptr ({original_ptr_int}) < base_ptr ({base_ptr_int})")

        return (mapped_base + offset, ext_mem)

    return (mapped_base, ext_mem)


def destroy_external_memory(ext_mem_handle):
    """
    Destroy an external memory handle created by hipImportExternalMemory.

    Args:
        ext_mem_handle: The external memory handle (hipExternalMemory_t) to destroy

    Raises:
        RuntimeError: If destroy fails
    """
    if not _is_amd_backend:
        raise RuntimeError("External memory only supported on AMD/HIP backend")

    # hipExternalMemory_t is an opaque handle (pointer)
    hipExternalMemory_t = ctypes.c_void_p

    gpu_runtime.hipDestroyExternalMemory.argtypes = [hipExternalMemory_t]
    gpu_runtime.hipDestroyExternalMemory.restype = ctypes.c_int

    err = gpu_runtime.hipDestroyExternalMemory(ext_mem_handle)
    if err != 0:
        gpu_try(err)


def get_address_range(ptr):
    """
    Query the base allocation and size for a given device pointer.

    Args:
        ptr: Device pointer (integer or ctypes pointer)

    Returns:
        tuple: (base_ptr, size) - base address and size of the allocation

    Raises:
        RuntimeError: If query fails
    """
    ptr_int = ptr if isinstance(ptr, int) else ptr.value

    base_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    gpu_runtime.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # void** pbase
        ctypes.POINTER(ctypes.c_size_t),  # size_t* psize
        ctypes.c_void_p,  # void* dptr
    ]
    gpu_runtime.hipMemGetAddressRange.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemGetAddressRange(ctypes.byref(base_ptr), ctypes.byref(size), ctypes.c_void_p(ptr_int)))

    return base_ptr.value, size.value


# ============================================================================
# HIP Virtual Memory (VMem) Management APIs
# ============================================================================

# Constants for VMem APIs
hipMemAllocationTypePinned = 0x1
hipMemHandleTypePosixFileDescriptor = 0x1
hipMemLocationTypeDevice = 0x1
hipMemAllocationGranularityRecommended = 0x1
hipMemAccessFlagsProtReadWrite = 0x3

# Type alias for VMem handle (pointer type)
hipMemGenericAllocationHandle_t = ctypes.c_void_p


class hipMemLocation(ctypes.Structure):
    """Structure describing a memory location (device)."""

    _fields_ = [
        ("type", ctypes.c_int),  # hipMemLocationType
        ("id", ctypes.c_int),  # Device ID
    ]


class hipMemAllocationProp(ctypes.Structure):
    """Properties for memory allocation."""

    class _allocFlags(ctypes.Structure):
        _fields_ = [
            ("smc", ctypes.c_ubyte),
            ("l2", ctypes.c_ubyte),
        ]

    _fields_ = [
        ("type", ctypes.c_int),  # hipMemAllocationType
        ("requestedHandleType", ctypes.c_int),  # hipMemHandleType
        ("location", hipMemLocation),  # Memory location
        ("win32Handle", ctypes.c_void_p),  # Windows handle (unused on Linux)
        ("allocFlags", _allocFlags),  # Allocation flags
    ]


class hipMemAccessDesc(ctypes.Structure):
    """Memory access descriptor for setting access permissions."""

    _fields_ = [
        ("location", hipMemLocation),  # Device location
        ("flags", ctypes.c_int),  # Access flags
    ]


def get_allocation_granularity(device_id):
    """
    Get the allocation granularity for VMem allocations on a device.

    Args:
        device_id: Device ID

    Returns:
        Allocation granularity in bytes

    Raises:
        RuntimeError: If query fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    prop = hipMemAllocationProp()
    prop.type = hipMemAllocationTypePinned
    prop.location.type = hipMemLocationTypeDevice
    prop.location.id = device_id
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor

    granularity = ctypes.c_size_t()

    gpu_try(
        gpu_runtime.hipMemGetAllocationGranularity(
            ctypes.byref(granularity),
            ctypes.byref(prop),
            hipMemAllocationGranularityRecommended,
        )
    )

    return granularity.value


def mem_create(size, device_id):
    """
    Create a physical memory allocation.

    Args:
        size: Size in bytes (should be aligned to granularity)
        device_id: Device ID

    Returns:
        hipMemGenericAllocationHandle_t handle

    Raises:
        RuntimeError: If creation fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    prop = hipMemAllocationProp()
    prop.type = hipMemAllocationTypePinned
    prop.location.type = hipMemLocationTypeDevice
    prop.location.id = device_id
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor

    handle = hipMemGenericAllocationHandle_t()

    # Set argument types explicitly to avoid 32/64-bit issues
    gpu_runtime.hipMemCreate.argtypes = [
        ctypes.POINTER(hipMemGenericAllocationHandle_t),  # handle
        ctypes.c_size_t,  # size (64-bit!)
        ctypes.POINTER(hipMemAllocationProp),  # prop
        ctypes.c_ulonglong,  # flags
    ]
    gpu_runtime.hipMemCreate.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0))

    return handle.value


def mem_export_to_shareable_handle(handle):
    """
    Export a VMem handle as a shareable file descriptor.

    Args:
        handle: hipMemGenericAllocationHandle_t

    Returns:
        File descriptor (integer)

    Raises:
        RuntimeError: If export fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    fd = ctypes.c_int(-1)

    # Set argument types
    gpu_runtime.hipMemExportToShareableHandle.argtypes = [
        ctypes.c_void_p,  # void* shareableHandle (pointer to fd)
        hipMemGenericAllocationHandle_t,  # hipMemGenericAllocationHandle_t handle
        ctypes.c_int,  # hipMemAllocationHandleType handleType
        ctypes.c_ulonglong,  # unsigned long long flags
    ]
    gpu_runtime.hipMemExportToShareableHandle.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemExportToShareableHandle(ctypes.byref(fd), handle, hipMemHandleTypePosixFileDescriptor, 0))

    return fd.value


def mem_import_from_shareable_handle(fd):
    """
    Import a VMem handle from a shareable file descriptor.

    Args:
        fd: File descriptor (integer)

    Returns:
        hipMemGenericAllocationHandle_t handle

    Raises:
        RuntimeError: If import fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    handle = hipMemGenericAllocationHandle_t()

    # Set argument types
    gpu_runtime.hipMemImportFromShareableHandle.argtypes = [
        ctypes.POINTER(hipMemGenericAllocationHandle_t),
        ctypes.c_void_p,  # void* - cast the fd integer to void*
        ctypes.c_int,  # hipMemAllocationHandleType
    ]
    gpu_runtime.hipMemImportFromShareableHandle.restype = ctypes.c_int

    # Cast the integer fd to void* (like the C++ tests do)
    gpu_try(
        gpu_runtime.hipMemImportFromShareableHandle(
            ctypes.byref(handle), ctypes.c_void_p(fd), hipMemHandleTypePosixFileDescriptor
        )
    )

    return handle.value


def mem_address_reserve(size, alignment=0, addr=0, flags=0):
    """
    Reserve a virtual address range.

    Args:
        size: Size in bytes
        alignment: Alignment requirement (0 for default)
        addr: Requested address (0 for automatic)
        flags: Flags

    Returns:
        Reserved virtual address (integer)

    Raises:
        RuntimeError: If reservation fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    ptr = ctypes.c_void_p()

    # Set argument types explicitly
    gpu_runtime.hipMemAddressReserve.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # void** ptr
        ctypes.c_size_t,  # size_t size
        ctypes.c_size_t,  # size_t alignment
        ctypes.c_void_p,  # void* addr
        ctypes.c_ulonglong,  # unsigned long long flags
    ]
    gpu_runtime.hipMemAddressReserve.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemAddressReserve(ctypes.byref(ptr), size, alignment, ctypes.c_void_p(addr), flags))

    return ptr.value


def mem_map(ptr, size, offset, handle, flags=0):
    """
    Map physical memory to virtual address range.

    Args:
        ptr: Virtual address (integer)
        size: Size in bytes
        offset: Offset within physical allocation
        handle: hipMemGenericAllocationHandle_t
        flags: Flags

    Raises:
        RuntimeError: If mapping fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    # Set argument types
    gpu_runtime.hipMemMap.argtypes = [
        ctypes.c_void_p,  # void* ptr
        ctypes.c_size_t,  # size_t size
        ctypes.c_size_t,  # size_t offset
        hipMemGenericAllocationHandle_t,  # hipMemGenericAllocationHandle_t handle
        ctypes.c_ulonglong,  # unsigned long long flags
    ]
    gpu_runtime.hipMemMap.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemMap(ctypes.c_void_p(ptr), size, offset, handle, flags))


def mem_unmap(ptr, size):
    """
    Unmap virtual address range.

    Args:
        ptr: Virtual address (integer)
        size: Size in bytes

    Raises:
        RuntimeError: If unmapping fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    # Set argument types explicitly
    gpu_runtime.hipMemUnmap.argtypes = [
        ctypes.c_void_p,  # void* ptr
        ctypes.c_size_t,  # size_t size
    ]
    gpu_runtime.hipMemUnmap.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemUnmap(ctypes.c_void_p(ptr), size))


def mem_address_free(ptr, size):
    """
    Free a reserved virtual address range.

    Args:
        ptr: Virtual address (integer)
        size: Size in bytes

    Raises:
        RuntimeError: If freeing fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    # Set argument types explicitly
    gpu_runtime.hipMemAddressFree.argtypes = [
        ctypes.c_void_p,  # void* ptr
        ctypes.c_size_t,  # size_t size
    ]
    gpu_runtime.hipMemAddressFree.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemAddressFree(ctypes.c_void_p(ptr), size))


def mem_release(handle):
    """
    Release a physical memory allocation handle.

    Args:
        handle: hipMemGenericAllocationHandle_t

    Raises:
        RuntimeError: If release fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    # Set argument types
    gpu_runtime.hipMemRelease.argtypes = [hipMemGenericAllocationHandle_t]
    gpu_runtime.hipMemRelease.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemRelease(handle))


def mem_set_access(ptr, size, desc_or_list):
    """
    Set access permissions for a virtual address range.

    Args:
        ptr: Virtual address (integer)
        size: Size in bytes
        desc_or_list: hipMemAccessDesc or list of hipMemAccessDesc for multi-device access

    Raises:
        RuntimeError: If setting access fails or backend doesn't support VMem
    """
    if not _is_amd_backend:
        raise RuntimeError("VMem only supported on AMD/HIP backend")

    # Support both single descriptor and list of descriptors
    if isinstance(desc_or_list, list):
        desc_array = (hipMemAccessDesc * len(desc_or_list))(*desc_or_list)
        count = len(desc_or_list)
    else:
        desc_array = (hipMemAccessDesc * 1)(desc_or_list)
        count = 1

    # Set argument types
    gpu_runtime.hipMemSetAccess.argtypes = [
        ctypes.c_void_p,  # void* ptr
        ctypes.c_size_t,  # size_t size
        ctypes.POINTER(hipMemAccessDesc),  # const hipMemAccessDesc* desc
        ctypes.c_size_t,  # size_t count
    ]
    gpu_runtime.hipMemSetAccess.restype = ctypes.c_int

    gpu_try(gpu_runtime.hipMemSetAccess(ctypes.c_void_p(ptr), size, desc_array, count))
