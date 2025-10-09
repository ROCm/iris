# SPDX-License-Identifier: MIT

import ctypes
import numpy as np
import torch

## Constants
_CUDA_DEV_ATTR_MULTI_PROCESSOR_COUNT = 16
_CUDA_DEV_ATTR_CLOCK_RATE = 13  # kHz
_CUDA_IPC_HANDLE_SIZE = 64  # bytes

rt_path = "libcudart.so"
try:
    cuda_runtime = ctypes.cdll.LoadLibrary(rt_path)
    cuda_runtime.cudaGetErrorString.restype = ctypes.c_char_p  # Readable errors
except OSError as e:
    raise RuntimeError(f"Could not load CUDA runtime '{rt_path}'. Is CUDA installed?\n{e}")


def cuda_try(err):
    if err != 0:
        msg = cuda_runtime.cudaGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"CUDA error code {err}: {msg}")


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _CUDA_IPC_HANDLE_SIZE)]


def open_ipc_handle(ipc_handle_data, rank):
    """Open a CUDA IPC memory handle from a (64,) uint8 ndarray → returns device pointer (int)."""
    ptr = ctypes.c_void_p()
    cudaIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)

    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != _CUDA_IPC_HANDLE_SIZE:
            raise ValueError(f"ipc_handle_data must be a {_CUDA_IPC_HANDLE_SIZE}-element uint8 numpy array")
        if not ipc_handle_data.flags.c_contiguous:
            ipc_handle_data = np.ascontiguousarray(ipc_handle_data)
        ipc_handle_struct = cudaIpcMemHandle_t.from_buffer_copy(ipc_handle_data)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    cuda_try(cuda_runtime.cudaIpcOpenMemHandle(ctypes.byref(ptr), ipc_handle_struct, cudaIpcMemLazyEnablePeerAccess))
    return ptr.value


def get_ipc_handle(ptr, rank):
    """Get a CUDA IPC handle for a device pointer. Returns cudaIpcMemHandle_t."""
    if isinstance(ptr, int):
        ptr = ctypes.c_void_p(ptr)
    handle = cudaIpcMemHandle_t()
    cuda_try(cuda_runtime.cudaIpcGetMemHandle(ctypes.byref(handle), ptr))
    return handle


def count_devices():
    n = ctypes.c_int()
    cuda_try(cuda_runtime.cudaGetDeviceCount(ctypes.byref(n)))
    return n.value


def set_device(gpu_id):
    cuda_try(cuda_runtime.cudaSetDevice(gpu_id))


def get_device_id():
    dev = ctypes.c_int()
    cuda_try(cuda_runtime.cudaGetDevice(ctypes.byref(dev)))
    return dev.value


def get_cu_count(device_id=None):
    """Number of SMs (CUDA equivalent of HIP CU count)."""
    if device_id is None:
        device_id = get_device_id()
    val = ctypes.c_int()
    cuda_try(cuda_runtime.cudaDeviceGetAttribute(ctypes.byref(val), _CUDA_DEV_ATTR_MULTI_PROCESSOR_COUNT, device_id))
    return val.value


def get_runtime_version():
    """Return (major, minor) for CUDA runtime."""
    ver = ctypes.c_int()
    cuda_try(cuda_runtime.cudaRuntimeGetVersion(ctypes.byref(ver)))
    v = ver.value
    return (v // 1000, (v % 1000) // 10)


# Backwards compatibility alias
get_cuda_version = get_runtime_version


def get_wall_clock_rate(device_id):
    """Device core clock rate in kHz (cudaDevAttrClockRate)."""
    val = ctypes.c_int()
    cuda_try(cuda_runtime.cudaDeviceGetAttribute(ctypes.byref(val), _CUDA_DEV_ATTR_CLOCK_RATE, device_id))
    return val.value


def get_arch_string(device_id=None):
    """Return 'sm_{major}{minor}', e.g., 'sm_90'."""
    if device_id is None:
        device_id = get_device_id()
    p = torch.cuda.get_device_properties(device_id)
    return f"sm_{p.major}{p.minor}"


def get_num_xcc(device_id=None):
    """
    No XCC on NVIDIA. Return 1 so scheduling math like:
      pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    is an identity when NUM_XCDS == 1.
    """
    return 1


def malloc_fine_grained(size):
    """Use managed (Unified) memory as closest analogue to HIP fine-grained."""
    ptr = ctypes.c_void_p()
    cudaMemAttachGlobal = 0x1
    cuda_try(cuda_runtime.cudaMallocManaged(ctypes.byref(ptr), size, cudaMemAttachGlobal))
    return ptr


def malloc(size):
    ptr = ctypes.c_void_p()
    cuda_try(cuda_runtime.cudaMalloc(ctypes.byref(ptr), size))
    return ptr


def free(ptr):
    if isinstance(ptr, int):
        ptr = ctypes.c_void_p(ptr)
    cuda_try(cuda_runtime.cudaFree(ptr))


## Backend-agnostic aliases (for compatibility with both CUDA and HIP)
hip_try = cuda_try
hip_malloc = malloc
hip_free = free
get_rocm_version = get_runtime_version
