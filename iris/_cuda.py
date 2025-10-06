# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import numpy as np
import sys
import torch
import subprocess
import os

rt_path = "libcudart.so"
cuda_runtime = ctypes.cdll.LoadLibrary(rt_path)


def hip_try(err):
    if err != 0:
        cuda_runtime.cudaGetErrorString.restype = ctypes.c_char_p
        error_string = cuda_runtime.cudaGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"CUDA error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def get_ipc_handle_size():
    """Return the size of IPC handle in bytes (128 for CUDA)."""
    return 128


def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    cudaIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    cuda_runtime.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 128:
            raise ValueError("ipc_handle_data must be a 128-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 128).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 128 elements")

    raw_memory = ctypes.create_string_buffer(128)
    ctypes.memset(raw_memory, 0x00, 128)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 128)

    hip_try(
        cuda_runtime.cudaIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            cudaIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def get_ipc_handle(ptr, rank):
    ipc_handle = hipIpcMemHandle_t()
    hip_try(cuda_runtime.cudaIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    device_count = ctypes.c_int()
    hip_try(cuda_runtime.cudaGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    hip_try(cuda_runtime.cudaSetDevice(gpu_id))


def get_device_id():
    device_id = ctypes.c_int()
    hip_try(cuda_runtime.cudaGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    if device_id is None:
        device_id = get_device_id()

    cudaDeviceAttributeMultiprocessorCount = 16
    cu_count = ctypes.c_int()

    hip_try(
        cuda_runtime.cudaDeviceGetAttribute(ctypes.byref(cu_count), cudaDeviceAttributeMultiprocessorCount, device_id)
    )

    return cu_count.value


def get_wall_clock_rate(device_id):
    cudaDevAttrMemoryClockRate = 36
    wall_clock_rate = ctypes.c_int()
    status = cuda_runtime.cudaDeviceGetAttribute(ctypes.byref(wall_clock_rate), cudaDevAttrMemoryClockRate, device_id)
    hip_try(status)
    return wall_clock_rate.value


def get_arch_string(device_id=None):
    if device_id is None:
        device_id = get_device_id()
    # For CUDA, get compute capability
    device_props = torch.cuda.get_device_properties(device_id)
    major = device_props.major
    minor = device_props.minor
    return f"sm_{major}{minor}"


def get_num_xcc(device_id=None):
    # XCC is AMD-specific, return 1 for CUDA
    return 1


def malloc_fine_grained(size):
    return hip_malloc(size)


def hip_malloc(size):
    ptr = ctypes.c_void_p()
    hip_try(cuda_runtime.cudaMalloc(ctypes.byref(ptr), size))
    return ptr


def hip_free(ptr):
    hip_try(cuda_runtime.cudaFree(ptr))
