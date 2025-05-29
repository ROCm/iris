import ctypes
import numpy as np
import sys

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)


def hip_try(err):
    if err != 0:
        hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
        error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"HIP error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def open_ipc_handle(ipc_handle_data, rank):
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError("ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements")

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    hip_try(
        hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def get_ipc_handle(ptr, rank):
    ipc_handle = hipIpcMemHandle_t()
    hip_try(hip_runtime.hipIpcGetMemHandle(ctypes.byref(ipc_handle), ptr))
    return ipc_handle


def count_devices():
    device_count = ctypes.c_int()
    hip_try(hip_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
    return device_count.value


def set_device(gpu_id):
    hip_try(hip_runtime.hipSetDevice(gpu_id))


def get_device():
    device_id = ctypes.c_int()
    hip_try(hip_runtime.hipGetDevice(ctypes.byref(device_id)))
    return device_id.value


def get_cu_count(device_id=None):
    if device_id is None:
        device_id = get_device()

    hipDeviceAttributeMultiprocessorCount = 63
    cu_count = ctypes.c_int()

    hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), hipDeviceAttributeMultiprocessorCount, device_id))

    return cu_count.value


# Starting ROCm 6.5
# def get_xcc_count(device_id=None):
#     if device_id is None:
#         device_id = get_device()

#     hipDeviceAttributeNumberOfXccs = ??
#     xcc_count = ctypes.c_int()

#     hip_try(hip_runtime.hipDeviceGetAttribute(
#         ctypes.byref(xcc_count),
#         hipDeviceAttributeNumberOfXccs,
#         device_id
#     ))

#     return xcc_count


def get_wall_clock_rate(device_id):
    hipDeviceAttributeWallClockRate = 10017
    wall_clock_rate = ctypes.c_int()
    status = hip_runtime.hipDeviceGetAttribute(
        ctypes.byref(wall_clock_rate), hipDeviceAttributeWallClockRate, device_id
    )
    hip_try(status)
    return wall_clock_rate.value


def malloc_fine_grained(size):
    hipDeviceMallocFinegrained = 0x1
    ptr = ctypes.c_void_p()
    hip_try(hip_runtime.hipExtMallocWithFlags(ctypes.byref(ptr), size, hipDeviceMallocFinegrained))
    return ptr


def hip_malloc(size):
    ptr = ctypes.c_void_p()
    hip_try(hip_runtime.hipMalloc(ctypes.byref(ptr), size))
    return ptr


def hip_free(ptr):
    hip_try(hip_runtime.hipFree(ptr))
