#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

bool is_logging_enabled() {
  const char* log_env = std::getenv("LOG_HIP");
  return log_env != nullptr;
}

void printHipIpcMemHandle(const hipIpcMemHandle_t& handle,
                          const std::string& message) {
  const unsigned char* data = reinterpret_cast<const unsigned char*>(&handle);
  std::cout << "[HIP SHIM] " << message
            << " hipIpcMemHandle_t contents:" << std::endl;
  for (size_t i = 0; i < sizeof(hipIpcMemHandle_t); ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
    if ((i + 1) % 16 == 0)
      std::cout << std::endl;
  }
  std::cout << std::dec;
}

void* load_hip_function(const char* func_name) {
  static void* hip_lib = dlopen("libamdhip64.so", RTLD_LAZY);
  if (!hip_lib) {
    throw std::runtime_error("Failed to load HIP library");
  }

  void* func = dlsym(hip_lib, func_name);
  if (!func) {
    throw std::runtime_error(std::string("Failed to load function: ") +
                             func_name);
  }

  return func;
}

extern "C" hipError_t hipMalloc(void** devPtr, size_t size) {
  static auto real_hipMalloc = reinterpret_cast<hipError_t (*)(void**, size_t)>(
      load_hip_function("hipMalloc"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipMalloc called with size: " << size << " bytes"
              << std::endl;
  }

  hipError_t result = real_hipMalloc(devPtr, size);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipMalloc result: " << result
              << ", ptr: " << *devPtr << std::endl;
  }

  return result;
}

extern "C" hipError_t hipFree(void* devPtr) {
  static auto real_hipFree =
      reinterpret_cast<hipError_t (*)(void*)>(load_hip_function("hipFree"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipFree called for pointer: " << devPtr
              << std::endl;
  }

  hipError_t result = real_hipFree(devPtr);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipFree result: " << result << std::endl;
  }

  return result;
}

extern "C" hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle,
                                         void* devPtr) {
  static auto real_hipIpcGetMemHandle =
      reinterpret_cast<hipError_t (*)(hipIpcMemHandle_t*, void*)>(
          load_hip_function("hipIpcGetMemHandle"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipIpcGetMemHandle called for pointer: " << devPtr
              << std::endl;
  }

  hipError_t result = real_hipIpcGetMemHandle(handle, devPtr);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipIpcGetMemHandle result: " << result
              << std::endl;
    printHipIpcMemHandle(*handle, "hipIpcGetMemHandle");
  }

  return result;
}

extern "C" hipError_t hipIpcOpenMemHandle(void** devPtr,
                                          hipIpcMemHandle_t handle,
                                          unsigned int flags) {
  static auto real_hipIpcOpenMemHandle =
      reinterpret_cast<hipError_t (*)(void**, hipIpcMemHandle_t, unsigned int)>(
          load_hip_function("hipIpcOpenMemHandle"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipIpcOpenMemHandle called with flags: " << flags
              << std::endl;
    printHipIpcMemHandle(handle, "hipIpcOpenMemHandle");
  }

  hipError_t result = real_hipIpcOpenMemHandle(devPtr, handle, flags);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipIpcOpenMemHandle result: " << result
              << ", ptr: " << *devPtr << std::endl;
  }

  return result;
}
extern "C" hipError_t hipGetDeviceCount(int* count) {
  static auto real_hipGetDeviceCount = reinterpret_cast<hipError_t (*)(int*)>(
      load_hip_function("hipGetDeviceCount"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipGetDeviceCount called" << std::endl;
  }

  hipError_t result = real_hipGetDeviceCount(count);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipGetDeviceCount result: " << result
              << ", count: " << *count << std::endl;
  }

  return result;
}

extern "C" hipError_t hipSetDevice(int deviceId) {
  static auto real_hipSetDevice =
      reinterpret_cast<hipError_t (*)(int)>(load_hip_function("hipSetDevice"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipSetDevice called with deviceId: " << deviceId
              << std::endl;
  }

  hipError_t result = real_hipSetDevice(deviceId);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipSetDevice result: " << result << std::endl;
  }

  return result;
}

extern "C" hipError_t hipGetDevice(int* deviceId) {
  static auto real_hipGetDevice =
      reinterpret_cast<hipError_t (*)(int*)>(load_hip_function("hipGetDevice"));

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipGetDevice called" << std::endl;
  }

  hipError_t result = real_hipGetDevice(deviceId);

  if (is_logging_enabled()) {
    std::cout << "[HIP SHIM] hipGetDevice result: " << result
              << ", deviceId: " << *deviceId << std::endl;
  }

  return result;
}