#include <sys/types.h>

#include <cstdlib>  // For getenv
#include <iostream>

#include "hip/hip_runtime.h"

#define hip_try(error)                                               \
  if (error != hipSuccess) {                                         \
    std::cerr << "[finegrained_allocator] Hip error: "               \
              << hipGetErrorString(error) << " at line " << __LINE__ \
              << std::endl;                                          \
    exit(-1);                                                        \
  }

inline bool is_logging_enabled() {
  return std::getenv("LOG_FINEGRAINED_ALLOCATOR") != nullptr;
}

inline void log_allocation(const char* operation, void* ptr, ssize_t size,
                           int device) {
  if (is_logging_enabled()) {
    std::cout << "[finegrained_allocator] " << operation << ": ptr=" << ptr
              << ", size=" << size << " bytes, device=" << device << std::endl;
  }
}

extern "C" {
void* finegrained_hipMalloc(ssize_t size, int device,
                            hipStream_t stream [[maybe_unused]]) {
  void* ptr;
  const auto flags = hipDeviceMallocUncached;

  int current_device;
  hip_try(hipGetDevice(&current_device));
  hip_try(hipSetDevice(device));
  hip_try(hipExtMallocWithFlags(&ptr, static_cast<size_t>(size), flags));

  log_allocation("Allocation", ptr, size, device);

  hip_try(hipSetDevice(current_device));
  return ptr;
}

void finegrained_hipFree(void* ptr, ssize_t size [[maybe_unused]], int device,
                         hipStream_t stream [[maybe_unused]]) {
  int current_device;
  hip_try(hipGetDevice(&current_device));
  hip_try(hipSetDevice(device));

  log_allocation("Deallocation", ptr, size, device);

  hip_try(hipFree(ptr));
  hip_try(hipSetDevice(current_device));
}
}
