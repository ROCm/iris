// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <mpi.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "iris/logger.hpp"

#define hip_try(error)                                                    \
  if (error != hipSuccess) {                                              \
    std::cerr << "Hip error: " << hipGetErrorString(error) << " at line " \
              << __LINE__ << std::endl;                                   \
    exit(-1);                                                             \
  }

namespace iris {

namespace mpi {

struct init_result {
  int rank;
  int world_size;
};

init_result initialize() {
  static bool initialized = false;
  static int rank, world_size;
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  }
  return {rank, world_size};
}

void finalize() { MPI_Finalize(); }

}  // namespace mpi
namespace detail {
void* malloc_fine_grained(std::size_t bytes) {
  void* ptr;
  const auto flags = hipDeviceMallocFinegrained;
  hip_try(hipExtMallocWithFlags(&ptr, bytes, flags));

  return ptr;
}

hipIpcMemHandle_t get_ipc_handle(void* ptr) {
  hipIpcMemHandle_t ipc_handle;
  hip_try(hipIpcGetMemHandle(&ipc_handle, ptr));
  return ipc_handle;
}

template <typename T>
void mpi_allgather(T* data) {
  MPI_Comm thread_comm = MPI_COMM_WORLD;
  MPI_Comm shmcomm;
  MPI_Comm_split_type(thread_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmcomm);
  int shm_size;
  MPI_Comm_size(shmcomm, &shm_size);
  int shm_rank;
  MPI_Comm_rank(shmcomm, &shm_rank);
  MPI_Allgather(MPI_IN_PLACE, sizeof(T), MPI_CHAR, data, sizeof(T), MPI_CHAR,
                shmcomm);
}

void world_barrier() { MPI_Barrier(MPI_COMM_WORLD); }
}  // namespace detail

// Memory scopes and orders
enum class memory_order {
  relaxed = __ATOMIC_RELAXED,
  consume = __ATOMIC_CONSUME,
  acquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST
};
inline constexpr memory_order memory_order_relaxed = memory_order::relaxed;
inline constexpr memory_order memory_order_consume = memory_order::consume;
inline constexpr memory_order memory_order_acquire = memory_order::acquire;
inline constexpr memory_order memory_order_release = memory_order::release;
inline constexpr memory_order memory_order_acq_rel = memory_order::acq_rel;
inline constexpr memory_order memory_order_seq_cst = memory_order::seq_cst;

enum class memory_scope {
  thread = __HIP_MEMORY_SCOPE_SINGLETHREAD,
  warp = __HIP_MEMORY_SCOPE_WAVEFRONT,
  block = __HIP_MEMORY_SCOPE_WORKGROUP,
  device = __HIP_MEMORY_SCOPE_AGENT,
  system = __HIP_MEMORY_SCOPE_SYSTEM,
};

inline constexpr memory_scope memory_scope_thread = memory_scope::thread;
inline constexpr memory_scope memory_scope_warp = memory_scope::warp;
inline constexpr memory_scope memory_scope_block = memory_scope::block;
inline constexpr memory_scope memory_scope_device = memory_scope::device;
inline constexpr memory_scope memory_scope_system = memory_scope::system;

// Iris class
class iris {
 public:
  iris(const std::size_t heap_size_bytes, int rank, int world_size,
       const bool verbose = false) {
    cur_rank_ = rank;
    world_size_ = world_size;

    // Initialize logger
    ::iris::logging::init_from_env(rank);

    IRIS_LOG_INFO("Initializing Iris: heap_size={} MB, world_size={}",
                  heap_size_bytes / (1024 * 1024), world_size);

    int num_gpus;
    hip_try(hipGetDeviceCount(&num_gpus));
    device_id_  = rank % num_gpus;
    hip_try(hipSetDevice(device_id_));

    IRIS_LOG_DEBUG("GPU selection: num_gpus={}, device_id={}", num_gpus, device_id_);

    if (verbose) {
      printf("Rank %d: num_gpus: %d, device_id: %d\n", rank, num_gpus, device_id_);
    }

    auto heap_base = detail::malloc_fine_grained(heap_size_bytes);
    bytes_capacity_ = heap_size_bytes;
    IRIS_LOG_DEBUG("Allocated fine-grained heap: size={} MB, base_addr=0x{:x}",
                   heap_size_bytes / (1024 * 1024),
                   reinterpret_cast<uintptr_t>(heap_base));

    // all gather ipc handles
    std::vector<hipIpcMemHandle_t> ipc_handles(world_size);
    std::vector<uintptr_t> heap_bases(world_size);
    ipc_handles[cur_rank_] = detail::get_ipc_handle(heap_base);
    heap_bases[cur_rank_] = reinterpret_cast<uintptr_t>(heap_base);

    detail::world_barrier();

    IRIS_LOG_DEBUG("Exchanging IPC handles and heap bases across {} ranks", world_size);
    detail::mpi_allgather(ipc_handles.data());
    detail::mpi_allgather(heap_bases.data());

    detail::world_barrier();

    if (verbose) {
      for (std::size_t i = 0; i < heap_bases.size(); i++) {
        std::cout << "Rank: " << cur_rank_ << " GPU: " << i
                  << " heap base: " << heap_bases[i] << std::endl;
      }
    }

    std::vector<uintptr_t> ipc_heap_bases(world_size_);

    IRIS_LOG_DEBUG("Opening IPC memory handles for remote ranks");
    for (size_t i = 0; i < world_size_; i++) {
      if (i != cur_rank_) {
        void** ipc_base_uncast = reinterpret_cast<void**>(&ipc_heap_bases[i]);
        hip_try(hipIpcOpenMemHandle(ipc_base_uncast, ipc_handles[i],
                                    hipIpcMemLazyEnablePeerAccess));
        IRIS_LOG_TRACE("Opened IPC handle for rank {}: addr=0x{:x}", i, ipc_heap_bases[i]);
      } else {
        ipc_heap_bases[i] = reinterpret_cast<uintptr_t>(heap_base);
      }
    }

    if (verbose) {
      for (std::size_t i = 0; i < heap_bases.size(); i++) {
        std::cout << "Rank: " << cur_rank_ << " GPU: " << i
                  << " ipc heap base: " << ipc_heap_bases[i] << std::endl;
      }
    }

    detail::world_barrier();

    std::memcpy(heap_bases_, ipc_heap_bases.data(),
                sizeof(uintptr_t) * world_size);

    IRIS_LOG_INFO("Iris initialization complete");
  }

  void barrier() {
    hip_try(hipDeviceSynchronize());
    detail::world_barrier();
  }

  ~iris() { hip_try(hipFree(reinterpret_cast<void*>(heap_bases_[cur_rank_]))); }

  [[nodiscard]] __host__ __device__ int cur_rank() const { return cur_rank_; }
  [[nodiscard]] __host__ __device__ int world_size() const {
    return world_size_;
  }

  template <typename T>
  T* allocate(const std::size_t num_elements) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    static constexpr std::size_t alignment = 1024;
    const auto num_bytes = num_elements * sizeof(T);
    auto bytes_to_allocate = (num_bytes + alignment - 1) & ~(alignment - 1);

    std::uintptr_t ptr{};

    if (bytes_capacity_ < bytes_to_allocate) {
      // Try to allocate from the free list
      const auto it = free_list_.lower_bound(bytes_to_allocate);
      if (it != free_list_.end()) {
        ptr = it->second;
        bytes_to_allocate = it->first;
        free_list_.erase(it);
        IRIS_LOG_DEBUG("Allocated {} elements from free list: size={} KB",
                       num_elements, bytes_to_allocate / 1024);
      } else {
        IRIS_LOG_ERROR("Allocation failed: requested={} KB, capacity={} KB, free_list_size={}",
                       bytes_to_allocate / 1024, bytes_capacity_ / 1024, free_list_.size());
        return nullptr;
      }
    } else {
      ptr = heap_bases_[cur_rank_] + allocated_bytes_;
      allocated_bytes_ += bytes_to_allocate;
      bytes_capacity_ -= bytes_to_allocate;
      IRIS_LOG_DEBUG("Allocated {} elements: size={} KB, remaining_capacity={} MB",
                     num_elements, bytes_to_allocate / 1024, bytes_capacity_ / (1024 * 1024));
    }

    allocated_bytes_map_[ptr] = bytes_to_allocate;

    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    const auto ptr_uint = reinterpret_cast<std::uintptr_t>(ptr);
    const auto num_bytes_aligned = allocated_bytes_map_[ptr_uint];
    free_list_.insert({num_bytes_aligned, ptr_uint});
    allocated_bytes_map_.erase(ptr_uint);
  }

 private:
  std::size_t heap_size_bytes_;

  static constexpr std::size_t max_world_size = 8;
  uintptr_t heap_bases_[max_world_size];

  int cur_rank_;
  int world_size_;
  int device_id_;

  std::mutex global_mutex_;
  std::size_t allocated_bytes_;
  std::size_t bytes_capacity_;
  std::unordered_map<std::uintptr_t, std::size_t> allocated_bytes_map_;
  std::map<std::size_t, std::uintptr_t> free_list_;

 public:
  class iris_device_view {
   public:
    __host__ iris_device_view(const iris& iris) {
      cur_rank_ = iris.cur_rank_;
      world_size_ = iris.world_size_;

      std::memcpy(heap_bases_, iris.heap_bases_,
                  sizeof(uintptr_t) * iris.world_size_);

      static constexpr auto verbose = false;

      if (verbose) {
        printf("iris.world_size_: %d\n", iris.world_size_);
        printf("iris.heap_bases_: %p\n", iris.heap_bases_);
      }

      if (verbose) {
        for (std::size_t i = 0; i < world_size_; i++) {
          std::cout << "iris.Rank: " << cur_rank_ << " GPU: " << i
                    << " iris.heap base: " << iris.heap_bases_[i] << std::endl;
        }
      }
      if (verbose) {
        for (std::size_t i = 0; i < world_size_; i++) {
          std::cout << "Rank: " << cur_rank_ << " GPU: " << i
                    << " heap base: " << heap_bases_[i] << std::endl;
        }
      }
    }

    [[nodiscard]] __host__ __device__ int cur_rank() const { return cur_rank_; }
    [[nodiscard]] __host__ __device__ int world_size() const {
      return world_size_;
    }

    template <typename T, memory_scope scope = memory_scope_thread>
    [[nodiscard]] __host__ __device__ T
    atomic_load(const T* ptr, int remote_rank,
                memory_order order = memory_order_seq_cst) {
      const auto remote_ptr = translate(ptr, remote_rank);
      return __hip_atomic_load(remote_ptr, static_cast<int>(order), static_cast<int>(scope));
    }

    template <typename T, memory_scope scope = memory_scope_thread>
    __host__ __device__ void atomic_store(
        T* ptr, T value, int remote_rank,
        memory_order order = memory_order_seq_cst) {
      const auto remote_ptr = translate(ptr, remote_rank);
      __hip_atomic_store(remote_ptr, value, static_cast<int>(order), static_cast<int>(scope));
    }

    template <typename T>
    [[nodiscard]] __host__ __device__ T load(const T* ptr, int remote_rank) {
      const auto remote_ptr = translate(ptr, remote_rank);
      return *remote_ptr;
    }

    template <typename T>
    __host__ __device__ void store(T* ptr, T value, int remote_rank) {
      const auto remote_ptr = translate(ptr, remote_rank);
      *remote_ptr = value;
    }

    template <typename T, memory_scope scope = memory_scope_thread>
    __host__ __device__ T fetch_add(T* ptr, T value, int remote_rank,
                                    memory_order order = memory_order_seq_cst) {
      const auto remote_ptr = translate(ptr, remote_rank);
      return __hip_atomic_fetch_add(remote_ptr, value, static_cast<int>(order), static_cast<int>(scope));
    }

    template <typename T, memory_scope scope = memory_scope_thread>
    __host__ __device__ T fetch_sub(T* ptr, T value, int remote_rank,
                                    memory_order order = memory_order_seq_cst) {
      const auto remote_ptr = translate(ptr, remote_rank);
      return __hip_atomic_fetch_sub(remote_ptr, value, static_cast<int>(order), static_cast<int>(scope));
    }

    template <typename T, memory_scope scope = memory_scope_thread>
    __host__ __device__ bool compare_exchange_strong(
        T* ptr, T& expected, T desired, int remote_rank,
        memory_order order = memory_order_seq_cst) {
      const auto remote_ptr = translate(ptr, remote_rank);
      return __hip_atomic_compare_exchange_strong(remote_ptr, &expected,
                                                  desired, static_cast<int>(order),
                                                  static_cast<int>(order),
                                                  static_cast<int>(scope));
    }

    template <memory_scope scope = memory_scope_thread>
    __host__ __device__ void fence(memory_order order = memory_order_seq_cst) {
      if constexpr (scope == memory_scope_thread) {
      } else if constexpr (scope == memory_scope_warp) {
      } else if constexpr (scope == memory_scope_block) {
        __threadfence_block();
      } else if constexpr (scope == memory_scope_device) {
        __threadfence();
      } else if constexpr (scope == memory_scope_system) {
        __threadfence_system();
      } else {
        static_assert(false, "Invalid memory scope");
      }
    }

    [[nodiscard]] __host__ __device__ uintptr_t get_heap_base(int rank) const {
      return heap_bases_[rank];
    }

   private:
    int cur_rank_;
    int world_size_;
    static constexpr std::size_t max_world_size = 8;
    uintptr_t heap_bases_[max_world_size];

    template <typename T>
    [[nodiscard]] __host__ __device__ T* translate(const T* ptr,
                                                   int remote_rank) const {
      const auto offset =
          reinterpret_cast<uintptr_t>(ptr) - heap_bases_[cur_rank_];
      auto remote_heap_base = heap_bases_[remote_rank];
      return reinterpret_cast<T*>(remote_heap_base + offset);
    }
  };

  auto get_device_view() { return iris_device_view(*this); }
};

using iris_device_view = iris::iris_device_view;

}  // namespace iris

#undef hip_try
