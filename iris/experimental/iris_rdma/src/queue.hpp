// GPU-to-CPU Queue - C++ Host Side
// Exposes queue pointer to Python/Triton

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

namespace iris {
namespace rdma {

// Operation types - simplified for Iris
enum class operation_type : uint8_t {
  NOP = 0,
  PUT = 1,         // RDMA write
  GET = 2,         // RDMA read
  FLUSH = 3,       // Flush connection
  ATOMIC_ADD = 4,  // Atomic add
  ATOMIC_EXCH = 5, // Atomic exchange
  ATOMIC_CAS = 6,  // Atomic compare-and-swap
};

// Work item structure - metadata only, no data storage
// Data is stored in the registered symmetric heap
struct alignas(16) work_item_header_t {
  uint64_t dst_ptr;     // Destination pointer (where to write on remote)
  uint64_t src_ptr;     // Source pointer (offset in local registered heap)
  uint32_t size_bytes;  // Size in bytes to transfer (WRITE LAST as ready flag)
  uint16_t rank;        // Remote rank
  uint8_t op_type;      // Operation type (see operation_type enum)
  uint8_t reserved;     // Reserved for future use
};

// Note: Completion is signaled by tail pointer advancement, not a flag
struct alignas(16) work_item_t {
  work_item_header_t header;    // 32 bytes (0-31, padded due to alignas(16))
  // For atomic operations: operand values
  uint64_t atomic_operand;      // Value to add/exchange (offset 32)
  uint64_t atomic_compare;      // For CAS: compare value (offset 40)
  // Total size: 48 bytes
};

// Queue state visible to both CPU and GPU
struct queue_state_t {
  work_item_t* items;   // Queue buffer (pinned host memory)
  uint64_t* head;       // Head pointer (device memory, GPU writes)
  uint64_t* tail;       // Tail pointer (host memory, CPU writes, GPU reads)
  uint64_t* tailCache;  // Cached tail (device memory)
  int32_t size;         // Queue capacity
};

// CPU-side queue management
class queue {
 public:
  explicit queue(int size = 512) : size_(size) {
    // Allocate pinned memory for queue_state_t struct (GPU needs to read this)
    hipHostMalloc(&state_, sizeof(queue_state_t));

    // Allocate pinned memory for queue items
    hipHostMalloc(&state_->items, size * sizeof(work_item_t));
    memset(state_->items, 0, size * sizeof(work_item_t));

    // Allocate device memory for head
    hipMalloc(&state_->head, sizeof(uint64_t));
    hipMemset(state_->head, 0, sizeof(uint64_t));

    // Allocate pinned memory for tail (CPU writes, GPU reads)
    hipHostMalloc(&state_->tail, sizeof(uint64_t));
    *state_->tail = 0;

    // Allocate device memory for tail cache
    hipMalloc(&state_->tailCache, sizeof(uint64_t));
    hipMemset(state_->tailCache, 0, sizeof(uint64_t));

    state_->size = size;
  }

  ~queue() {
    hipHostFree(state_->items);
    hipFree(state_->head);
    hipHostFree(state_->tail);
    hipFree(state_->tailCache);
    hipHostFree(state_);
  }

  // Get raw pointer to queue state for Triton
  queue_state_t* get_queue_ptr() { return state_; }

  // Poll for new work item (non-blocking)
  bool poll(work_item_t& item) {
    uint64_t currentTail = *state_->tail;
    work_item_t* ptr = &state_->items[currentTail % size_];

    // Atomic load of size_bytes (acquire semantics) - use as ready flag
    // size_bytes == 0 means slot is empty/processed
    uint32_t size_bytes =
        reinterpret_cast<std::atomic<uint32_t>*>(&ptr->header.size_bytes)->load(std::memory_order_acquire);

    // Check if slot is ready
    if (size_bytes == 0) {
      return false;  // Queue empty
    }

    // Copy entire work item (just header now, no data array)
    memcpy(&item, ptr, sizeof(work_item_t));

    return true;
  }

  // Mark work item as processed
  void pop() {
    uint64_t currentTail = *state_->tail;

    // Clear the size_bytes to mark as processed
    state_->items[currentTail % size_].header.size_bytes = 0;

    // Advance tail with release semantics (GPU will reload this into tailCache)
    uint64_t newTail = currentTail + 1;
    reinterpret_cast<std::atomic<uint64_t>*>(state_->tail)->store(newTail, std::memory_order_release);
  }

  // Get queue statistics
  uint64_t get_tail() const { return *state_->tail; }

  uint64_t get_head() const {
    uint64_t h;
    hipMemcpy(&h, state_->head, sizeof(uint64_t), hipMemcpyDeviceToHost);
    return h;
  }

  int get_size() const { return size_; }
  
  // Check if queue is empty (all work processed)
  bool is_empty() const {
    uint64_t h;
    hipMemcpy(&h, state_->head, sizeof(uint64_t), hipMemcpyDeviceToHost);
    return h == *state_->tail;
  }

 private:
  queue_state_t* state_;
  int size_;
};

}  // namespace rdma
}  // namespace iris
