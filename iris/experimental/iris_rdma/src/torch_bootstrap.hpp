// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstring>
#include "ibv_utils.hpp"

namespace iris {
namespace rdma {

/**
 * @brief Bootstrap implementation using PyTorch Distributed
 *
 * Wraps PyTorch's c10d process group to provide synchronization
 * primitives needed for InfiniBand setup (all_gather, barrier)
 */
class torch_bootstrap {
 public:
  /**
   * @brief Constructor
   * @param process_group PyTorch distributed process group
   */
  inline explicit torch_bootstrap(c10::intrusive_ptr<c10d::ProcessGroup> process_group)
      : process_group_(process_group) {
    if (!process_group_) {
      throw std::runtime_error("Process group cannot be null");
    }
    rank_ = process_group_->getRank();
    world_size_ = process_group_->getSize();
    LOG_INFO("torch_bootstrap initialized: rank=%d, world_size=%d", rank_, world_size_);
  }

  /**
   * @brief Get rank of current process
   */
  int get_rank() const { return rank_; }

  /**
   * @brief Get total number of ranks
   */
  int get_world_size() const { return world_size_; }

  /**
   * @brief All-gather operation
   *
   * Gathers data from all ranks. Each rank contributes 'size' bytes
   * starting at allData[rank * size].
   *
   * @param allData Buffer to hold all gathered data (world_size * size bytes)
   * @param size Size of data contributed by each rank
   */
  inline void all_gather(void* allData, int size) {
    auto cpu_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto cuda_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

    auto cpu_input = torch::from_blob(
        static_cast<uint8_t*>(allData) + rank_ * size, {size}, cpu_options);
    auto input = cpu_input.to(torch::kCUDA);

    std::vector<at::Tensor> output_tensors;
    for (int i = 0; i < world_size_; i++) {
      output_tensors.push_back(torch::empty({size}, cuda_options));
    }

    std::vector<std::vector<at::Tensor>> output_tensor_lists = {output_tensors};
    std::vector<at::Tensor> input_tensors = {input};
    auto work = process_group_->allgather(output_tensor_lists, input_tensors);
    work->wait();

    for (int i = 0; i < world_size_; i++) {
      auto cpu_output = output_tensors[i].to(torch::kCPU);
      std::memcpy(static_cast<uint8_t*>(allData) + i * size,
                  cpu_output.data_ptr<uint8_t>(), size);
    }
    LOG_DEBUG("AllGather completed: %d bytes per rank", size);
  }

  /**
   * @brief Barrier synchronization
   *
   * Blocks until all ranks reach the barrier
   */
  inline void barrier() {
    auto work = process_group_->barrier();
    work->wait();
    LOG_DEBUG("Barrier completed");
  }

  /**
   * @brief Point-to-point send (optional, not needed for basic setup)
   */
  inline void send(void* data, int size, int peer, int tag) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto tensor = torch::from_blob(static_cast<uint8_t*>(data), {size}, options);
    std::vector<at::Tensor> tensors = {tensor};
    auto work = process_group_->send(tensors, peer, tag);
    work->wait();
  }

  /**
   * @brief Point-to-point receive (optional, not needed for basic setup)
   */
  inline void recv(void* data, int size, int peer, int tag) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto tensor = torch::from_blob(static_cast<uint8_t*>(data), {size}, options);
    std::vector<at::Tensor> tensors = {tensor};
    auto work = process_group_->recv(tensors, peer, tag);
    work->wait();
  }

 private:
  c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
  int rank_;
  int world_size_;
};

}  // namespace rdma
}  // namespace iris

