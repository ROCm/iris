// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <infiniband/verbs.h>
#include "ibv_utils.hpp"

namespace iris {

/**
 * @brief Simplified Queue Pair wrapper for host-side operations
 *
 * Unlike the full rocSHMEM QueuePair, this version only maintains
 * metadata needed for RDMA operations from Python/host code.
 */
class queue_pair {
 public:
  /**
   * @brief Constructor
   * @param qp InfiniBand queue pair
   * @param cq InfiniBand completion queue
   * @param dst_rank Destination rank for this QP
   * @param vendor NIC vendor type
   */
  inline queue_pair(struct ibv_qp* qp,
                    struct ibv_cq* cq,
                    int dst_rank,
                    rdma::nic_vendor vendor)
      : qp_(qp),
        cq_(cq),
        dst_rank_(dst_rank),
        vendor_(vendor),
        lkey_(0),
        rkey_(0) {
    CHECK_NNULL(qp_, "QueuePair: ibv_qp");
    CHECK_NNULL(cq_, "QueuePair: ibv_cq");
    qp_num_ = qp_->qp_num;
    LOG_DEBUG("queue_pair created: qp_num=%u, dst_rank=%d", qp_num_, dst_rank_);
  }

  /**
   * @brief Destructor
   */
  inline ~queue_pair() {
    LOG_DEBUG("queue_pair destroyed: qp_num=%u, dst_rank=%d", qp_num_, dst_rank_);
  }

  /**
   * @brief Get QP number
   */
  uint32_t get_qp_num() const { return qp_num_; }

  /**
   * @brief Get local key for memory region
   */
  uint32_t get_lkey() const { return lkey_; }

  /**
   * @brief Get remote key for destination rank
   */
  uint32_t get_rkey() const { return rkey_; }

  /**
   * @brief Get destination rank
   */
  int get_dst_rank() const { return dst_rank_; }

  /**
   * @brief Set remote key (after exchange)
   */
  void set_rkey(uint32_t rkey) { rkey_ = rkey; }

  /**
   * @brief Set local key (from memory registration)
   */
  void set_lkey(uint32_t lkey) { lkey_ = lkey; }

  /**
   * @brief Get underlying ibv_qp pointer
   */
  struct ibv_qp* get_ibv_qp() { return qp_; }

  /**
   * @brief Get underlying ibv_cq pointer
   */
  struct ibv_cq* get_ibv_cq() { return cq_; }

  /**
   * @brief Get QP info for Python
   */
  inline rdma::qp_info_t get_info() const {
    rdma::qp_info_t info;
    info.qp_num = qp_num_;
    info.lkey = lkey_;
    info.rkey = rkey_;
    info.dst_rank = dst_rank_;
    return info;
  }


  void dump_cq_info() const {
    LOG_DEBUG("cq: %p", cq_);
    LOG_DEBUG("handle: %u", cq_->channel);
    LOG_DEBUG("cq_context: %p", cq_->cq_context);
    LOG_DEBUG("context: %p", cq_->context);
    LOG_DEBUG("cqe: %u", cq_->cqe);
    LOG_DEBUG("comp_events_completed: %u", cq_->comp_events_completed);
    LOG_DEBUG("async_events_completed: %u", cq_->async_events_completed);

  }
 private:
  struct ibv_qp* qp_;
  struct ibv_cq* cq_;
  int dst_rank_;
  rdma::nic_vendor vendor_;

  uint32_t qp_num_;
  uint32_t lkey_;
  uint32_t rkey_;
};

}  // namespace iris

