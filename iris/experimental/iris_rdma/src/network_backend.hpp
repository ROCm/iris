// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once 

#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include <stdexcept>

#include "ibv_utils.hpp"
#include "queue_pair.hpp"
#include "torch_bootstrap.hpp"

// Vendor-specific headers
#ifdef HAVE_MLX5
#include <infiniband/mlx5dv.h>
#endif

#ifdef HAVE_BNXT
#include <infiniband/bnxt_re-abi.h>
#endif

namespace iris {

/**
 * @brief Main network backend for InfiniBand setup
 *
 * Handles:
 * - Device detection and initialization
 * - Protection domain creation
 * - Queue pair creation and state transitions
 * - Memory registration
 * - QP connection info exchange
 */
class network_backend {
 public:
  /**
   * @brief Constructor
   * @param bootstrap PyTorch bootstrap for cross-rank communication
   * @param device_name Optional device name (NULL for auto-detect)
   */
  network_backend(std::shared_ptr<rdma::torch_bootstrap> bootstrap,
                  const char* device_name = nullptr)
      : bootstrap_(bootstrap),
        requested_dev_(device_name),
        context_(nullptr),
        pd_orig_(nullptr),
        vendor_(rdma::nic_vendor::NONE),
        port_(1),
        gid_index_(0),
        heap_mr_(nullptr),
        heap_base_(0),
        heap_size_(0),
        mlx5dv_handle_(nullptr),
        bnxtdv_handle_(nullptr) {
    if (!bootstrap_) {
      throw std::runtime_error("Bootstrap cannot be null");
    }
    rank_ = bootstrap_->get_rank();
    world_size_ = bootstrap_->get_world_size();
    LOG_INFO("network_backend created: rank=%d, world_size=%d", rank_, world_size_);
  }

  /**
   * @brief Destructor - cleanup InfiniBand resources
   */
  ~network_backend() {
    LOG_DEBUG("network_backend cleanup started");
    
    qps_.clear();
    
    for (auto* cq : cqs_) {
      if (cq) {
        ibv_destroy_cq(cq);
      }
    }
    cqs_.clear();
    
    if (heap_mr_) {
      ibv_dereg_mr(heap_mr_);
      heap_mr_ = nullptr;
    }
    
    if (pd_orig_) {
      ibv_dealloc_pd(pd_orig_);
      pd_orig_ = nullptr;
    }
    
    if (context_) {
      ibv_close_device(context_);
      context_ = nullptr;
    }
    
    if (mlx5dv_handle_) {
      dlclose(mlx5dv_handle_);
      mlx5dv_handle_ = nullptr;
    }
    
    if (bnxtdv_handle_) {
      dlclose(bnxtdv_handle_);
      bnxtdv_handle_ = nullptr;
    }
    
    LOG_DEBUG("NetworkBackend cleanup completed");
  }

  /**
   * @brief Initialize the network (setup QPs, transition to RTS)
   */
  void init() {
    LOG_INFO("network_backend::init() started");
    
    autodetect_dv_libs();
    open_ib_device();
    create_queues();
    exchange_qp_dest_info();
    modify_qps_reset_to_init();
    modify_qps_init_to_rtr();
    modify_qps_rtr_to_rts();
    bootstrap_->barrier();
    
    LOG_INFO("network_backend::init() completed");
  }

  /**
   * @brief Register memory for RDMA
   * @param ptr Pointer to memory region
   * @param size Size in bytes
   */
  void register_memory(void* ptr, size_t size) {
    LOG_INFO("Registering memory: ptr=%p, size=%zu", ptr, size);

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    heap_mr_ = ibv_reg_mr(pd_orig_, ptr, size, access);
    if (heap_mr_ == nullptr) {
      int err = errno;
      fprintf(stderr, "[ERROR] ibv_reg_mr returned NULL for ptr=%p, size=%zu, errno=%d (%s)\n", 
              ptr, size, err, strerror(err));
      char error_msg[256];
      snprintf(error_msg, sizeof(error_msg), 
               "ibv_reg_mr failed with errno %d (%s) - GPUDirect RDMA may not be enabled", 
               err, strerror(err));
      throw std::runtime_error(error_msg);
    }

    // Store local heap base
    heap_base_ = reinterpret_cast<uint64_t>(ptr);
    heap_size_ = size;

    // Exchange remote keys
    rkeys_.resize(world_size_);
    std::vector<uint32_t> all_rkeys(world_size_);
    all_rkeys[rank_] = heap_mr_->rkey;
    bootstrap_->all_gather(all_rkeys.data(), sizeof(uint32_t));
    for (int i = 0; i < world_size_; i++) {
      rkeys_[i] = all_rkeys[i];
    }

    // Exchange heap base addresses (collective operation)
    remote_heap_bases_.resize(world_size_);
    std::vector<uint64_t> all_heap_bases(world_size_);
    all_heap_bases[rank_] = heap_base_;
    bootstrap_->all_gather(all_heap_bases.data(), sizeof(uint64_t));
    for (int i = 0; i < world_size_; i++) {
      remote_heap_bases_[i] = all_heap_bases[i];
    }

    // Update QPs with lkey and rkey
    uint32_t lkey = heap_mr_->lkey;
    for (int i = 0; i < world_size_; i++) {
      if (i < qps_.size() && qps_[i]) {
        qps_[i]->set_lkey(lkey);
        qps_[i]->set_rkey(rkeys_[i]);
      }
    }

    LOG_INFO("Memory registered: lkey=%u, rkey=%u, heap_base=%p", 
             lkey, heap_mr_->rkey, ptr);
  }

  /**
   * @brief Get queue pair for destination rank
   * @param dst_rank Destination rank
   * @return Pointer to QueuePair object
   */
  queue_pair* get_qp(int dst_rank) {
    if (dst_rank >= 0 && dst_rank < qps_.size()) {
      return qps_[dst_rank].get();
    }
    return nullptr;
  }

  /**
   * @brief Get QP info for Python
   * @param dst_rank Destination rank
   * @return QPInfo structure
   */
  rdma::qp_info_t get_qp_info(int dst_rank) {
    queue_pair* qp = get_qp(dst_rank);
    if (qp) {
      return qp->get_info();
    }
    return rdma::qp_info_t{0, 0, 0, dst_rank};
  }




  /**
   * @brief Get rank
   */
  int get_rank() const { return rank_; }

  /**
   * @brief Get world size
   */
  int get_world_size() const { return world_size_; }

  /**
   * @brief Get remote heap base address for a rank
   * @param rank Remote rank
   * @return Remote heap base address (0 if not registered)
   */
  uint64_t get_remote_heap_base(int rank) const {
    if (rank >= 0 && rank < remote_heap_bases_.size()) {
      return remote_heap_bases_[rank];
    }
    return 0;
  }

  /**
   * @brief Get local heap base address
   * @return Local heap base address (0 if not registered)
   */
  uint64_t get_heap_base() const { return heap_base_; }

  /**
   * @brief Get heap size
   * @return Heap size in bytes (0 if not registered)
   */
  size_t get_heap_size() const { return heap_size_; }

  /**
   * @brief RDMA Write operation
   * @param dst_rank Destination rank
   * @param local_addr Local buffer address
   * @param remote_addr Remote buffer address
   * @param size Size in bytes
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdma_write(int dst_rank, void* local_addr, uint64_t remote_addr, 
                size_t size, uint64_t wr_id = 0) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_addr;
    sge.length = size;
    sge.lkey = qp->get_lkey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = qp->get_rkey();

    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp->get_ibv_qp(), &wr, &bad_wr);
    
    LOG_DEBUG("RDMA Write to rank %d: local=%p remote=%lx size=%zu ret=%d", 
              dst_rank, local_addr, remote_addr, size, ret);
    
    return ret;
  }

  /**
   * @brief RDMA Read operation
   * @param dst_rank Destination rank
   * @param local_addr Local buffer address
   * @param remote_addr Remote buffer address
   * @param size Size in bytes
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdma_read(int dst_rank, void* local_addr, uint64_t remote_addr,
               size_t size, uint64_t wr_id = 0) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)local_addr;
    sge.length = size;
    sge.lkey = qp->get_lkey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = qp->get_rkey();

    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp->get_ibv_qp(), &wr, &bad_wr);
    
    LOG_DEBUG("RDMA Read from rank %d: local=%p remote=%lx size=%zu ret=%d", 
              dst_rank, local_addr, remote_addr, size, ret);
    
    return ret;
  }

  /**
   * @brief RDMA atomic fetch-and-add operation
   * @param dst_rank Destination rank
   * @param result_addr Local buffer to store the original value
   * @param remote_addr Remote address to perform atomic add on
   * @param add_value Value to add
   * @param size Size in bytes (must be 4 or 8)
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdma_atomic_fetch_add(int dst_rank, void* result_addr, uint64_t remote_addr,
                            uint64_t add_value, size_t size, uint64_t wr_id = 0) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    if (size != 4 && size != 8) {
      LOG_ERROR("Atomic operations only support 4 or 8 byte sizes, got %zu", size);
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)result_addr;
    sge.length = size;
    sge.lkey = qp->get_lkey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = qp->get_rkey();
    wr.wr.atomic.compare_add = add_value;  // Value to add

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp->get_ibv_qp(), &wr, &bad_wr);
    
    LOG_DEBUG("RDMA Atomic Fetch-Add to rank %d: result=%p remote=%lx add=%lu size=%zu ret=%d",
              dst_rank, result_addr, remote_addr, add_value, size, ret);
    
    return ret;
  }

  /**
   * @brief RDMA atomic exchange (swap) operation
   * @param dst_rank Destination rank
   * @param result_addr Local buffer to store the original value
   * @param remote_addr Remote address to exchange
   * @param new_value New value to write
   * @param size Size in bytes (must be 4 or 8)
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdma_atomic_exchange(int dst_rank, void* result_addr, uint64_t remote_addr,
                           uint64_t new_value, size_t size, uint64_t wr_id = 0) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    if (size != 4 && size != 8) {
      LOG_ERROR("Atomic operations only support 4 or 8 byte sizes, got %zu", size);
      return -1;
    }

    // For exchange, we need a staging buffer for the new value
    // ibverbs doesn't have a direct exchange, so we use CAS in a loop
    // But for simplicity, we can use MLX5 extended atomics if available
    // For now, we'll return an error and note this needs vendor-specific support
    LOG_ERROR("RDMA atomic exchange not yet implemented - needs vendor-specific support");
    return -1;
  }

  /**
   * @brief RDMA atomic compare-and-swap operation
   * @param dst_rank Destination rank
   * @param result_addr Local buffer to store the original value
   * @param remote_addr Remote address to perform CAS on
   * @param compare_value Expected value
   * @param swap_value Value to swap in if comparison succeeds
   * @param size Size in bytes (must be 4 or 8)
   * @param wr_id Work request ID (for completion tracking)
   * @return 0 on success, non-zero on error
   */
  int rdma_atomic_compare_swap(int dst_rank, void* result_addr, uint64_t remote_addr,
                                uint64_t compare_value, uint64_t swap_value,
                                size_t size, uint64_t wr_id = 0) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    if (size != 4 && size != 8) {
      LOG_ERROR("Atomic operations only support 4 or 8 byte sizes, got %zu", size);
      return -1;
    }

    struct ibv_sge sge;
    sge.addr = (uintptr_t)result_addr;
    sge.length = size;
    sge.lkey = qp->get_lkey();

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = qp->get_rkey();
    wr.wr.atomic.compare_add = compare_value;  // Expected value
    wr.wr.atomic.swap = swap_value;             // New value if compare succeeds

    struct ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp->get_ibv_qp(), &wr, &bad_wr);
    
    LOG_DEBUG("RDMA Atomic CAS to rank %d: result=%p remote=%lx compare=%lu swap=%lu size=%zu ret=%d",
              dst_rank, result_addr, remote_addr, compare_value, swap_value, size, ret);
    
    return ret;
  }

  /**
   * @brief Poll completion queue for RDMA operations
   * @param dst_rank Destination rank (to poll specific CQ)
   * @param max_completions Maximum number of completions to poll
   * @return Number of completions polled (negative on error)
   */
  int poll_cq(int dst_rank, int max_completions = 1) {
    queue_pair* qp = get_qp(dst_rank);
    if (!qp) {
      return -1;
    }

    struct ibv_wc wc[16];
    int num_to_poll = (max_completions < 16) ? max_completions : 16;
    int n = ibv_poll_cq(qp->get_ibv_cq(), num_to_poll, wc);
    
    if (n < 0) {
      LOG_ERROR_RANK(rank_, "CQ poll error for QP to rank %d", dst_rank);
      return n;
    }
    
    // Check for errors in completions
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        LOG_ERROR_RANK(rank_, "Work completion failed: status=%d (%s) opcode=%d wr_id=%lu dst_rank=%d",
                       wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].opcode, wc[i].wr_id, dst_rank);
        return -1;
      }
      LOG_DEBUG_RANK(rank_, "Work completion: status=SUCCESS opcode=%d wr_id=%lu dst_rank=%d", 
                     wc[i].opcode, wc[i].wr_id, dst_rank);
    }
    
    // Dump CQ info and check its healthy
    qp->dump_cq_info();

    //LOG_DEBUG_RANK(rank_, "Polled %d completions from CQ for QP to rank %d", n, dst_rank);
    return n;
  }



 private:
  // Bootstrap
  std::shared_ptr<rdma::torch_bootstrap> bootstrap_;
  int rank_;
  int world_size_;

  // Device configuration
  const char* requested_dev_;
  struct ibv_context* context_;
  struct ibv_pd* pd_orig_;
  struct ibv_device_attr device_attr_;
  rdma::nic_vendor vendor_;

  // Port configuration
  struct ibv_port_attr portinfo_;
  union ibv_gid gid_;
  int port_;
  int gid_index_;

  // Memory registration
  struct ibv_mr* heap_mr_;
  std::vector<uint32_t> rkeys_;  // Remote keys from all ranks
  uint64_t heap_base_;  // Local heap base address
  size_t heap_size_;  // Local heap size
  std::vector<uint64_t> remote_heap_bases_;  // Heap base addresses from all ranks

  // Queue pairs
  std::vector<std::unique_ptr<queue_pair>> qps_;
  std::vector<struct ibv_cq*> cqs_;
  std::vector<rdma::qp_dest_info_t> dest_info_;

  // Dynamic library handles for vendor-specific libraries
  void* mlx5dv_handle_;
  void* bnxtdv_handle_;

  // Setup functions (extracted from rocSHMEM)

  // Vendor-specific init
  void autodetect_dv_libs() {
    LOG_DEBUG("Auto-detecting vendor libraries...");

    // Try MLX5
    if (mlx5_dv_dl_init() == 0) {
      vendor_ = rdma::nic_vendor::MLX5;
      LOG_INFO("Detected MLX5 vendor");
      return;
    }

    // Try BNXT
    if (bnxt_dv_dl_init() == 0) {
      vendor_ = rdma::nic_vendor::BNXT;
      LOG_INFO("Detected BNXT vendor");
      return;
    }

    // Default to standard verbs
    vendor_ = rdma::nic_vendor::NONE;
    LOG_INFO("Using standard InfiniBand verbs");
  }

  int mlx5_dv_dl_init() {
    mlx5dv_handle_ = dlopen("libmlx5.so", RTLD_NOW);
    if (!mlx5dv_handle_) {
      mlx5dv_handle_ = dlopen("libmlx5.so.1", RTLD_NOW);
    }

    if (!mlx5dv_handle_) {
      LOG_DEBUG("Could not open libmlx5.so");
      return -1;
    }

    return 0;
  }

  int bnxt_dv_dl_init() {
    bnxtdv_handle_ = dlopen("libbnxt_re.so", RTLD_NOW);
    if (!bnxtdv_handle_) {
      bnxtdv_handle_ = dlopen("/usr/local/lib/libbnxt_re.so", RTLD_NOW);
    }

    if (!bnxtdv_handle_) {
      LOG_DEBUG("Could not open libbnxt_re.so");
      return -1;
    }

    return 0;
  }

  void open_ib_device() {
    LOG_INFO("Opening InfiniBand device...");

    struct ibv_device** device_list = nullptr;
    struct ibv_device* device = nullptr;
    int num_devices = 0;

    device_list = ibv_get_device_list(&num_devices);
    CHECK_NNULL(device_list, "ibv_get_device_list");

    if (num_devices == 0) {
      throw std::runtime_error("No InfiniBand devices found");
    }

    // Select device
    device = device_list[0];  // Default to first device

    if (requested_dev_) {
      for (int i = 0; i < num_devices; i++) {
        const char* dev_name = ibv_get_device_name(device_list[i]);
        CHECK_NNULL(dev_name, "ibv_get_device_name");

        if (strstr(dev_name, requested_dev_)) {
          device = device_list[i];
          break;
        }
      }
    }

    // Open device
    context_ = ibv_open_device(device);
    CHECK_NNULL(context_, "ibv_open_device");
    rdma::dump_ibv_context(context_);
    rdma::dump_ibv_device(context_->device);

    // Query device attributes (needed for atomic operations)
    int err = ibv_query_device(context_, &device_attr_);
    CHECK_ZERO(err, "ibv_query_device");
    LOG_DEBUG("Device attributes: max_qp_rd_atom=%d max_qp_init_rd_atom=%d",
              device_attr_.max_qp_rd_atom, device_attr_.max_qp_init_rd_atom);

    // Allocate protection domain
    pd_orig_ = ibv_alloc_pd(context_);
    CHECK_NNULL(pd_orig_, "ibv_alloc_pd");
    rdma::dump_ibv_pd(pd_orig_);

    // Query port
    err = ibv_query_port(context_, port_, &portinfo_);
    CHECK_ZERO(err, "ibv_query_port");
    rdma::dump_ibv_port_attr(&portinfo_);

    // Select GID index
    select_gid_index();

    ibv_free_device_list(device_list);

    LOG_INFO("InfiniBand device opened: %s",
             ibv_get_device_name(context_->device));
  }

  void select_gid_index() {
    LOG_DEBUG("Selecting GID index...");

    const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
    int selected_gid_index = -1;
    union ibv_gid selected_gid;
    int err;

    int gid_tbl_len = portinfo_.gid_tbl_len;

    for (int i = 0; i < gid_tbl_len; i++) {
      union ibv_gid current_gid;
      err = ibv_query_gid(context_, port_, i, &current_gid);
      if (err != 0)
        continue;

      // Skip local GIDs
      if (memcmp(current_gid.raw, &local_gid_prefix, 2) == 0) {
        continue;
      }

      // Use first non-local GID
      if (selected_gid_index == -1) {
        selected_gid_index = i;
        selected_gid = current_gid;
        break;
      }
    }

    if (selected_gid_index == -1) {
      selected_gid_index = 0;
      err = ibv_query_gid(context_, port_, 0, &selected_gid);
      CHECK_ZERO(err, "ibv_query_gid");
    }

    gid_index_ = selected_gid_index;
    gid_ = selected_gid;

    LOG_DEBUG("Selected GID index: %d", gid_index_);
  }

  void create_queues() {
    LOG_DEBUG("Creating queues...");

    int ncqes = 64;      // Number of CQ entries
    int sq_length = 64;  // Send queue length // TODO: FIX THAT

    // Resize vectors
    dest_info_.resize(world_size_);
    cqs_.resize(world_size_);
    qps_.resize(world_size_);

    // Create CQs and QPs
    create_cqs(ncqes);
    create_qps(sq_length);

    LOG_INFO("Created %d queue pairs", world_size_);
  }

  void create_cqs(int ncqes) {
    LOG_DEBUG("Creating completion queues: ncqes=%d", ncqes);

    for (int i = 0; i < world_size_; i++) {
      cqs_[i] = ibv_create_cq(context_, ncqes, nullptr, nullptr, 0);
      CHECK_NNULL(cqs_[i], "ibv_create_cq");
    }
  }

  void create_qps(int sq_length) {
    LOG_DEBUG("Creating queue pairs: sq_length=%d", sq_length);

    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.cap.max_send_wr = sq_length;
    attr.cap.max_send_sge = 1;
    attr.cap.max_inline_data = 8;
    attr.sq_sig_all = 0;
    attr.qp_type = IBV_QPT_RC;

    for (int i = 0; i < world_size_; i++) {
      attr.send_cq = cqs_[i];
      attr.recv_cq = cqs_[i];

      struct ibv_qp* qp = ibv_create_qp(pd_orig_, &attr);
      CHECK_NNULL(qp, "ibv_create_qp");

      qps_[i] = std::make_unique<queue_pair>(qp, cqs_[i], i, vendor_);
    }
  }

  void exchange_qp_dest_info() {
    LOG_DEBUG("Exchanging QP destination info...");

    // Fill local dest info
    for (int i = 0; i < world_size_; i++) {
      dest_info_[i].lid = portinfo_.lid;
      dest_info_[i].qpn = qps_[i]->get_qp_num();
      dest_info_[i].psn = 0;
      dest_info_[i].gid = gid_;
    }

    // All-gather dest info
    bootstrap_->all_gather(dest_info_.data(), sizeof(rdma::qp_dest_info_t));

    LOG_DEBUG("QP destination info exchanged");
  }

  void modify_qps_reset_to_init() {
    LOG_DEBUG("Transitioning QPs: RESET -> INIT");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    for (int i = 0; i < world_size_; i++) {
      int err = ibv_modify_qp(qps_[i]->get_ibv_qp(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (RESET->INIT)");
    }
  }

  void modify_qps_init_to_rtr() {
    LOG_DEBUG("Transitioning QPs: INIT -> RTR");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = portinfo_.active_mtu;
    attr.min_rnr_timer = 12;
    attr.max_dest_rd_atomic = device_attr_.max_qp_rd_atom;  // Use device capability
    attr.ah_attr.port_num = port_;

    if (portinfo_.link_layer == IBV_LINK_LAYER_ETHERNET) {
      attr.ah_attr.grh.sgid_index = gid_index_;
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.sl = 1;
      attr.ah_attr.grh.traffic_class = 0;
    }

    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN |
                    IBV_QP_DEST_QPN | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER;

    for (int i = 0; i < world_size_; i++) {
      attr.rq_psn = dest_info_[i].psn;
      attr.dest_qp_num = dest_info_[i].qpn;

      if (portinfo_.link_layer == IBV_LINK_LAYER_ETHERNET) {
        memcpy(&attr.ah_attr.grh.dgid, &dest_info_[i].gid, 16);
      } else {
        attr.ah_attr.dlid = dest_info_[i].lid;
      }

      int err = ibv_modify_qp(qps_[i]->get_ibv_qp(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (INIT->RTR)");
    }
  }

  void modify_qps_rtr_to_rts() {
    LOG_DEBUG("Transitioning QPs: RTR -> RTS");

    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = device_attr_.max_qp_init_rd_atom;  // Use device capability

    int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC |
                    IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

    for (int i = 0; i < world_size_; i++) {
      attr.sq_psn = dest_info_[i].psn;

      int err = ibv_modify_qp(qps_[i]->get_ibv_qp(), &attr, attr_mask);
      CHECK_ZERO(err, "modify_qp (RTR->RTS)");
    }
  }

};

}  // namespace iris
