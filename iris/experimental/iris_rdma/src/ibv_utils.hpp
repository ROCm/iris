// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "logging.hpp"

namespace iris {
namespace rdma {

// Error checking macros
#define CHECK_ZERO(expr, msg)                                           \
  do {                                                                  \
    int ret = (expr);                                                   \
    if (ret != 0) {                                                     \
      LOG_ERROR("%s failed with code %d: %s", msg, ret, strerror(ret)); \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define CHECK_NNULL(ptr, msg)                             \
  do {                                                    \
    if ((ptr) == nullptr) {                               \
      LOG_ERROR("%s returned NULL", msg);                 \
      abort();                                            \
    }                                                     \
  } while (0)

// Vendor detection
enum class nic_vendor { NONE, IONIC, BNXT, MLX5 };

// QP destination info for connection
struct qp_dest_info_t {
  int lid;
  int qpn;
  int psn;
  union ibv_gid gid;
};

// QP metadata exposed to Python
struct qp_info_t {
  uint32_t qp_num;
  uint32_t lkey;
  uint32_t rkey;
  int dst_rank;
};

// Helper functions
inline void dump_ibv_device(struct ibv_device* device) {
  LOG_DEBUG("IBV Device: %s", ibv_get_device_name(device));
}

inline void dump_ibv_context(struct ibv_context* ctx) {
  LOG_DEBUG("IBV Context: device=%s", ctx->device->name);
}

inline void dump_ibv_pd(struct ibv_pd* pd) {
  LOG_DEBUG("IBV PD: handle=%u", pd->handle);
}

inline void dump_ibv_port_attr(struct ibv_port_attr* attr) {
  LOG_DEBUG("Port Attr: state=%d, lid=%d, link_layer=%d, active_mtu=%d",
            attr->state, attr->lid, attr->link_layer, attr->active_mtu);
}

inline int ibv_mtu_to_int(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:
      return 256;
    case IBV_MTU_512:
      return 512;
    case IBV_MTU_1024:
      return 1024;
    case IBV_MTU_2048:
      return 2048;
    case IBV_MTU_4096:
      return 4096;
    default:
      fprintf(stderr, "[ERROR] Invalid ibv_mtu\n");
      return 0;
  }
}

}  // namespace rdma
}  // namespace iris

