// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

/*
 * Flat C ABI over the RCCL host calls Iris needs to drive GIN from Python.
 *
 * ctypes could call most of RCCL directly, but ncclDevCommRequirements,
 * ncclCommProperties and ncclDevComm are versioned structs whose layout belongs
 * to the RCCL headers. Building this file against those headers keeps that
 * layout out of Python entirely: Python sees opaque handles and plain ints.
 *
 * Every function returns an ncclResult_t (0 on success); iris_gin_error_string
 * turns it into text.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <new>

#include "nccl.h"
#include "nccl_device.h"

namespace {

struct DevCommHandle {
  ncclComm_t comm;
  ncclDevComm host;  // the copy RCCL filled in; needed again by ncclDevCommDestroy
  void* device;      // the same bytes in device memory, passed to kernels by address
};

}  // namespace

extern "C" {

const char* iris_gin_error_string(int result) { return ncclGetErrorString(static_cast<ncclResult_t>(result)); }

int iris_gin_version(int* version) { return ncclGetVersion(version); }

// NCCL_VERSION_CODE of the headers this helper -- and, built by the same
// build.sh run, the device bitcode -- was compiled against. The bitcode bakes
// in the ncclDevComm layout, so it must match the librccl loaded at run time.
int iris_gin_header_version() { return NCCL_VERSION_CODE; }

int iris_gin_unique_id_bytes() { return NCCL_UNIQUE_ID_BYTES; }

int iris_gin_get_unique_id(char* out) {
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  if (r == ncclSuccess) std::memcpy(out, id.internal, NCCL_UNIQUE_ID_BYTES);
  return r;
}

int iris_gin_comm_init(void** out_comm, int nranks, const char* unique_id, int rank) {
  ncclUniqueId id;
  std::memcpy(id.internal, unique_id, NCCL_UNIQUE_ID_BYTES);
  ncclComm_t comm = nullptr;
  ncclResult_t r = ncclCommInitRank(&comm, nranks, id, rank);
  *out_comm = comm;
  return r;
}

int iris_gin_comm_destroy(void* comm) {
  ncclComm_t c = static_cast<ncclComm_t>(comm);
  ncclResult_t r = ncclCommFinalize(c);
  if (r != ncclSuccess) return r;
  return ncclCommDestroy(c);
}

// device_api, gin_type, n_lsa_teams: what the communicator can do. gin_type is
// ncclGinType_t (0 = none, 2 = IB proxy, ...).
int iris_gin_comm_query(void* comm, int* device_api, int* gin_type, int* n_lsa_teams) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t r = ncclCommQueryProperties(static_cast<ncclComm_t>(comm), &props);
  if (r != ncclSuccess) return r;
  *device_api = props.deviceApiSupport ? 1 : 0;
  *gin_type = static_cast<int>(props.ginType);
  *n_lsa_teams = props.nLsaTeams;
  return ncclSuccess;
}

// Collective: every rank must call with the same size. Returns the local
// pointer and the window handle (a device pointer, usable from kernels).
int iris_gin_window_alloc(void* comm, size_t bytes, void** out_ptr, void** out_win) {
  void* ptr = nullptr;
  ncclResult_t r = ncclMemAlloc(&ptr, bytes);
  if (r != ncclSuccess) return r;
  ncclWindow_t win = nullptr;
  r = ncclCommWindowRegister(static_cast<ncclComm_t>(comm), ptr, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);
  if (r != ncclSuccess) {
    ncclMemFree(ptr);
    return r;
  }
  *out_ptr = ptr;
  *out_win = win;
  return ncclSuccess;
}

int iris_gin_window_free(void* comm, void* ptr, void* win) {
  ncclResult_t r = ncclCommWindowDeregister(static_cast<ncclComm_t>(comm), static_cast<ncclWindow_t>(win));
  if (r != ncclSuccess) return r;
  return ncclMemFree(ptr);
}

// Collective. Creates a device communicator with full GIN connectivity (if
// context_count > 0) and
// `signal_count` signals, and copies it to device memory. *out_device is what
// kernels receive; *out_handle is what iris_gin_devcomm_destroy takes back.
int iris_gin_devcomm_create(void* comm, int context_count, int signal_count, int counter_count, void** out_handle,
                            void** out_device, int* out_context_count, int* out_rank, int* out_nranks,
                            int* out_lsa_rank, int* out_lsa_size) {
  DevCommHandle* h = new (std::nothrow) DevCommHandle{};
  if (h == nullptr) return ncclSystemError;
  h->comm = static_cast<ncclComm_t>(comm);

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  // context_count == 0 asks for a device comm without GIN, e.g. to read the LSA
  // team geometry. It must also leave the connection type at its NONE default:
  // RCCL 2.30.4 corrupts the heap given FULL connectivity with zero contexts.
  if (context_count > 0) reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginContextCount = context_count;
  reqs.ginSignalCount = signal_count;
  reqs.ginCounterCount = counter_count;
#if NCCL_VERSION_CODE >= 23100
  // Kernels only use weak signals, each carried by the put it covers.
  reqs.ginStrongSignalsRequired = false;
#endif
  ncclResult_t r = ncclDevCommCreate(h->comm, &reqs, &h->host);
  if (r != ncclSuccess) {
    delete h;
    return r;
  }
  if (hipMalloc(&h->device, sizeof(ncclDevComm)) != hipSuccess ||
      hipMemcpy(h->device, &h->host, sizeof(ncclDevComm), hipMemcpyHostToDevice) != hipSuccess) {
    ncclDevCommDestroy(h->comm, &h->host);
    delete h;
    return ncclUnhandledCudaError;
  }
  *out_handle = h;
  *out_device = h->device;
  *out_context_count = static_cast<int>(h->host.ginContextCount);
  *out_rank = h->host.rank;
  *out_nranks = h->host.nRanks;
  *out_lsa_rank = h->host.lsaRank;
  *out_lsa_size = h->host.lsaSize;
  return ncclSuccess;
}

int iris_gin_devcomm_destroy(void* handle) {
  DevCommHandle* h = static_cast<DevCommHandle*>(handle);
  (void)hipFree(h->device);
  ncclResult_t r = ncclDevCommDestroy(h->comm, &h->host);
  delete h;
  return r;
}

}  // extern "C"
