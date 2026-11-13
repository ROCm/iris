// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

/*
 * extern "C" __device__ entry points over RCCL's GIN device API, for Triton.
 *
 * RCCL already ships an IR binding layer (librccl_device.bc, -DEMIT_LLVM_IR=ON)
 * meant for JIT consumers, but its thunks take ncclTeam and ncclCoopAny by value
 * and ncclGin_C by pointer-to-struct. Triton's extern_elementwise types every
 * argument as a scalar or a pointer, so it cannot call those. This file is the
 * scalar-ABI equivalent: it includes the same device headers and instantiates
 * the templates itself, so the result is the same device code with a signature
 * Triton can express.
 *
 * Calling convention, which every function here relies on:
 *
 *   - Every thread of the Triton program makes the call. A scalar
 *     extern_elementwise call is emitted once per thread, not once per program,
 *     so the functions use ncclCoopCta and let RCCL pick the issuing thread.
 *     ncclCoopCta synchronises the whole workgroup, so the call must sit in
 *     program-uniform control flow (branching on tl.program_id is fine,
 *     branching on a per-lane value is not).
 *   - Pointers and RCCL handles travel as int64. ncclWindow_t is a device
 *     pointer, and the ncclDevComm is copied into device memory by the host
 *     helper and passed by address.
 *   - Functions that are void in RCCL return int32 0, because Triton cannot
 *     type a void extern result.
 *
 * No device globals: the IB-proxy GIN backend keeps all of its state behind
 * pointers in ncclDevComm, so linking this into a Triton module needs no
 * module-level initialisation. Only the proxy backend (NCCL_GIN_TYPE=2) is
 * compiled in. The rocSHMEM GDA backend read __constant__ state that a
 * separately loaded Triton module would not have, until ROCm/rocm-systems#14034
 * removed it; supporting GDA is planned once a RCCL with that change ships.
 *
 * Must be compiled against the headers of the librccl it runs with:
 * sizeof(ncclDevComm) and its field offsets come from them.
 */
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nccl.h"
#include "nccl_device.h"

namespace {

// RCCL 2.31 split signal actions into Weak* and Strong*; 2.30 has only the
// unqualified ones, which are weak there. Kernels here only need weak.
#if NCCL_VERSION_CODE >= 23100
using SignalAdd = ncclGin_WeakSignalAdd;
#else
using SignalAdd = ncclGin_SignalAdd;
#endif

__device__ inline ncclDevComm const& as_comm(int64_t p) { return *reinterpret_cast<ncclDevComm const*>(p); }

__device__ inline ncclWindow_t as_win(int64_t w) { return reinterpret_cast<ncclWindow_t>(w); }

// Whether this RCCL has gin.flush(coop, order, descriptor, cycles), the timed
// overload (present in 2.32, absent in 2.30.x). Detected rather than keyed on a
// version number, since the exact release that added it is not pinned down.
template <typename G, typename = void>
struct HasTimedFlush : std::false_type {};
template <typename G>
struct HasTimedFlush<G, std::void_t<decltype(std::declval<G const&>().flush(
                            ncclCoopCta{}, cuda::memory_order_acquire, ncclGin_None{}, uint64_t{}))>>
    : std::true_type {};

// A template, so that the branch a RCCL lacks is discarded rather than checked.
template <typename G>
__device__ int32_t flush_bounded(G const& gin, uint64_t cycles) {
  if constexpr (HasTimedFlush<G>::value) {
    int r = static_cast<int>(gin.flush(ncclCoopCta{}, cuda::memory_order_acquire, ncclGin_None{}, cycles));
    return __syncthreads_or(r != ncclSuccess) ? 1 : 0;
  } else {
    (void)cycles;
    gin.flush(ncclCoopCta{});
    return 0;
  }
}

// Make a put's queue descriptor visible to RCCL's CPU proxy now, not at kernel
// end. RCCL normally writes the descriptor with system-scope 128-bit stores,
// which need nothing more. build.sh selects RCCL's fallback instead
// (DWORDX4_INTRINSICS_FORCE_OFF, because Triton's LLVM cannot lower those
// stores), and the fallback's non-temporal stores can stay in this XCD's L2
// until the kernel exits. Until then the proxy cannot see the put, so anything
// in the same kernel that waits on it -- flush, or a reply the put triggers --
// never returns. Measured: flush after a put timed out every time without this
// fence and completed every time with it.
//
// It is the same hazard, and the same remedy, as the producer side of
// Iris's triggered plans (docs/conceptual/triggered-plans.md, "Memory
// ordering"): an agent outside the shader reads memory behind L2, so the
// writer releases at system scope, which writes back L2 first. Every thread
// fences, including the one that wrote the descriptor, so no barrier is needed.
__device__ inline void publish_descriptor() { __threadfence_system(); }

}  // namespace

extern "C" {

__device__ int32_t iris_gin_rank(int64_t comm) { return as_comm(comm).rank; }

__device__ int32_t iris_gin_nranks(int64_t comm) { return as_comm(comm).nRanks; }

__device__ int32_t iris_gin_lsa_rank(int64_t comm) { return as_comm(comm).lsaRank; }

__device__ int32_t iris_gin_lsa_size(int64_t comm) { return as_comm(comm).lsaSize; }

// Address of `offset` within window `win` as mapped on LSA (same-node) peer
// `lsa_peer`. Used once at setup to build the Iris peer_bases table.
__device__ int64_t iris_gin_lsa_ptr(int64_t win, int64_t offset, int32_t lsa_peer) {
  return reinterpret_cast<int64_t>(ncclGetLsaPointer(as_win(win), static_cast<size_t>(offset), lsa_peer));
}

__device__ int64_t iris_gin_local_ptr(int64_t win, int64_t offset) {
  return reinterpret_cast<int64_t>(ncclGetLocalPointer(as_win(win), static_cast<size_t>(offset)));
}

// One-sided write of `bytes` from (src_win, src_off) on this rank to
// (dst_win, dst_off) on world rank `peer`, with no completion signal.
__device__ int32_t iris_gin_put(int64_t comm, int32_t context, int32_t peer, int64_t dst_win, int64_t dst_off,
                                int64_t src_win, int64_t src_off, int64_t bytes) {
  ncclDevComm const& dc = as_comm(comm);
  ncclGin gin{dc, context};
  gin.put(ncclTeamWorld(dc), peer, as_win(dst_win), static_cast<size_t>(dst_off), as_win(src_win),
          static_cast<size_t>(src_off), static_cast<size_t>(bytes), ncclGin_None{}, ncclGin_None{}, ncclCoopCta{});
  publish_descriptor();
  return 0;
}

// As iris_gin_put, then add `value` to signal `signal` on `peer`. The weak
// signal guarantees the peer observes this put's data once it observes the
// increment -- and nothing about any other put, which is why callers send
// each payload as a single put.
__device__ int32_t iris_gin_put_signal(int64_t comm, int32_t context, int32_t peer, int64_t dst_win, int64_t dst_off,
                                       int64_t src_win, int64_t src_off, int64_t bytes, int32_t signal,
                                       int64_t value) {
  ncclDevComm const& dc = as_comm(comm);
  ncclGin gin{dc, context};
  gin.put(ncclTeamWorld(dc), peer, as_win(dst_win), static_cast<size_t>(dst_off), as_win(src_win),
          static_cast<size_t>(src_off), static_cast<size_t>(bytes),
          SignalAdd{static_cast<ncclGinSignal_t>(signal), static_cast<uint64_t>(value)}, ncclGin_None{},
          ncclCoopCta{});
  publish_descriptor();
  return 0;
}

// Wait until every put issued on `context` has been consumed by RCCL's proxy,
// i.e. its source buffer may be reused. Says nothing about remote visibility;
// that is what signals are for. Relies on publish_descriptor() in the puts.
__device__ int32_t iris_gin_flush(int64_t comm, int32_t context) {
  ncclGin gin{as_comm(comm), context};
  gin.flush(ncclCoopCta{});
  return 0;
}

// flush, giving up after `cycles` clock cycles: returns 0 once every put has
// been consumed, 1 on timeout. RCCL reports per thread; reduce so that one
// thread timing out is never hidden by another that did not. RCCL releases
// without the timed overload (2.30.x) wait without bound instead.
__device__ int32_t iris_gin_flush_timeout(int64_t comm, int32_t context, int64_t cycles) {
  ncclGin gin{as_comm(comm), context};
  return flush_bounded(gin, static_cast<uint64_t>(cycles));
}

__device__ int64_t iris_gin_read_signal(int64_t comm, int32_t context, int32_t signal) {
  ncclGin gin{as_comm(comm), context};
  return static_cast<int64_t>(gin.readSignal(static_cast<ncclGinSignal_t>(signal)));
}

// Block the program until signal `signal` on this rank reaches at least `least`,
// with acquire ordering: data carried by the puts that raised it is visible to
// every thread of the program afterwards.
__device__ int32_t iris_gin_wait_signal(int64_t comm, int32_t context, int32_t signal, int64_t least) {
  ncclGin gin{as_comm(comm), context};
  gin.waitSignal(ncclCoopCta{}, static_cast<ncclGinSignal_t>(signal), static_cast<uint64_t>(least));
  return 0;
}

}  // extern "C"
