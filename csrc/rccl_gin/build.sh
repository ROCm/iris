#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Build the two RCCL GIN artifacts iris.experimental.rccl_gin loads:
#
#   iris_gin_device_<arch>.bc   device bitcode, linked into Triton kernels
#   libiris_gin_host.so         host helper, loaded with ctypes
#
# Both must be built against the RCCL install they will run with: the device
# bitcode bakes in the ncclDevComm layout, and the host helper links librccl.
#
#   RCCL_HOME=/path/to/rccl-install ./build.sh
#
# RCCL_HOME needs a RCCL with the GIN device API (2.30 or newer): the one
# bundled with ROCm 10.1 (RCCL_HOME=$(rocm-sdk path --root)), or a build from
# ROCm/rocm-systems. ROCm 7.2's packaged RCCL 2.27 does not have it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_HOME="${RCCL_HOME:?set RCCL_HOME to a RCCL install with the GIN device API}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ARCH="${ARCH:-gfx950}"
OUT="${OUT:-${HERE}/build}"
CLANG="${ROCM_PATH}/llvm/bin/clang++"

mkdir -p "$OUT"
INCS=(-I "${RCCL_HOME}/include" -I "${RCCL_HOME}/include/rccl" -I "${ROCM_PATH}/include")

# Device-only LLVM bitcode. -fgpu-rdc keeps the extern "C" functions external:
# without it HIP internalises every function with no caller in this TU, and the
# bitcode ends up with nothing for Triton to link.
#
# DWORDX4_INTRINSICS_FORCE_OFF: by default the GIN proxy writes each queue
# descriptor with __builtin_amdgcn_global_store_b128, an intrinsic newer than
# the LLVM inside Triton 3.8, whose AMDGPU codegen aborts on it ("Unknown
# type!"). This RCCL switch selects its own fallback, non-temporal 64-bit
# stores, which any LLVM can lower -- but which are not visible to RCCL's CPU
# proxy until written back; iris_gin_device.hip's publish_descriptor() does that.
"$CLANG" -x hip --cuda-device-only -emit-llvm -std=c++17 \
  --offload-arch="$ARCH" -fgpu-rdc -fvisibility=default -O3 \
  -DDWORDX4_INTRINSICS_FORCE_OFF \
  -Xclang -mcode-object-version=none \
  "${INCS[@]}" -c "${HERE}/iris_gin_device.hip" -o "${OUT}/iris_gin_device_${ARCH}.bc"

# Triton's compile cache keys on the extern_libs path, not on the bitcode, so a
# rebuilt shim at the same path silently reuses stale kernels. Record a content
# hash next to it; the Python side folds it into the path it hands Triton.
sha256sum "${OUT}/iris_gin_device_${ARCH}.bc" | cut -c1-16 > "${OUT}/iris_gin_device_${ARCH}.bc.sha"

"$CLANG" -x hip --offload-arch="$ARCH" -std=c++17 -O2 -fPIC -shared \
  "${INCS[@]}" "${HERE}/iris_gin_host.cpp" \
  -L "${RCCL_HOME}/lib" -lrccl -Wl,-rpath,"${RCCL_HOME}/lib" \
  -o "${OUT}/libiris_gin_host.so"

# The bitcode must not define or reference device globals: a Triton module is
# loaded on its own, so nothing would initialise them.
DIS="${ROCM_PATH}/llvm/bin/llvm-dis"
if [ -x "$DIS" ]; then
  # Harmless and expected, because none carries state that a loader would
  # have to initialise: -fgpu-rdc's per-TU __hip_cuid_* byte; read-only
  # constants (the assert format string: `private` or `internal constant`,
  # depending on the compiler); and workgroup-shared (LDS, addrspace(3)) scratch,
  # which clang emits for __syncthreads_or and which is allocated per launch.
  globals=$("$DIS" -o - "${OUT}/iris_gin_device_${ARCH}.bc" | grep -E '^@[A-Za-z_]' \
    | grep -vE '^@(llvm\.|__hip_cuid_)|= private |= internal (unnamed_addr )?addrspace\(4\) constant |addrspace\(3\) global' || true)
  if [ -n "$globals" ]; then
    echo "WARNING: device bitcode has module globals:" >&2
    echo "$globals" >&2
  fi
  echo "exported:"
  "$DIS" -o - "${OUT}/iris_gin_device_${ARCH}.bc" | grep -o '^define[^@]*@iris_gin_[a-z_]*' | sed 's/.*@/  /'
fi
ls -la "$OUT"
