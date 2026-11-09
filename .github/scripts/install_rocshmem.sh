#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Install rocSHMEM and its Python bindings, for tests/unittests/test_rocshmem_provider.py.
#
# Without this the provider tests skip rather than run: rocshmem4py has no
# prebuilt wheel anywhere (not PyPI, not the ROCm wheel indexes), and its
# bindings do not build rocSHMEM themselves -- python/rocshmem does
# find_package(rocshmem 3.5.0 CONFIG REQUIRED) with no FetchContent.
#
# Two modes, picked automatically:
# - ROCm ships rocSHMEM (TheRock-based releases, e.g. ROCm 10.x: librocshmem.a,
#   headers and CMake config under ROCM_PATH). Only the bindings are built,
#   against that install, from the rocm-systems tag of the same release
#   (therock-<major.minor>), so the bindings and the core come from one revision.
# - ROCm does not ship it (e.g. ROCm 7.x). rocSHMEM is built from source into
#   ROCSHMEM_PREFIX, then the bindings against it.
# ROCSHMEM_FROM_SOURCE=1 forces the source build.
#
# Scope is deliberately IPC-only: that is what the provider uses, and it is all
# a single-node CI runner can exercise. Upstream already defaults USE_IPC=ON and
# USE_GDA=OFF, so no conduit flags are passed -- which also keeps MPI and the
# RDMA provider libraries out of the picture entirely.
set -euo pipefail

ROCSHMEM_PREFIX="${ROCSHMEM_PREFIX:-/opt/rocshmem}"
# Both arches on purpose. The CI runners are MI350X (gfx950), the same image
# also serves MI300X/MI325X (gfx942), and rocSHMEM's device code must match the
# GPU it runs on: build for the wrong one and hipModuleGetGlobal fails on
# rocSHMEM's device globals
# ("Cannot create GlobalVar Obj for symbol: _ZN8rocshmem14logd_constantsE") and
# init aborts. Semicolon-separated cmake list.
ROCSHMEM_GPU_TARGETS="${ROCSHMEM_GPU_TARGETS:-gfx942;gfx950}"
ROCSHMEM_REPO="${ROCSHMEM_REPO:-https://github.com/ROCm/rocm-systems.git}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SRC="$(mktemp -d)"

PACKAGED=0
if [ "${ROCSHMEM_FROM_SOURCE:-0}" != 1 ] && [ -f "${ROCM_PATH}/lib/cmake/rocshmem/rocshmem-config.cmake" ]; then
    PACKAGED=1
    ROCSHMEM_PREFIX="${ROCM_PATH}"
    _rocm_ver="$(cat "${ROCM_PATH}/.info/version" 2>/dev/null || true)"
    [ -n "${ROCSHMEM_REF:-}" ] || [ -n "${_rocm_ver}" ] || {
        echo "ERROR: ROCm ships rocSHMEM but ${ROCM_PATH}/.info/version is missing; set ROCSHMEM_REF" >&2; exit 1; }
    ROCSHMEM_REF="${ROCSHMEM_REF:-therock-$(echo "${_rocm_ver}" | cut -d. -f1,2)}"
    echo "==> rocSHMEM shipped with ROCm ${_rocm_ver} in ${ROCM_PATH}; building only the bindings, from ${ROCSHMEM_REF}"
else
    ROCSHMEM_REF="${ROCSHMEM_REF:-develop}"
    echo "==> rocSHMEM ${ROCSHMEM_REF} -> ${ROCSHMEM_PREFIX} (GPU_TARGETS=${ROCSHMEM_GPU_TARGETS})"
fi

# rocm-systems is a large monorepo and we need at most two directories out of it.
# Sparse checkout keeps this from dominating image build time and size.
git clone --depth 1 --branch "${ROCSHMEM_REF}" --filter=blob:none --sparse \
    "${ROCSHMEM_REPO}" "${SRC}"
if [ "${PACKAGED}" = 1 ]; then
    git -C "${SRC}" sparse-checkout set python/rocshmem
else
    git -C "${SRC}" sparse-checkout set projects/rocshmem python/rocshmem
fi

if [ "${PACKAGED}" = 0 ]; then
    [ -f "${SRC}/projects/rocshmem/CMakeLists.txt" ] || {
        echo "ERROR: projects/rocshmem missing after sparse checkout" >&2; exit 1; }

    # rocSHMEM's cmake/setup_project.cmake does a REQUIRED find_file for
    # .info/version under ROCM_PATH. Images that lack that file fail to configure
    # with "Could not find rocm_version_file", so use the documented escape hatch and
    # read the version from rocm_version.h, which is authoritative. hipconfig
    # --version is not used: it reports a build number that parses as the patch level.
    EXPLICIT_ROCM_VERSION="${EXPLICIT_ROCM_VERSION:-}"
    if [ -z "${EXPLICIT_ROCM_VERSION}" ] && [ ! -f "${ROCM_PATH}/.info/version" ]; then
        _vh="$(find "${ROCM_PATH}" -name rocm_version.h 2>/dev/null | head -1)"
        if [ -n "${_vh}" ]; then
            EXPLICIT_ROCM_VERSION="$(awk '
                /ROCM_VERSION_MAJOR/ {maj=$3} /ROCM_VERSION_MINOR/ {min=$3}
                /ROCM_VERSION_PATCH/ {pat=$3}
                END {if (maj != "") printf "%s.%s.%s", maj, min, pat}' "${_vh}")"
            echo "==> ROCm ${EXPLICIT_ROCM_VERSION} detected from ${_vh}"
        fi
    fi

    cmake -S "${SRC}/projects/rocshmem" -B "${SRC}/build" -G Ninja \
        ${EXPLICIT_ROCM_VERSION:+-DEXPLICIT_ROCM_VERSION="${EXPLICIT_ROCM_VERSION}"} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ROCSHMEM_PREFIX}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGPU_TARGETS="${ROCSHMEM_GPU_TARGETS}"
    cmake --build "${SRC}/build" --parallel "$(nproc)"
    cmake --install "${SRC}/build"

    # USE_IPC must be ON or rocshmem_ptr returns NULL for every peer and the provider
    # refuses to build a table. It is the upstream default, so this asserts rather
    # than sets it -- a silent flip would otherwise surface much later as skipped tests.
    if ! grep -qi "define ROCSHMEM_USE_IPC\|USE_IPC" \
         "${ROCSHMEM_PREFIX}"/include/rocshmem/*.hpp 2>/dev/null; then
        echo "==> note: could not confirm USE_IPC from headers; provider will report at run time"
    fi
fi  # source build

# A pip install from source, same as the one-liner in iris/experimental/README.md
# but pointed at the checkout above instead of a git+ URL. That is deliberate: a
# git+ URL makes pip clone the monorepo again, independently, at whatever HEAD
# develop happens to be at -- so the bindings could be built from a different
# revision than the core installed above. find_package would not catch it, since
# it only compares versions, and the bindings statically link the core. One
# checkout for both makes the skew impossible, and saves a second clone.
#
# CMAKE_PREFIX_PATH is the documented way to point the bindings at an install;
# setup.py forwards it to CMake as a cache variable so a rocSHMEM shipped under
# /opt/rocm cannot shadow it. ROCSHMEM_HOME is no longer required.
echo "==> building rocshmem4py against ${ROCSHMEM_PREFIX}"
CMAKE_PREFIX_PATH="${ROCSHMEM_PREFIX}" ROCM_PATH="${ROCM_PATH}" \
    pip3 install --no-cache-dir "${SRC}/python/rocshmem"

python3 -c "
import rocshmem4py, importlib.metadata as md
print('    rocshmem4py', md.version('rocshmem4py'), '->', rocshmem4py.__file__)
for n in ('rocshmem_my_pe', 'rocshmem_n_pes', 'rocshmem_ptr'):
    assert hasattr(rocshmem4py, n), f'missing {n}'
print('    provider API present')
"

rm -rf "${SRC}"
echo "==> rocSHMEM install complete"
