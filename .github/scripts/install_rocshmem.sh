#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Install rocSHMEM and its Python bindings, for tests/unittests/test_rocshmem_provider.py.
#
# Without this the provider tests skip rather than run: rocshmem4py has no
# prebuilt wheel anywhere (not PyPI, not the ROCm wheel indexes), and its
# bindings do not build rocSHMEM themselves -- python/rocshmem does
# find_package(rocshmem 3.5.0 CONFIG REQUIRED) with no FetchContent. So rocSHMEM
# has to be built first, and then the bindings against it.
#
# Scope is deliberately IPC-only: that is what the provider uses, and it is all
# a single-node CI runner can exercise. Upstream already defaults USE_IPC=ON and
# USE_GDA=OFF, so no conduit flags are passed -- which also keeps MPI and the
# RDMA provider libraries out of the picture entirely.
set -euo pipefail

ROCSHMEM_PREFIX="${ROCSHMEM_PREFIX:-/opt/rocshmem}"
# MI325X runners are gfx942. Semicolon-separated for more than one.
ROCSHMEM_GPU_TARGETS="${ROCSHMEM_GPU_TARGETS:-gfx942}"
ROCSHMEM_REPO="${ROCSHMEM_REPO:-https://github.com/ROCm/rocm-systems.git}"
ROCSHMEM_REF="${ROCSHMEM_REF:-develop}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SRC="$(mktemp -d)"

echo "==> rocSHMEM ${ROCSHMEM_REF} -> ${ROCSHMEM_PREFIX} (GPU_TARGETS=${ROCSHMEM_GPU_TARGETS})"

# rocm-systems is a large monorepo and we need two directories out of it. Sparse
# checkout keeps this from dominating image build time and size.
git clone --depth 1 --branch "${ROCSHMEM_REF}" --filter=blob:none --sparse \
    "${ROCSHMEM_REPO}" "${SRC}"
git -C "${SRC}" sparse-checkout set projects/rocshmem python/rocshmem

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
