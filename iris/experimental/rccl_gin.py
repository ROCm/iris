# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""RCCL GIN as an allocation provider and inter-node transport for Iris.

Two halves, one RCCL communicator:

- **Intra-node** stays plain Iris. Memory comes from a RCCL symmetric window
  (``ncclMemAlloc`` + ``ncclCommWindowRegister``), and RCCL maps that window on
  every same-node (LSA) peer. ``allocate_window`` returns a
  ``SymmetricAddressMap`` whose ``peer_bases`` holds those mappings, so
  unmodified ``iris.store``/``load``/``atomic_*`` reach same-node peers.
- **Inter-node** peers are ``direct=False`` in that map: their entry is 0 and
  must never be translated. The Triton functions below (``put``,
  ``put_signal``, ``wait_signal``, ...) reach them through RCCL's GPU-Initiated
  Networking (GIN) device API instead, on the same window.

The device functions are RCCL's GIN templates compiled to LLVM bitcode by
``csrc/rccl_gin/build.sh`` and linked into Triton kernels. A kernel that calls
any of them must be launched with ``extern_libs=rccl_gin.extern_libs()``.

Every device function here is a *program-wide* operation: all threads of the
program must reach it, in program-uniform control flow. See the header of
``csrc/rccl_gin/iris_gin_device.hip`` for why.

Requires a RCCL with the GIN device API on ROCm with HIP >= 7.12: the RCCL
bundled with ROCm 10.1, or a newer build. See
``examples/35_wide_ep_rccl_gin/README.md`` for the exact requirements.
"""

from __future__ import annotations

import ctypes
import logging
import os
import resource
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from iris.experimental.symmetric_memory import SymmetricAddressMap

__all__ = [
    "RcclGinProvider",
    "RcclGinUnavailable",
    "GinWindow",
    "DeviceComm",
    "extern_libs",
    "put",
    "put_signal",
    "flush",
    "flush_timeout",
    "read_signal",
    "wait_signal",
]

# ncclGinType_t values RCCL reports from ncclCommQueryProperties.
GIN_TYPE_NAMES = {0: "none", 2: "proxy", 4: "gpi", 6: "rocshmem_gda", 7: "anvil_sdma"}

_MIN_RCCL_VERSION = 23000  # first RCCL with the GIN device API

# With cuMem, RCCL shares every mapped peer buffer as a file descriptor over a
# Unix socket, and each communicator in the process does so separately (torch's
# own process group is one). Eight ranks with two communicators already exceed
# the 1024 soft limit containers start with, and RCCL then reports only
# "unhandled system error" from ncclCommWindowRegister on some ranks while the
# rest wait in the collective. IRIS_RCCL_GIN_MIN_OPEN_FILES lowers the bar for
# small jobs that are known to fit.
_DEFAULT_MIN_OPEN_FILES = 65536

_log = logging.getLogger("iris")


def _open_file_limit_problem() -> str | None:
    """Raise this process's soft open-file limit toward the target; report a
    problem if the hard limit keeps it below. Unprivileged and process-local:
    only the soft limit moves, and never past the hard limit."""
    need = int(os.environ.get("IRIS_RCCL_GIN_MIN_OPEN_FILES", _DEFAULT_MIN_OPEN_FILES))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = resource.RLIM_INFINITY
    if soft != inf and soft < need:
        target = need if hard == inf else min(hard, need)
        if target > soft:
            # A failure here is reported, not raised, so that _agree() fails every rank together.
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            except (ValueError, OSError) as e:
                return f"could not raise the open-file soft limit from {soft} to {target} for RCCL cuMem sharing: {e}"
            _log.info(f"rccl_gin: raised the open-file soft limit from {soft} to {target} for RCCL cuMem sharing")
            soft = target
    if soft != inf and soft < need:
        return (
            f"open-file limit too low for RCCL cuMem sharing: soft {soft}, hard {hard}, need {need}. Raise the "
            f"hard limit, e.g. docker run --ulimit nofile={need}:{need} or ulimit -Hn {need}; for a small job "
            f"known to fit, set IRIS_RCCL_GIN_MIN_OPEN_FILES lower."
        )
    return None


def _agree(group, problem: str | None) -> None:
    """Collective: raise RcclGinUnavailable on every rank if any rank has a
    problem. Checks that can fail on some ranks only (a shim missing on one
    node, a different librccl, a low limit) must agree before RCCL's own
    collectives, or the ranks that pass wait forever for those that raised."""
    problems = [None] * dist.get_world_size(group)
    dist.all_gather_object(problems, problem, group=group)
    failed = [(r, p) for r, p in enumerate(problems) if p]
    if failed:
        detail = "; ".join(f"rank {r}: {p}" for r, p in failed[:4])
        more = f" (and {len(failed) - 4} more ranks)" if len(failed) > 4 else ""
        raise RcclGinUnavailable(f"{len(failed)} of {len(problems)} ranks cannot use RCCL GIN -- {detail}{more}")


class RcclGinUnavailable(RuntimeError):
    """A prerequisite of this provider is missing from the environment: the
    shim is not built, the loaded RCCL predates GIN, or no GIN backend came up.
    Distinct from other errors so callers (and tests) can fall back or skip on
    exactly this, without masking real failures."""


# ── locating the build ──────────────────────────────────────────────────────


def _build_dir() -> Path:
    env = os.environ.get("IRIS_RCCL_GIN_BUILD")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "csrc" / "rccl_gin" / "build"


def _arch() -> str:
    """The current GPU's architecture (e.g. gfx950), which names the bitcode.
    IRIS_RCCL_GIN_ARCH overrides it."""
    env = os.environ.get("IRIS_RCCL_GIN_ARCH")
    if env:
        return env
    from iris.host.platform.hip import get_arch_string

    return get_arch_string(torch.cuda.current_device())


_bitcode_cache: str | None = None


def _bitcode_path() -> str:
    """Path of the device bitcode, made unique per build.

    Triton's compile cache keys on the extern_libs path, not on the bitcode's
    contents, so rebuilding the shim in place would silently keep running the
    old one. build.sh writes a content hash beside the bitcode; we hand Triton a
    copy whose path contains it.
    """
    global _bitcode_cache
    if _bitcode_cache is None:
        src = _build_dir() / f"iris_gin_device_{_arch()}.bc"
        if not src.exists():
            raise RcclGinUnavailable(f"{src} not found; build it with RCCL_HOME=<rccl> csrc/rccl_gin/build.sh")
        sha_file = src.with_name(src.name + ".sha")
        sha = sha_file.read_text().strip() if sha_file.exists() else str(int(src.stat().st_mtime))
        cache = Path(os.environ.get("IRIS_RCCL_GIN_CACHE", Path.home() / ".cache" / "iris" / "rccl_gin")) / sha
        dst = cache / src.name
        if not dst.exists():
            cache.mkdir(parents=True, exist_ok=True)
            # A unique temp name: ranks on different nodes sharing this cache
            # over NFS can have the same PID (identical containers do).
            fd, tmp = tempfile.mkstemp(dir=cache, prefix=src.name + ".")
            os.close(fd)
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)  # atomic, so concurrent ranks never see a partial file
        _bitcode_cache = str(dst)
    return _bitcode_cache


def extern_libs() -> dict[str, str]:
    """Launch kwarg for kernels that call this module's device functions.

    The key is not a label: Triton's AMD backend links a library only if some
    undefined symbol in the kernel starts with it, and silently drops it
    otherwise (the kernel then fails to load with HIP error 209). Every symbol
    here starts with ``iris_gin``.
    """
    return {"iris_gin": _bitcode_path()}


# ── device functions ────────────────────────────────────────────────────────


def _extern(fn_name: str, arg_types: list, ret_type):
    @tl.core.extern
    def _call(*args, _semantic=None):
        return tl.core.extern_elementwise(
            "iris_gin",
            _bitcode_path(),
            list(args),
            {tuple(arg_types): (fn_name, ret_type)},
            is_pure=False,
            _semantic=_semantic,
        )

    return _call


_i32, _i64 = tl.int32, tl.int64
_lsa_ptr = _extern("iris_gin_lsa_ptr", [_i64, _i64, _i32], _i64)
_put = _extern("iris_gin_put", [_i64, _i32, _i32, _i64, _i64, _i64, _i64, _i64], _i32)
_put_signal = _extern("iris_gin_put_signal", [_i64, _i32, _i32, _i64, _i64, _i64, _i64, _i64, _i32, _i64], _i32)
_flush = _extern("iris_gin_flush", [_i64, _i32], _i32)
_flush_timeout = _extern("iris_gin_flush_timeout", [_i64, _i32, _i64], _i32)
_read_signal = _extern("iris_gin_read_signal", [_i64, _i32, _i32], _i64)
_wait_signal = _extern("iris_gin_wait_signal", [_i64, _i32, _i32, _i64], _i32)


@triton.jit
def put(comm, context, peer, dst_win, dst_offset, src_win, src_offset, nbytes):
    """Write ``nbytes`` from ``src_win+src_offset`` here to ``dst_win+dst_offset``
    on world rank ``peer``. No completion signal; use ``put_signal`` for that."""
    _put(
        tl.cast(comm, tl.int64),
        tl.cast(context, tl.int32),
        tl.cast(peer, tl.int32),
        tl.cast(dst_win, tl.int64),
        tl.cast(dst_offset, tl.int64),
        tl.cast(src_win, tl.int64),
        tl.cast(src_offset, tl.int64),
        tl.cast(nbytes, tl.int64),
    )


@triton.jit
def put_signal(comm, context, peer, dst_win, dst_offset, src_win, src_offset, nbytes, signal, value):
    """``put``, then add ``value`` to ``signal`` on ``peer``.

    The signal is weak: observing it guarantees this put's data is visible, and
    nothing about any other put. Send each payload as a single put_signal."""
    _put_signal(
        tl.cast(comm, tl.int64),
        tl.cast(context, tl.int32),
        tl.cast(peer, tl.int32),
        tl.cast(dst_win, tl.int64),
        tl.cast(dst_offset, tl.int64),
        tl.cast(src_win, tl.int64),
        tl.cast(src_offset, tl.int64),
        tl.cast(nbytes, tl.int64),
        tl.cast(signal, tl.int32),
        tl.cast(value, tl.int64),
    )


@triton.jit
def flush(comm, context):
    """Wait until every put on ``context`` has been consumed by RCCL's proxy,
    so its source buffer may be reused. Remote visibility is the signal's job.

    Program-wide, like every function here; waits without bound. Prefer
    ``flush_timeout`` where a lost put should fail rather than hang."""
    _flush(tl.cast(comm, tl.int64), tl.cast(context, tl.int32))


@triton.jit
def flush_timeout(comm, context, cycles):
    """``flush`` that gives up after ``cycles`` GPU clock cycles. Returns 0 on
    success and 1 on timeout, the same on every thread of the program. With an
    RCCL that has no timed flush (2.30.x) it waits without bound and returns 0."""
    return _flush_timeout(tl.cast(comm, tl.int64), tl.cast(context, tl.int32), tl.cast(cycles, tl.int64))


@triton.jit
def read_signal(comm, context, signal):
    return _read_signal(tl.cast(comm, tl.int64), tl.cast(context, tl.int32), tl.cast(signal, tl.int32))


@triton.jit
def wait_signal(comm, context, signal, least):
    """Block until local ``signal`` >= ``least``; acquire for the whole program."""
    _wait_signal(
        tl.cast(comm, tl.int64), tl.cast(context, tl.int32), tl.cast(signal, tl.int32), tl.cast(least, tl.int64)
    )


@triton.jit
def _lsa_bases_kernel(win, out, lsa_start, lsa_size):
    for p in range(lsa_size):
        tl.store(out + lsa_start + p, _lsa_ptr(tl.cast(win, tl.int64), tl.cast(0, tl.int64), tl.cast(p, tl.int32)))


@triton.jit
def _read_signals_kernel(comm, context, out, num_signals):
    for s in range(num_signals):
        tl.store(out + s, read_signal(comm, context, s))


# ── host side ───────────────────────────────────────────────────────────────


def _load_host_lib() -> ctypes.CDLL:
    path = _build_dir() / "libiris_gin_host.so"
    if not path.exists():
        raise RcclGinUnavailable(f"{path} not found; build it with RCCL_HOME=<rccl> csrc/rccl_gin/build.sh")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        # torch links librccl.so.1 too, and a library already loaded under that
        # soname wins over this helper's rpath. If it is ROCm's packaged RCCL,
        # the GIN symbols are missing and dlopen fails here.
        raise RcclGinUnavailable(
            f"{e}\nThe RCCL loaded in this process lacks the GIN device API. Replace the librccl "
            f"torch loads with a GIN-capable build; see examples/35_wide_ep_rccl_gin/README.md."
        ) from e
    vp = ctypes.c_void_p
    ip = ctypes.POINTER(ctypes.c_int)
    vpp = ctypes.POINTER(ctypes.c_void_p)
    sigs = {
        "iris_gin_error_string": ([ctypes.c_int], ctypes.c_char_p),
        "iris_gin_version": ([ip], ctypes.c_int),
        "iris_gin_header_version": ([], ctypes.c_int),
        "iris_gin_unique_id_bytes": ([], ctypes.c_int),
        "iris_gin_get_unique_id": ([ctypes.c_char_p], ctypes.c_int),
        "iris_gin_comm_init": ([vpp, ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
        "iris_gin_comm_destroy": ([vp], ctypes.c_int),
        "iris_gin_comm_query": ([vp, ip, ip, ip], ctypes.c_int),
        "iris_gin_window_alloc": ([vp, ctypes.c_size_t, vpp, vpp], ctypes.c_int),
        "iris_gin_window_free": ([vp, vp, vp], ctypes.c_int),
        "iris_gin_devcomm_create": (
            [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, vpp, vpp, ip, ip, ip, ip, ip],
            ctypes.c_int,
        ),
        "iris_gin_devcomm_destroy": ([vp], ctypes.c_int),
    }
    for name, (argtypes, restype) in sigs.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as e:
            raise RcclGinUnavailable(
                f"{path} has no {name}: it was built from older Iris sources. Rebuild it with "
                f"RCCL_HOME=<rccl> csrc/rccl_gin/build.sh"
            ) from e
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


@dataclass(frozen=True)
class GinWindow:
    """One RCCL symmetric window. ``handle`` is what device functions take as a
    window argument; offsets into it are byte offsets from ``tensor``'s start."""

    tensor: torch.Tensor  # uint8, the whole window
    handle: int  # ncclWindow_t, a device pointer
    address_map: SymmetricAddressMap


@dataclass(frozen=True)
class DeviceComm:
    """An ncclDevComm in device memory; ``ptr`` is the kernel argument."""

    ptr: int
    context_count: int
    signal_count: int
    rank: int
    world_size: int
    lsa_rank: int
    lsa_size: int
    _handle: int = field(default=0, repr=False)  # host-side handle, for destroy_device_comm

    @property
    def lsa_start(self) -> int:
        """First world rank of this rank's LSA team. RCCL's LSA teams are
        contiguous world-rank ranges, as its own hybrid examples assume."""
        return self.rank - self.lsa_rank


class RcclGinProvider:
    """Owns a RCCL communicator, its symmetric windows and its device comm.

    ``torch.distributed`` must already be initialised; it is used only to
    broadcast the RCCL unique id, so a gloo process group is enough. Each rank
    must have selected its GPU (``torch.cuda.set_device``) beforehand.
    """

    def __init__(self, group: dist.ProcessGroup | None = None):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialised before constructing RcclGinProvider")
        self._group = group if group is not None else dist.group.WORLD
        self.cur_rank = dist.get_rank(self._group)
        self.num_ranks = dist.get_world_size(self._group)
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._comm = None
        self._lib = None
        problem = None
        try:
            self._lib = _load_host_lib()
            version = ctypes.c_int()
            self._check(self._lib.iris_gin_version(ctypes.byref(version)), "ncclGetVersion")
            self.rccl_version = version.value
            headers = self._lib.iris_gin_header_version()
            if self.rccl_version < _MIN_RCCL_VERSION:
                problem = (
                    f"RCCL {self.rccl_version} has no GIN device API (need >= {_MIN_RCCL_VERSION}). "
                    f"Is ROCm's packaged librccl loaded instead of the GIN build?"
                )
            elif headers != self.rccl_version:
                problem = (
                    f"the shim was built against RCCL {headers} headers but RCCL {self.rccl_version} is loaded; "
                    f"rebuild csrc/rccl_gin against the RCCL that runs (its ncclDevComm layout is compiled in)"
                )
        except RcclGinUnavailable as exc:
            problem = str(exc)
        problem = problem or _open_file_limit_problem()
        _agree(self._group, problem)

        # Rank 0 broadcasts the unique id, or the reason it has none, so no
        # rank is left waiting on a broadcast that never comes.
        nbytes = self._lib.iris_gin_unique_id_bytes()
        obj = [None]
        if self.cur_rank == 0:
            uid = ctypes.create_string_buffer(nbytes)
            r = self._lib.iris_gin_get_unique_id(uid)
            obj = [uid.raw if r == 0 else f"ncclGetUniqueId failed: {self._lib.iris_gin_error_string(r).decode()}"]
        src = dist.get_global_rank(self._group, 0) if self._group is not dist.group.WORLD else 0
        dist.broadcast_object_list(obj, src=src, group=self._group)
        if isinstance(obj[0], str):
            raise RuntimeError(f"rank 0: {obj[0]}")
        uid = ctypes.create_string_buffer(obj[0], nbytes)

        comm = ctypes.c_void_p()
        self._check(
            self._lib.iris_gin_comm_init(ctypes.byref(comm), self.num_ranks, uid, self.cur_rank),
            "ncclCommInitRank",
        )
        self._comm = comm

        device_api, gin_type, n_lsa = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._check(
            self._lib.iris_gin_comm_query(
                self._comm, ctypes.byref(device_api), ctypes.byref(gin_type), ctypes.byref(n_lsa)
            ),
            "ncclCommQueryProperties",
        )
        if not device_api.value:
            # Every rank sees the same properties, so all of them get here;
            # release the communicator collectively before reporting.
            self._check(self._lib.iris_gin_comm_destroy(self._comm), "ncclCommDestroy")
            self._comm = None
            raise RcclGinUnavailable(
                "RCCL reports no device API support (symmetric windows) on this communicator. RCCL "
                "needs all of: cuMem (HIP >= 7.12, NCCL_CUMEM_ENABLE=1), peer-to-peer access between "
                "all local GPUs, windows enabled (NCCL_WIN_ENABLE), and either a GIN backend or every "
                "rank in one load/store team. Ranks spanning nodes therefore need GIN: NCCL_GIN_TYPE=2, "
                "NCCL_DMABUF_ENABLE=1 and a working RDMA NIC. NCCL_DEBUG=INFO "
                "NCCL_DEBUG_SUBSYS=INIT,NET says which condition failed."
            )
        self.gin_type = gin_type.value
        self.num_lsa_teams = n_lsa.value

        self._windows: list[tuple[int, int]] = []  # (ptr, handle)
        self._devcomms: list[int] = []  # host-side handles
        # LSA team geometry, from the communicator's properties rather than a
        # throwaway device comm: RCCL builds equal-sized teams of consecutive
        # world ranks (computeLsaSize), so nLsaTeams fixes them. A probe device
        # comm with zero GIN contexts used to supply this, but RCCL before 2.32
        # corrupts its heap creating one when a GIN backend is up
        # (ROCm/rocm-systems#14178). Every real device comm checks the result.
        if self.num_lsa_teams < 1 or self.num_ranks % self.num_lsa_teams:
            raise RuntimeError(f"RCCL reports {self.num_lsa_teams} LSA teams for {self.num_ranks} ranks")
        self.lsa_size = self.num_ranks // self.num_lsa_teams
        self.lsa_rank = self.cur_rank % self.lsa_size
        self.lsa_start = self.cur_rank - self.lsa_rank

        # Without a GIN backend (e.g. no RDMA NIC) the provider still works as
        # long as every peer is load/store reachable: all of them are direct,
        # and nothing ever needs GIN. Only a peer outside RCCL's load/store team
        # makes GIN mandatory. Current RCCL already refuses the device API in
        # that case (reported above); this keeps the guarantee if it stops
        # doing so. lsa_size is the same on every rank, so all ranks decide alike.
        if not self.gin_available and self.lsa_size < self.num_ranks:
            self.close()
            raise RcclGinUnavailable(
                f"RCCL reports no GIN backend, and {self.num_ranks - self.lsa_size} of {self.num_ranks} ranks are "
                f"outside this rank's load/store team, so they can only be reached through GIN. The IB proxy "
                f"backend needs NCCL_GIN_TYPE=2, NCCL_CUMEM_ENABLE=1, NCCL_DMABUF_ENABLE=1 and a working RDMA "
                f"NIC; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET says which prerequisite failed."
            )
        if not self.gin_available:
            _log.info("rccl_gin: no GIN backend; every peer is load/store reachable, so none is needed")

    def _check(self, result: int, what: str):
        if result != 0:
            msg = self._lib.iris_gin_error_string(result).decode()
            raise RuntimeError(f"{what} failed on rank {self.cur_rank}: {msg} ({result})")

    @property
    def gin_available(self) -> bool:
        """Whether RCCL brought up a GIN backend. Without one, every peer is
        direct and the GIN device functions must not be called."""
        return self.gin_type != 0

    @property
    def gin_type_name(self) -> str:
        return GIN_TYPE_NAMES.get(self.gin_type, str(self.gin_type))

    def is_lsa_peer(self, rank: int) -> bool:
        return self.lsa_start <= rank < self.lsa_start + self.lsa_size

    # ── memory ──────────────────────────────────────────────────────────────

    def allocate_window(self, nbytes: int) -> GinWindow:
        """Collectively allocate and register a symmetric window of ``nbytes``.

        Same-node peers are ``direct`` in the returned map; everyone else must be
        reached with the GIN device functions.
        """
        ptr, win = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(
            self._lib.iris_gin_window_alloc(self._comm, nbytes, ctypes.byref(ptr), ctypes.byref(win)),
            "ncclMemAlloc/ncclCommWindowRegister",
        )
        self._windows.append((ptr.value, win.value))

        from iris.host.memory.tensor_utils import tensor_from_ptr

        tensor = tensor_from_ptr(ptr.value, nbytes, torch.uint8, str(self._device))

        peer_bases = torch.zeros(self.num_ranks, dtype=torch.int64, device=self._device)
        _lsa_bases_kernel[(1,)](
            win.value, peer_bases, self.lsa_start, self.lsa_size, num_warps=1, extern_libs=extern_libs()
        )
        torch.cuda.synchronize()
        # The window's LSA mapping of *this* rank may be a second virtual alias
        # of the same memory. Iris translates relative to the local entry, and
        # the pointers a kernel holds are the ncclMemAlloc ones, so the local
        # entry must be that address.
        peer_bases[self.cur_rank] = ptr.value
        direct = tuple(self.is_lsa_peer(r) for r in range(self.num_ranks))
        bases = peer_bases.tolist()
        bad = [r for r in range(self.num_ranks) if direct[r] and bases[r] == 0]
        if bad:
            raise RuntimeError(f"RCCL returned no LSA mapping for same-node peers {bad}")

        amap = SymmetricAddressMap(
            peer_bases=peer_bases,
            local_rank=self.cur_rank,
            allocation_base=ptr.value,
            allocation_bytes=nbytes,
            direct=direct,
        )
        return GinWindow(tensor=tensor, handle=win.value, address_map=amap)

    # ── device comm ─────────────────────────────────────────────────────────

    def _create_devcomm(self, context_count: int, signal_count: int) -> DeviceComm:
        handle, dev = ctypes.c_void_p(), ctypes.c_void_p()
        ctx, rank, nranks, lsa_rank, lsa_size = (ctypes.c_int() for _ in range(5))
        self._check(
            self._lib.iris_gin_devcomm_create(
                self._comm,
                context_count,
                signal_count,
                0,
                ctypes.byref(handle),
                ctypes.byref(dev),
                ctypes.byref(ctx),
                ctypes.byref(rank),
                ctypes.byref(nranks),
                ctypes.byref(lsa_rank),
                ctypes.byref(lsa_size),
            ),
            "ncclDevCommCreate",
        )
        self._devcomms.append(handle.value)
        if (lsa_rank.value, lsa_size.value) != (self.lsa_rank, self.lsa_size):
            raise RuntimeError(
                f"rank {self.cur_rank}: RCCL's device comm reports LSA rank {lsa_rank.value} of "
                f"{lsa_size.value}, but its properties implied {self.lsa_rank} of {self.lsa_size}"
            )
        return DeviceComm(
            _handle=handle.value,
            ptr=dev.value,
            context_count=ctx.value,
            signal_count=signal_count,
            rank=rank.value,
            world_size=nranks.value,
            lsa_rank=lsa_rank.value,
            lsa_size=lsa_size.value,
        )

    def device_comm(self, signal_count: int, context_count: int = 1) -> DeviceComm:
        """Collectively create a device communicator with full GIN connectivity
        and ``signal_count`` signals (ids ``0..signal_count-1``).

        Without a GIN backend (``gin_available`` is False) it has no GIN
        resources, and the GIN device functions must not be called on it;
        every peer is direct then, so a kernel routing on ``direct`` never does.
        """
        if not self.gin_available:
            return self._create_devcomm(context_count=0, signal_count=0)
        dc = self._create_devcomm(context_count=context_count, signal_count=signal_count)
        if dc.context_count < 1:
            raise RuntimeError("RCCL granted no GIN contexts")
        return dc

    def read_signals(self, dc: DeviceComm, context: int = 0, count: int | None = None) -> torch.Tensor:
        """Current values of ``dc``'s signals on this rank, as int64[count]
        (default: all of them). Zeros when ``dc`` has no GIN resources.

        Raises ValueError if ``count`` is negative, or larger than
        ``dc.signal_count`` on a device comm that has signals."""
        n = dc.signal_count if count is None else count
        if n < 0 or (dc.signal_count and n > dc.signal_count):
            raise ValueError(f"count {n} out of range: the device comm has {dc.signal_count} signals")
        out = torch.zeros(n, dtype=torch.int64, device=self._device)
        if dc.signal_count == 0:
            return out
        _read_signals_kernel[(1,)](dc.ptr, context, out, n, num_warps=1, extern_libs=extern_libs())
        return out

    def barrier(self):
        torch.cuda.synchronize()
        dist.barrier(group=self._group)

    def free_window(self, window: GinWindow):
        """Collectively deregister and free one window from allocate_window.
        No kernel may still be using it on any rank."""
        entry = next((e for e in self._windows if e[1] == window.handle), None)
        if entry is None:
            raise ValueError("window was not allocated by this provider, or is already freed")
        torch.cuda.synchronize()
        self._check(self._lib.iris_gin_window_free(self._comm, entry[0], entry[1]), "ncclCommWindowDeregister")
        self._windows.remove(entry)

    def destroy_device_comm(self, dc: DeviceComm):
        """Collectively destroy one device communicator from device_comm."""
        if dc._handle not in self._devcomms:
            raise ValueError("device comm was not created by this provider, or is already destroyed")
        torch.cuda.synchronize()
        self._check(self._lib.iris_gin_devcomm_destroy(ctypes.c_void_p(dc._handle)), "ncclDevCommDestroy")
        self._devcomms.remove(dc._handle)

    def close(self):
        """Release everything, in reverse order of creation. Collective."""
        if self._comm is None:
            return
        torch.cuda.synchronize()
        for h in reversed(self._devcomms):
            self._check(self._lib.iris_gin_devcomm_destroy(ctypes.c_void_p(h)), "ncclDevCommDestroy")
        self._devcomms.clear()
        for ptr, win in reversed(self._windows):
            self._check(self._lib.iris_gin_window_free(self._comm, ptr, win), "ncclCommWindowDeregister")
        self._windows.clear()
        self._check(self._lib.iris_gin_comm_destroy(self._comm), "ncclCommDestroy")
        self._comm = None

    def get_rank(self) -> int:
        return self.cur_rank

    def get_num_ranks(self) -> int:
        return self.num_ranks
