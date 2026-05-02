# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
IrisProcessGroup -- torch.distributed backend backed by iris CCL.

This module implements a ProcessGroup subclass that routes all collective
operations through iris's symmetric-heap CCL kernels (all_reduce,
all_gather, reduce_scatter, all_to_all).  It is registered as the "iris"
backend via ``torch.distributed.Backend.register_backend`` in the
package ``__init__``.

There are two ways to use it:

1. **On top of an existing NCCL backend** (recommended for most users)::

       import torch.distributed as dist
       dist.init_process_group(backend="nccl")  # bootstrap

       from iris.distributed.process_group import IrisProcessGroup
       pg = IrisProcessGroup.from_existing(heap_size=2**30)
       # use pg directly for collective calls

2. **As the sole backend** (``backend="iris"``)::

       import iris.distributed   # registers the backend
       import torch.distributed as dist
       dist.init_process_group(backend="iris")

   In this mode the PG bootstraps an internal NCCL group for the
   symmetric-heap address exchange, then tears it down.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from iris.host.iris import Iris
from iris.host.logging.logging import logger


# ---------------------------------------------------------------------------
# Work handle -- returned by every collective
# ---------------------------------------------------------------------------


def _ret_work(tensors):
    """Wrap finished tensors into a ``torch._C._distributed_c10d.Work`` object.

    Iris CCL calls are synchronous (they barrier internally unless
    ``async_op=True``), so the Work is always already complete by the time
    we return.
    """
    fut = torch.futures.Future()
    fut.set_result(tensors)
    return torch._C._distributed_c10d._create_work_from_future(fut)


# ---------------------------------------------------------------------------
# ReduceOp mapping
# ---------------------------------------------------------------------------

_TORCH_TO_IRIS_REDUCE_OP = {
    dist.ReduceOp.SUM: None,  # None means "use iris default (SUM)"
}


def _map_reduce_op(torch_op):
    """Convert a ``torch.distributed.ReduceOp`` to the iris equivalent.

    Currently only SUM is supported; anything else raises.
    """
    if torch_op in _TORCH_TO_IRIS_REDUCE_OP:
        return _TORCH_TO_IRIS_REDUCE_OP[torch_op]
    raise NotImplementedError(f"Iris backend only supports ReduceOp.SUM, got {torch_op}")


# ---------------------------------------------------------------------------
# ProcessGroup implementation
# ---------------------------------------------------------------------------


class IrisProcessGroup(ProcessGroup):
    """``torch.distributed.ProcessGroup`` backed by iris CCL.

    The iris context (``Iris`` instance) is lazily created on the first
    collective call because ``torch.distributed`` creates the PG before
    setting the CUDA device in some launchers.

    Parameters
    ----------
    prefix_store : ``torch.distributed.Store``
        The distributed store.  Used to bootstrap an internal NCCL PG
        when iris is the sole backend.
    rank : int
        Global rank of this process.
    world_size : int
        Total number of ranks.
    timeout : timedelta
        Operation timeout (unused by iris).
    heap_size : int, optional
        Size of the iris symmetric heap in bytes.  Defaults to 1 GiB.
        Can be overridden via the ``IRIS_HEAP_SIZE`` environment variable.
    """

    def __init__(
        self,
        prefix_store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        heap_size: int = 1 << 30,
    ):
        super().__init__(rank, world_size)
        self._prefix_store = prefix_store
        self._timeout = timeout
        self._heap_size = int(os.environ.get("IRIS_HEAP_SIZE", heap_size))

        # Lazily initialised on first use.
        self._ctx: Optional[Iris] = None

    @classmethod
    def from_existing(cls, heap_size: int = 1 << 30):
        """Create an IrisProcessGroup on top of an already-initialised PG.

        This is the preferred way to get an IrisProcessGroup when NCCL
        (or another backend) is already running.  The returned PG can be
        used directly for collective calls.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before calling "
                "IrisProcessGroup.from_existing(). Call "
                "dist.init_process_group(backend='nccl') first."
            )
        store = dist.distributed_c10d._get_default_store()
        timeout = dist.distributed_c10d._get_default_timeout()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return cls(store, rank, world_size, timeout, heap_size=heap_size)

    # -- lazy init -------------------------------------------------------

    def _ensure_ctx(self) -> Iris:
        """Create the ``Iris`` context if it does not exist yet.

        Iris internally calls ``init_distributed()`` which expects
        ``torch.distributed`` to already be initialised with a working
        backend for the heap-address allgather and barrier.

        When iris is used *on top of* an existing NCCL PG (the common
        case with ``from_existing()`` or the test harness), this just
        works -- ``Iris()`` uses the already-initialised default PG.

        When iris is used as the *sole* backend
        (``dist.init_process_group(backend='iris')``), we must bootstrap
        a temporary NCCL PG so that ``Iris()`` can perform the heap
        address exchange.  We create it here and tear it down once
        ``Iris.__init__`` completes.
        """
        if self._ctx is not None:
            return self._ctx

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[IrisProcessGroup] lazy-init rank=%d/%d heap=%s",
                self.rank(),
                self.size(),
                self._heap_size,
            )

        # Check whether the default PG can actually run NCCL collectives
        # (i.e., it is NOT us).  If the default backend is "iris" then we
        # need to bootstrap a real NCCL group for Iris.__init__.
        need_bootstrap = False
        if dist.is_initialized():
            try:
                backend_name = dist.get_backend()
                if backend_name == "iris":
                    need_bootstrap = True
            except Exception:
                need_bootstrap = True

        if need_bootstrap:
            # Create an internal NCCL process group that Iris can use for
            # the heap address allgather / barrier during init.
            # We use new_group (which all ranks must call collectively) so
            # that the bootstrap communicator is available.  We prefer
            # NCCL (GPU-native), falling back to gloo (always available).
            for _backend in ("nccl", "gloo"):
                try:
                    self._bootstrap_group = dist.new_group(
                        ranks=list(range(self.size())),
                        backend=_backend,
                    )
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    "Failed to create bootstrap process group for iris backend. "
                    "Neither NCCL nor Gloo backends are available."
                )

            # Temporarily swap the default PG so Iris.__init__ uses the
            # bootstrap group for its heap-address allgather/barrier.
            saved_default = dist.distributed_c10d._get_default_group()
            dist.distributed_c10d._set_default_group(self._bootstrap_group)
            try:
                self._ctx = Iris(heap_size=self._heap_size)
            finally:
                dist.distributed_c10d._set_default_group(saved_default)
        else:
            self._ctx = Iris(heap_size=self._heap_size)

        return self._ctx

    # -- collectives -----------------------------------------------------

    def allreduce(self, tensors, opts):
        """All-reduce *in-place*.

        ``tensors`` is a list with a single tensor (PyTorch convention).
        After the call the tensor contains the element-wise sum across
        all ranks.
        """
        ctx = self._ensure_ctx()

        # opts is an AllreduceOptions C++ object with a .reduceOp attribute
        _map_reduce_op(opts.reduceOp)  # validate (only SUM supported)

        tensor = tensors[0]
        # Ensure tensor lives on the symmetric heap.
        sym_input = ctx.zeros(tensor.shape, dtype=tensor.dtype)
        sym_input.copy_(tensor)
        sym_output = ctx.zeros(tensor.shape, dtype=tensor.dtype)

        ctx.ccl.all_reduce(sym_output, sym_input)

        tensor.copy_(sym_output)
        return _ret_work(tensors)

    def allgather(self, output_tensors_list, input_tensors, opts=None):
        """All-gather into a list of tensors.

        ``input_tensors`` is ``[tensor]`` (one per rank).
        ``output_tensors_list`` is ``[[t0, t1, ...]]`` -- one list of
        ``world_size`` tensors inside a single outer list.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()

        tensor = input_tensors[0]
        output_list = output_tensors_list[0]

        # Reshape to 2-D for iris CCL (which requires (M, N) tensors)
        if tensor.dim() == 1:
            M, N = 1, tensor.numel()
        elif tensor.dim() >= 2:
            M = tensor.shape[0]
            N = tensor.numel() // M
        else:
            # scalar
            M, N = 1, 1
        tensor_2d = tensor.reshape(M, N)

        sym_input = ctx.zeros((M, N), dtype=tensor.dtype)
        sym_input.copy_(tensor_2d)
        sym_output = ctx.zeros((world_size * M, N), dtype=tensor.dtype)

        ctx.ccl.all_gather(sym_output, sym_input)

        # Split output into per-rank chunks and copy back
        for i in range(world_size):
            chunk = sym_output[i * M : (i + 1) * M]
            output_list[i].copy_(chunk.reshape(output_list[i].shape))

        return _ret_work(output_tensors_list)

    def _allgather_base(self, output_tensor, input_tensor, opts=None):
        """``all_gather_into_tensor`` -- flat variant.

        ``output_tensor`` has shape ``(world_size * N,)`` or
        ``(world_size * M, N)``; ``input_tensor`` has shape ``(N,)`` or
        ``(M, N)``.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()

        if input_tensor.dim() == 1:
            input_2d = input_tensor.unsqueeze(0)  # (1, N)
        else:
            input_2d = input_tensor

        M, N = input_2d.shape[0], input_2d.shape[1]

        sym_input = ctx.zeros((M, N), dtype=input_tensor.dtype)
        sym_input.copy_(input_2d)
        sym_output = ctx.zeros((world_size * M, N), dtype=input_tensor.dtype)

        ctx.ccl.all_gather(sym_output, sym_input)

        output_tensor.copy_(sym_output.reshape(output_tensor.shape))
        return _ret_work([output_tensor])

    def reduce_scatter(self, output_tensors, input_tensors_list, opts):
        """Reduce-scatter.

        ``input_tensors_list`` is ``[[t0, t1, ...]]`` -- a list of
        ``world_size`` input tensors inside a single outer list.
        ``output_tensors`` is ``[tensor]``.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()

        output = output_tensors[0]
        inputs = input_tensors_list[0]

        # Stack inputs along dim-0 to get (world_size * M, N)
        stacked = torch.cat(inputs, dim=0)

        M_in = stacked.shape[0]
        N_in = stacked.numel() // M_in if stacked.dim() >= 2 else stacked.numel()

        # Iris reduce_scatter: input and output have the same shape (M, N).
        # It distributes tiles among ranks so each rank reduces its portion.
        # The PyTorch semantics are: reduce the stacked input then scatter
        # the i-th chunk to rank i.  We emulate this by doing a full
        # all_reduce on the stacked input then slicing our chunk.
        sym_input = ctx.zeros((M_in, N_in), dtype=output.dtype)
        sym_input.copy_(stacked.reshape(M_in, N_in))
        sym_output = ctx.zeros((M_in, N_in), dtype=output.dtype)

        ctx.ccl.all_reduce(sym_output, sym_input)

        # Slice our chunk
        rank = self.rank()
        chunk_size = M_in // world_size
        my_chunk = sym_output[rank * chunk_size : (rank + 1) * chunk_size]
        output.copy_(my_chunk.reshape(output.shape))

        return _ret_work(output_tensors)

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts):
        """``reduce_scatter_tensor`` -- flat variant.

        ``input_tensor`` has shape ``(world_size * N,)`` or
        ``(world_size * M, N)``; ``output_tensor`` gets the reduced
        chunk for this rank.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()
        rank = self.rank()

        if input_tensor.dim() == 1:
            input_2d = input_tensor.unsqueeze(0)
        else:
            input_2d = input_tensor

        M, N = input_2d.shape[0], input_2d.shape[1]

        sym_input = ctx.zeros((M, N), dtype=input_tensor.dtype)
        sym_input.copy_(input_2d)
        sym_output = ctx.zeros((M, N), dtype=input_tensor.dtype)

        ctx.ccl.all_reduce(sym_output, sym_input)

        chunk_size = M // world_size
        my_chunk = sym_output[rank * chunk_size : (rank + 1) * chunk_size]
        output_tensor.copy_(my_chunk.reshape(output_tensor.shape))

        return _ret_work([output_tensor])

    def alltoall(self, output_tensor_list, input_tensor_list, opts):
        """All-to-all.

        Both lists contain ``world_size`` tensors of equal shape.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()

        # Determine per-chunk shape
        sample = input_tensor_list[0]
        M = sample.shape[0] if sample.dim() >= 2 else 1
        N = sample.shape[-1] if sample.dim() >= 2 else sample.numel()

        # Concatenate inputs along columns: (M, N * world_size)
        if sample.dim() < 2:
            stacked = torch.stack(input_tensor_list, dim=0).reshape(M, N * world_size)
        else:
            stacked = torch.cat(input_tensor_list, dim=1)  # (M, N * world_size)

        sym_input = ctx.zeros((M, N * world_size), dtype=sample.dtype)
        sym_input.copy_(stacked)
        sym_output = ctx.zeros((M, N * world_size), dtype=sample.dtype)

        ctx.ccl.all_to_all(sym_output, sym_input)

        # Copy results back to output list
        for i in range(world_size):
            chunk = sym_output[:, i * N : (i + 1) * N]
            output_tensor_list[i].copy_(chunk.reshape(output_tensor_list[i].shape))

        return _ret_work(output_tensor_list)

    def alltoall_base(self, output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts):
        """All-to-all with a single tensor (equal splits only).

        ``input_tensor`` and ``output_tensor`` have shape ``(world_size * N,)``
        or ``(M, world_size * N)``.  Non-uniform splits are not supported.
        """
        ctx = self._ensure_ctx()
        world_size = self.size()

        if output_split_sizes or input_split_sizes:
            raise NotImplementedError("Iris backend does not support non-uniform split sizes in alltoall_base")

        if input_tensor.dim() == 1:
            N_total = input_tensor.numel()
            input_2d = input_tensor.reshape(1, N_total)
        else:
            input_2d = input_tensor

        M = input_2d.shape[0]
        N_total = input_2d.shape[1]

        sym_input = ctx.zeros((M, N_total), dtype=input_tensor.dtype)
        sym_input.copy_(input_2d)
        sym_output = ctx.zeros((M, N_total), dtype=input_tensor.dtype)

        ctx.ccl.all_to_all(sym_output, sym_input)

        output_tensor.copy_(sym_output.reshape(output_tensor.shape))
        return _ret_work([output_tensor])

    def barrier(self, opts=None):
        """Barrier across all ranks."""
        ctx = self._ensure_ctx()
        ctx.barrier()
        return _ret_work([])

    def broadcast(self, tensors, opts):
        """Broadcast from ``opts.rootRank``.

        Falls back to an all-reduce-based implementation since iris does
        not have a dedicated broadcast primitive.
        """
        ctx = self._ensure_ctx()
        tensor = tensors[0]
        root = opts.rootRank

        # Zero out non-root contributions then all-reduce to broadcast
        sym_input = ctx.zeros(tensor.shape, dtype=tensor.dtype)
        if self.rank() == root:
            sym_input.copy_(tensor)

        sym_output = ctx.zeros(tensor.shape, dtype=tensor.dtype)
        ctx.ccl.all_reduce(sym_output, sym_input)

        tensor.copy_(sym_output)
        return _ret_work(tensors)

    # -- informational ---------------------------------------------------

    def getBackendName(self):
        return "iris"


# ---------------------------------------------------------------------------
# Factory function used by register_backend
# ---------------------------------------------------------------------------


def _create_iris_backend(prefix_store, rank, world_size, timeout):
    """Factory called by ``torch.distributed.init_process_group(backend='iris')``."""
    return IrisProcessGroup(prefix_store, rank, world_size, timeout)
