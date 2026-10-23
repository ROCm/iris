# RCCL GIN and Wide Expert Parallelism

Iris's load/store model reaches every GPU whose memory it can map: the GPUs of
one node, and, with the fabric drivers, every host in one scale-up fabric
domain such as a Helios or NVL72 rack. GPUs beyond a fabric domain are
reachable only over RDMA NICs. RCCL's GPU-Initiated Networking (GIN) covers
that tier: a kernel asks for a one-sided write to a peer in another fabric
domain, and the network carries it. `iris.experimental.rccl_gin` brings the
two together on one buffer, and `iris.experimental.wide_ep` builds MoE dispatch
and combine across fabric domains on it.

```
                     one RCCL symmetric window (same offsets on every rank)
                                          │
  same fabric domain (direct=True)   ──► iris.store / iris.atomic_*   (load/store)
  other fabric domains (direct=False) ──► rccl_gin.put_signal         (GIN, over the NIC)
```

On a system of standalone nodes, such as MI355X nodes linked by RDMA NICs, a
fabric domain is one node.

Both modules are experimental: they may change between releases without
deprecation. A runnable example, the full environment requirements and
measured results are in
[examples/35_wide_ep_rccl_gin](https://github.com/ROCm/iris/tree/main/examples/35_wide_ep_rccl_gin).

## Model

- **Window.** `RcclGinProvider.allocate_window(nbytes)` allocates with
  `ncclMemAlloc` and registers the buffer as a RCCL symmetric window,
  collectively. RCCL maps the window on every peer in its load/store team and
  registers it with the NIC for the others. The returned `GinWindow` has the local tensor,
  the window handle that GIN calls take, and a `SymmetricAddressMap`.
- **Address map.** The map follows the same per-allocation contract as the
  other experimental providers. Peers in RCCL's load/store (LSA) team are
  `direct`, and their `peer_bases` entry is RCCL's mapping of the window, so
  unmodified Iris device code reaches them. RCCL sizes that team to what it can
  map: the node, or the fabric domain on rack-scale systems. All other peers are
  `direct=False` with a base of 0: never translate against them.
- **Device communicator.** `RcclGinProvider.device_comm(signal_count)` creates
  a RCCL device communicator with full GIN connectivity. Its `ptr` is the
  kernel argument the GIN functions take. Signals are per-source counters on
  the receiving rank, and they only count up.
- **Device functions.** `put`, `put_signal`, `flush`, `flush_timeout`,
  `read_signal` and `wait_signal` are RCCL's GIN device templates, compiled to
  bitcode by `csrc/rccl_gin/build.sh` and linked into the Triton kernel. Launch
  such kernels with `extern_libs=rccl_gin.extern_libs()`. Every one of them is
  a program-wide operation: all threads of the program must reach it, in
  program-uniform control flow.

## Routing

A kernel picks the transport per destination from the map's `direct` mask:

```python
@triton.jit
def send(win_ptr, heap_bases, comm, win, direct, rank, dst, off, nbytes, ...):
    if tl.load(direct + dst) != 0:
        iris.store(...)                                  # load/store reachable
    else:
        rccl_gin.put_signal(comm, 0, dst, win, off, win, off, nbytes, rank, 1)
```

Route on the mask itself rather than on a rank range. Then routing and the map
cannot disagree about which peers are addressable. RCCL builds its load/store
(LSA) team from consecutive world ranks, so launch with node-major ranks, as
torchrun does by default. With interleaved ranks, RCCL shrinks the team, and
peers it could have mapped fall back to GIN: correct, but slower.

## Memory ordering

**Remote visibility is the signal's job.** `put_signal` adds to a signal on the
peer after the put's data. Once the peer observes the signal (`wait_signal` /
`read_signal`, acquire), that put's data is visible to it. The signal is weak:
it says nothing about other puts. Send each payload as one `put_signal`.

**Local completion is `flush`'s job.** A put returns before the NIC has read
its source. `flush` waits until every put on the context has been consumed, so
the source may be reused. `flush_timeout(comm, ctx, cycles)` gives up after
`cycles` GPU clocks and returns 1 instead of hanging. A protocol whose
destination replies before a source is rewritten, as wide EP's does, needs no
flush.

**The put's descriptor must reach the CPU.** With the IB-proxy backend, a put
is a descriptor that the GPU writes to a queue in host memory, for RCCL's CPU
proxy thread to post. Triton's LLVM cannot lower the system-scope stores RCCL
normally uses for it. So the shim is built with RCCL's fallback, whose stores
can stay in L2 until the kernel exits (ROCm/rocm-systems#15056). The shim
therefore calls `__threadfence_system()` after every put, which writes the
descriptor out of L2 to where the proxy can see it. This is the same hazard,
and the same remedy, as the producer side of
[triggered plans](triggered-plans.md#memory-ordering): an agent outside the
shader reads behind L2. Without the fence, anything in the same kernel that
waits on its own put, such as a flush or a reply, never returns.

## Wide expert parallelism

`WideEPBuffer(provider, num_experts, hidden, max_tokens, topk)` shards experts
over every rank and follows DeepEP's normal mode:

```python
recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_weights)
y = run_local_experts(recv_x, recv_expert, recv_weight)
out = buf.combine(y, handle)
```

One window holds a send slot and a receive slot per peer for each exchange. A
slot is a 64-byte header with the row count, followed by rows. Each
destination gets its slot in one transfer: `iris.store` plus a release flag for
a direct peer, or one `put_signal` for the rest. Completion is
epoch-counted: a receiver waits until every source's flag or signal reaches the
exchange's epoch. No exchange needs a barrier, because a rank only rewrites its
send slots after every destination has replied to the previous round.

## Requirements

- ROCm with HIP 7.12 or newer. RCCL compiles out the cuMem support that GIN
  needs below that.
- An RCCL with the GIN device API that delivers. The RCCL 2.30.7 bundled with
  ROCm 10.1 does, as does RCCL develop 2.32. The RCCL 2.30.4 bundled with
  ROCm 10.0 accepts puts but does not deliver them.
- `NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1`, RDMA NICs
  visible to the process, and RDMA userspace (`libibverbs` and the NIC's
  provider), which ROCm 10.1 images do not include.
  `NCCL_CUMEM_ENABLE=1` applies to every RCCL communicator in the process:
  with RCCL 2.30.7 it makes some all-to-all and all-gather sizes several times
  slower (ROCm/rocm-systems#15494).
- With a separately built RCCL, exactly one RCCL in the process: replace the
  library torch loads rather than preloading a second one.
- Enough open files for RCCL's cuMem sharing. The provider raises the soft
  limit itself, and raises `RcclGinUnavailable` on every rank together if the
  hard limit is too low.

**Without a GIN backend**, for example on a machine with no RDMA NIC, the
provider still works when every rank is in RCCL's load/store team. Then every
peer is `direct`, `gin_available` is False, and wide EP runs on Iris stores and
flags alone. Ranks outside the team need GIN.

Missing prerequisites raise `RcclGinUnavailable`; nothing else does. The
example README explains how each requirement was found and what it looks like
when unmet.
