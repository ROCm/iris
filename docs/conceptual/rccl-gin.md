# RCCL GIN and Wide Expert Parallelism

Iris's load/store model reaches every GPU whose memory it can map: the GPUs of
one node, and, with the fabric drivers, every host in one scale-up fabric
domain such as a Helios or NVL72 rack. GPUs beyond a fabric domain are
reachable only over RDMA NICs. RCCL's GPU-Initiated Networking (GIN) covers
that tier: a kernel asks for a one-sided write to a peer in another fabric
domain, and the network carries it. Iris exposes it as an RDMA backend:

```python
ctx = iris.iris(heap_size, engines=("rdma",), rdma_backend="rccl_gin")
```

and `iris.experimental.wide_ep` builds MoE dispatch and combine across fabric
domains on it.

```
                 the Iris heap = one RCCL symmetric window (same offsets on every rank)
                                          │
  same fabric domain (ctx.rdma.direct)  ──► iris.store / iris.atomic_*   (load/store)
  other fabric domains                  ──► iris.rdma.device.put_signal  (GIN, over the NIC)
```

On a system of standalone nodes, such as MI355X nodes linked by RDMA NICs, a
fabric domain is one node.

Both modules are experimental: they may change between releases without
deprecation. A runnable example, the full environment requirements and
measured results are in
[examples/35_wide_ep_rccl_gin](https://github.com/ROCm/iris/tree/main/examples/35_wide_ep_rccl_gin).

## RDMA backends

`engines=("rdma",)` is backend-neutral. `iris.rdma.BACKENDS` maps a name to a
backend class, and kernels call `iris.rdma.device` (`put`, `put_signal`,
`wait_signal`, `flush`), which dispatches at compile time on the backend's
constexpr `ctx.rdma.BACKEND_ID`; a kernel links only its own backend. A backend
either provides the heap (`provides_heap`, as RCCL GIN does: GIN only sends
from and to RCCL windows, and RCCL maps the window on same-node peers itself),
or registers a heap Iris allocated, which is where a NIC verbs backend fits.
`rccl_gin` is the only backend registered so far.

With `rccl_gin`, Iris's heap is one RCCL window of `heap_size` (plus Iris's own
reservation): `ncclMemAlloc` + `ncclCommWindowRegister`, collectively. It
cannot grow; an allocation past it raises. Peers in RCCL's load/store team get
RCCL's mapping as their heap base, so `iris.store`, `load`, atomics and copies
reach them unchanged. Every other peer has heap base 0: reach it only through
`iris.rdma.device`. `device_barrier` raises when some peers are not direct;
use `barrier()` or signals.

```python
dc = ctx.rdma.device_comm(signal_count=world)       # collective
rdma_ctx = ctx.rdma.device_context(dc)              # kernel arg
base = ctx.rdma.read_signals(dc)                    # signals only count up
kernel[grid](..., ctx.get_heap_bases(), rdma_ctx, BACKEND=ctx.rdma.BACKEND_ID,
             extern_libs=ctx.rdma.extern_libs())
...
ctx.rdma.close()                                    # collective; frees the heap too
```

Inside the kernel, pointers are ordinary heap pointers, as with `iris.put`:

```python
rdma.put_signal(src, dst, nbytes, rank, peer, heap_bases, rdma_ctx, 0, signal, 1, BACKEND)
rdma.wait_signal(rdma_ctx, 0, signal, least, BACKEND)   # least is absolute, not a count
```

## The provider underneath

`iris.experimental.rccl_gin.RcclGinProvider` is what the backend is built on
(`ctx.rdma.provider`), and can be used directly for separate windows outside
the Iris heap.

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
  bitcode by `iris/csrc/rccl_gin/build.sh` and linked into the Triton kernel.
  Iris builds them on first use, against the RCCL found in `RCCL_HOME`, the
  pip-installed ROCm, or `ROCM_PATH`, and caches the result in
  `IRIS_RCCL_GIN_CACHE` (default `~/.cache/iris/rccl_gin`) per RCCL, GPU
  architecture and source version; `IRIS_RCCL_GIN_BUILD` names a prebuilt
  shim instead. It is not built at `pip install` time because it must match
  the RCCL that runs, usually the one torch loads. Launch
  such kernels with `extern_libs=rccl_gin.extern_libs()`. Every one of them is
  a program-wide operation: all threads of the program must reach it, in
  program-uniform control flow.

## Routing

A kernel picks the transport per destination from the backend's `direct` mask:

```python
@triton.jit
def send(src, dst_buf, heap_bases, rdma_ctx, direct, rank, dst, nbytes, ..., BACKEND: tl.constexpr):
    if tl.load(direct + dst) != 0:
        iris.store(...)                                  # load/store reachable
    else:
        rdma.put_signal(src, dst_buf, nbytes, rank, dst, heap_bases, rdma_ctx, 0, rank, 1, BACKEND)
```

Route on the mask itself rather than on a rank range. Then routing and the
backend cannot disagree about which peers are addressable. RCCL builds its load/store
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
proxy thread to post. RCCL releases before ROCm/rocm-systems#15594 write it
with an intrinsic only AMD's LLVM fork has, which Triton's upstream LLVM cannot
lower. So by default the shim is built with RCCL's fallback, whose stores can
stay in L2 until the kernel exits (ROCm/rocm-systems#15056), and calls
`__threadfence_system()` after every put, which writes the descriptor out of L2
to where the proxy can see it. With a RCCL that has #15594 and Triton 3.8 or
newer, `IRIS_RCCL_GIN_STORE_B128=native` builds RCCL's own path instead, with
no fence; it measured the same speed. This is the same hazard,
and the same remedy, as the producer side of
[triggered plans](triggered-plans.md#memory-ordering): an agent outside the
shader reads behind L2. Without the fence, anything in the same kernel that
waits on its own put, such as a flush or a reply, never returns.

## Wide expert parallelism

`WideEPBuffer(ctx, num_experts, hidden, max_tokens, topk)`, with `ctx` an
`engines=("rdma",)` context, shards experts over every rank and follows DeepEP's normal mode:

```python
recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_weights)
y = run_local_experts(recv_x, recv_expert, recv_weight)
out = buf.combine(y, handle)
```

One heap buffer holds a send slot and a receive slot per peer for each exchange. A
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
- `NCCL_GIN_TYPE=2` (IB proxy) `NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1`, RDMA NICs
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

**GIN backends.** RCCL reports the one it brought up as `gin_type_name`:
`proxy` (`NCCL_GIN_TYPE=2`), `rocshmem_gda` (6, GPU-initiated; needs RCCL built
with `-DENABLE_ROCSHMEM_GIN=ON` and ROCm/rocm-systems#14034, and works through
the same shim) or `anvil_sdma` (7). `RCCL_GIN_TYPE_LIST` enables several. By
default every GIN context to a peer shares one connection (queue pair), so
spreading puts over contexts does not add bandwidth unless
`NCCL_GIN_NCONNECTIONS` is raised.

**Without a GIN backend**, for example on a machine with no RDMA NIC, the
provider still works when every rank is in RCCL's load/store team. Then every
peer is `direct`, `gin_available` is False, and wide EP runs on Iris stores and
flags alone. Ranks outside the team need GIN.

Missing prerequisites raise `RcclGinUnavailable`; nothing else does. The
example README explains how each requirement was found and what it looks like
when unmet.
