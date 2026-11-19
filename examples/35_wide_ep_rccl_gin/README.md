# Wide expert parallelism with RCCL GIN

A prototype of native wide-EP MoE dispatch and combine in Iris, spanning fabric
domains: load/store within one, RCCL GIN between them. On standalone nodes
linked by RDMA NICs, as on the MI355X systems this was validated on, a fabric
domain is one node. Experts are sharded over every GPU. Each token travels once to
each rank that owns one of its top-k experts, and the partial results come back
the same way. The API follows DeepEP's normal mode:

```python
import iris
from iris.experimental.wide_ep import WideEPBuffer

# after dist.init_process_group (gloo is enough); the heap is one RCCL window
ctx = iris.iris(heap_size, engines=("rdma",), rdma_backend="rccl_gin")
buf = WideEPBuffer(ctx, num_experts, hidden, max_tokens, topk)

recv_x, recv_expert, recv_weight, handle = buf.dispatch(x, topk_idx, topk_weights)
y = run_my_experts(recv_x, recv_expert, recv_weight)  # [rows, hidden], weighted, summed per row
out = buf.combine(y, handle)  # [tokens, hidden]
```

## How it works

The Iris heap is one RCCL symmetric window, and one buffer on it holds every
send and receive slot. A single Triton
kernel picks the transport for each destination:

| destination | transport | completion |
| --- | --- | --- |
| load/store reachable (`ctx.rdma.direct`: RCCL's LSA team, the node or the fabric domain on rack-scale systems) | `iris.store` with the heap bases | `iris.atomic_xchg` release flag |
| any other rank | `iris.rdma.device.put_signal`, which wraps RCCL's GIN `put` | weak GIN signal carried by the same put |

The load/store half is unmodified Iris: the heap bases of same-node peers are
RCCL's mappings of the window. Peers outside the LSA team have base 0 and are
not `direct`, and the kernel sends them through RDMA instead. The kernels route
on that `direct` mask itself, so routing and the backend cannot disagree about
which peers are addressable.

RCCL builds its LSA team from consecutive world ranks, so launch with
node-major ranks: torchrun's default, ranks 0-7 on the first node. If ranks
interleave across nodes, RCCL shrinks the team, and peers it could have mapped
fall back to GIN. That is correct, just slower.

The GIN half is RCCL's own GIN device API. `iris/csrc/rccl_gin` compiles it
to LLVM bitcode that Triton links (`extern_libs=ctx.rdma.extern_libs()`). It
uses a scalar C ABI because Triton's `extern_elementwise` cannot pass the
structs that RCCL's own `librccl_device.bc` thunks take.

## Requirements

Each of these was a separate failure on the way to a working run, and none of
the error messages names the cause:

- **ROCm with HIP 7.12 or newer; ROCm 10.1 is the simplest.** RCCL compiles
  out cuMem (VMM) support below HIP 7.12, and GIN and the device API need it.
  Built against ROCm 7.2, RCCL logs `cuMem VMM APIs are unavailable in this
  build` and then reports no device API support.
- **An RCCL whose GIN delivers.** The RCCL 2.30.7 bundled with ROCm 10.1 does,
  and needs no rebuild. The RCCL 2.30.4 bundled with ROCm 10.0 does not: its
  GIN proxy accepts puts, but nothing is ever delivered and `flush` hangs. RCCL
  develop (2.32.3, from `ROCm/rocm-systems`) delivers too. Two limits of
  2.30.x: `flush_timeout` waits without bound, because it has no timed flush;
  and it corrupts its heap creating a device comm with zero GIN contexts
  (ROCm/rocm-systems#14178, fixed in later RCCL), which the provider never
  does.
- **RDMA userspace in the container.** ROCm 10.1 images carry no
  `libibverbs`, and without it RCCL reports GIN unavailable without saying
  why. Install `libibverbs1` and `ibverbs-providers`, and make the NIC
  vendor's provider library visible: for AMD Pensando Pollara NICs that is
  `libionic`, from the host's driver install, bind-mounted in.
- **The GIN IB-proxy backend:** `NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1
  NCCL_DMABUF_ENABLE=1`, and an RDMA NIC visible in the container.
  `NCCL_CUMEM_ENABLE=1` changes every RCCL communicator in the process, torch's
  included: with RCCL 2.30.7, all-to-all becomes 4.6-5.7x slower on 8 GPUs, and
  all-gather 6.4x at 32 MiB per rank (ROCm/rocm-systems#15494). That is why CI
  sets it only for the provider's test file.
- **RCCL's fallback descriptor stores need a fence, which the shim adds.**
  `csrc/rccl_gin/build.sh` builds with `DWORDX4_INTRINSICS_FORCE_OFF`, since
  Triton's LLVM cannot lower RCCL's system-scope 128-bit stores. The fallback
  stores can stay in L2 until the kernel exits, invisible to RCCL's CPU proxy,
  so a flush in the same kernel as a put timed out (ROCm/rocm-systems#15056).
  The workaround is in the shim: after every put it calls
  `__threadfence_system()`, which writes the descriptor out of L2 so the proxy
  sees it at once. Iris's triggered plans do the same for the copy engine. A standalone HIP repro of the bug is in that issue.
- **With a separately built RCCL, exactly one RCCL in the process: replace
  the bundled one, don't add another.** torch's ROCm wheels load their bundled
  `librccl` by full path, so `LD_PRELOAD` loads a second copy rather than
  replacing it. Two RCCLs in one process abort with `double free or
  corruption` at exit; even a bare `import torch` does it. In a container,
  bind-mount the GIN build over the bundled file:
  `-v <rccl>/lib/librccl.so.1.0:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1:ro`.
  With ROCm 10.1's own RCCL there is only one, so this does not apply.
- **Enough open files.** With cuMem, RCCL shares each mapped peer buffer as a
  file descriptor, separately for every communicator in the process, and
  torch's own process group counts as one. At 8 ranks per node that exceeds
  the 1024 soft limit containers start with, and window registration fails
  with only `unhandled system error`; with `NCCL_DEBUG=WARN` it shows up as
  `UDS: Socket creation error : Too many open files`.
  `RcclGinProvider` raises the soft limit itself, up to the hard limit. If any
  rank's hard limit is below 65536, every rank raises `RcclGinUnavailable`
  together, before any RCCL collective, rather than some failing and the rest
  hanging. Start the container with `--ulimit nofile=65536:65536` or higher,
  or for a small job known to fit, lower the bar with
  `IRIS_RCCL_GIN_MIN_OPEN_FILES`.
- **With a separately built RCCL on a TheRock-style image, put torch's
  `_rocm_sdk_core/lib` first on `LD_LIBRARY_PATH`.** Otherwise the new librccl
  pulls in the second HIP/comgr copy from `_rocm_sdk_devel`. Two LLVMs in one
  process then abort with `Option ... registered more than once`.

Iris builds the shim (device bitcode and host helper) on first use, against
`RCCL_HOME` if set, else the pip-installed ROCm (`rocm-sdk path --root`), else
`ROCM_PATH`, and caches it in `IRIS_RCCL_GIN_CACHE` (default
`~/.cache/iris/rccl_gin`). Point `RCCL_HOME` at a separately built RCCL when
that is the one that runs. To build by hand instead, and use it through
`IRIS_RCCL_GIN_BUILD`:

```bash
RCCL_HOME=/path/to/rccl-install ARCH=gfx950 OUT=/path/to/out iris/csrc/rccl_gin/build.sh
```

By default the device bitcode is built with `-DDWORDX4_INTRINSICS_FORCE_OFF`
(`STORE_B128=fallback`). RCCL releases before ROCm/rocm-systems#15594 write
their queue descriptors with `__builtin_amdgcn_global_store_b128`, which exists
only in AMD's LLVM fork; Triton's upstream LLVM aborts with `Unknown type!`.
The switch selects RCCL's own fallback, non-temporal 64-bit stores, plus a
fence after each put (see the docs page on memory ordering). With a RCCL that
has #15594 and Triton 3.8 or newer, `IRIS_RCCL_GIN_STORE_B128=native` (or
`STORE_B128=native` for build.sh) uses RCCL's own stores and no fence: same
speed, and the path RCCL itself is tested on. Iris CI pins Triton 3.7, so the
fallback stays the default.

## Running

One `torchrun` per node:

```bash
# inside a container that meets the Requirements
export NCCL_GIN_TYPE=2 NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1
export NCCL_SOCKET_IFNAME=eno0 GLOO_SOCKET_IFNAME=eno0
torchrun --nnodes 2 --node_rank $NODE --nproc_per_node 8 \
    --master_addr $HEAD --master_port 29500 \
    examples/35_wide_ep_rccl_gin/gin_smoke.py      # transport check
torchrun ... examples/35_wide_ep_rccl_gin/example_run.py   # MoE layer, checked + timed
```

`gin_smoke.py` runs an all-to-all of tagged blocks over both paths.
`example_run.py` runs MoE steps and checks them against a reference that each
rank computes alone. Expert weights come from a seed, so no communication is
needed for the reference.

## Status

Prototype, validated on 2 nodes of 8 MI355X each (gfx950, AMD Pensando Pollara
NICs), with ROCm 10.1 and its own RCCL 2.30.7, torch 2.14 and Triton 3.8:

| run | ranks | result |
| --- | --- | --- |
| `gin_smoke.py` | 2 x 8, 2 x 2 | PASS (8 direct + 8 GIN peers per rank at 16 ranks) |
| `example_run.py` 64 experts, top-8, 512 tok, hidden 2048 | 2 x 8 | PASS, max abs err 6.9e-3 vs fp32 reference |

On the 2 x 8 run, half of each rank's token copies cross nodes. Untuned timing
for 14.2 MB of token data per rank each way: dispatch 1.14 ms, combine 0.36 ms.
Both figures include host-side layout and synchronisation in `dispatch` (see
Known limitations), so they overstate the communication cost; the split
between that, the copy kernels and GIN delivery has not been measured.

Without a GIN backend (no RDMA NIC, as on CI runners), the backend still works
when every rank is in one load/store team, as on one node. The load/store tests
then run and the GIN tests skip. RCCL itself refuses the device API across nodes
without GIN.

### Known limitations

- **Wide-EP does not flush, by design.** A receiver's signal is the
  completion, and a send slot is only rewritten after its destination has
  replied, which proves the NIC has read it. `rccl_gin.flush` and
  `flush_timeout` are there for code that reuses a source buffer without such
  a reply.
- **`dispatch` synchronises with the host twice:** once to lay out tokens
  (`nonzero`) and once to read the received counts. Each receive slot is sized
  for the full `max_tokens`. DeepEP's low-latency mode avoids both, and so
  could this.
- **One GIN put per destination, issued from one workgroup, on one GIN
  context.** Large messages would benefit from chunking across contexts and
  NICs.
- **Validated on the proxy backend.** The GPU writes descriptors and a CPU
  thread posts the RDMA. RCCL's rocSHMEM-GDA backend (`NCCL_GIN_TYPE=6`)
  removes that hop and works through the same shim with no change, given an
  RCCL built with `-DENABLE_ROCSHMEM_GIN=ON` that includes
  ROCm/rocm-systems#14034 and #14219 (RCCL `develop`; the RCCL 2.30.7 in
  ROCm 10.1 predates them). Its GIN contexts share one queue pair per peer
  unless `NCCL_GIN_NCONNECTIONS` is raised.
- **The heap cannot grow.** It is one RCCL window of `heap_size`.
