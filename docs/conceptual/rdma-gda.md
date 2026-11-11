# GPU-initiated RDMA

With `engines=("rdma",)` a Triton kernel writes to ranks on other hosts through an mlx5 NIC: it builds the
work request, rings the doorbell and polls the completion queue itself. It needs rocm-xio built with
`-DGDA_MLX5=ON -DXIO_BUILD_PYTHON=ON` (ROCm 7).

```python
ctx = iris.iris(heap_size, allocator_type="vmem_chunked", engines=("rdma",), rdma_device="mlx5_0")

@triton.jit
def send(data, flag, rank, peer, heap_bases, nbytes, rdma_ctx, status):
    index = rdma_put_signal(data, data, flag, rank, peer, heap_bases, nbytes, 1, rdma_ctx)
    tl.store(status, rdma_quiet(peer, index, rdma_ctx).to(tl.int32))  # 0, or 1 + the NIC's error syndrome
```

- Peers are not mapped; every peer is reached over the NIC, so use one rank per host.
- The heap is one physical chunk registered with the NIC (`torch` or `vmem_chunked`); it cannot grow past
  `heap_size`, which also holds Iris's own allocations.
- A queue pair is driven by the host or by kernels, not both.
- Call `rdma_quiet` at least every few hundred requests: 350 or more outstanding did not complete in testing.
