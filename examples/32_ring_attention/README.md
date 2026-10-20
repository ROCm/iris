<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# Ring Attention

An implementation of **Ring Attention with Blockwise Transformers** for
near-infinite context on AMD GPUs using [Iris](../../README.md).

> Liu, H., Li, M., Hall, A., Dao, T., & Abbeel, P. (2023).
> *Ring Attention with Blockwise Transformers for Near-Infinite Context.*
> arXiv:2310.01889. <https://arxiv.org/pdf/2310.01889>

---

## Algorithm

Standard self-attention requires O(n²) memory in the sequence length n.
Ring Attention enables sequences far longer than what fits on a single device
by distributing them across a *ring* of GPUs:

1. The full sequence is split evenly across **N GPUs** along the sequence
   dimension. Each device holds a chunk of Q, K, and V of length
   `seq_total / N`.
2. **Q stays local**. K and V rotate around the ring one step at a time.
3. At each of the **N steps**, every device runs a local
   [Flash Attention](https://arxiv.org/abs/2205.14135) pass and accumulates
   the result using **online softmax**.
4. After all N steps the accumulator is normalised to yield the final output.

For **causal (autoregressive) attention** only the steps where the KV chunk
precedes or coincides with the Q chunk contribute, allowing early termination
for some ranks and reducing total compute.

```
Step 0:  rank r processes its own K_r, V_r          (causal block diagonal)
Step 1:  rank r receives K_{r-1}, V_{r-1}           (full attention, past)
...
Step r:  rank r receives K_0, V_0                   (full attention, past)
Step r+1..N-1: all-future chunks – skipped          (causal mode only)
```

---

## Files

| File | Description |
|------|-------------|
| `ring_attention_kernels.py` | Triton flash-attention kernel + Python ring-rotation helper |
| `ring_attention_layer.py`   | `RingAttention` – a `torch.nn.Module` wrapper |
| `example_run.py`            | End-to-end demo with timing |

---

## Usage

### Quick demo

```bash
# 2 GPUs, causal attention (default)
python examples/32_ring_attention/example_run.py

# 4 GPUs, bidirectional
python examples/32_ring_attention/example_run.py --num_ranks 4 --no_causal

# Custom sizes
python examples/32_ring_attention/example_run.py \
    --num_ranks 8 \
    --total_seq_len 131072 \
    --num_heads 32 \
    --head_dim 128
```

### Validation

```bash
python tests/run_tests_distributed.py tests/examples/test_ring_attention.py --num_ranks 2 -v
```

---

## Python API

```python
import iris
from examples.ring_attention.ring_attention_layer import RingAttention

shmem = iris.iris()

# Each rank holds its local chunk
layer = RingAttention(
    shmem,
    num_heads=16,
    head_dim=64,
    causal=True,       # autoregressive masking
)

# q, k, v: [seq_local, num_heads, head_dim]  (float16 or bfloat16)
output = layer(q, k, v)   # [seq_local, num_heads, head_dim]
```

---

## Design Notes

* **Communication**: KV rotation uses `iris.put` Triton kernels — each rank
  pushes its K/V chunk directly to the next rank's symmetric heap buffer.
  A `shmem.barrier()` after each push ensures all ranks have received the
  data before the next attention step proceeds. No `torch.distributed` APIs
  are used.
* **Ping-pong buffers**: Two symmetric buffer pairs (`k_ping`/`k_pong` and
  `v_ping`/`v_pong`) alternate as source and destination on each step.  This
  guarantees the source being read and the destination being written are
  always different allocations, avoiding any read-after-write hazard.
* **Online softmax**: The kernel maintains running max (`M`) and sum (`L`)
  accumulators in float32 for numerical stability.  The final output is
  `O / L` after all ring steps.
* **Causal masking**: Handled entirely at the granularity of KV *chunks* –
  full attention, diagonal block attention, or skip – so the per-element mask
  is applied only in the same-block diagonal case.  All ranks still
  participate in the rotation (required for the barrier to be well-defined).
