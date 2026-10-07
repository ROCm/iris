# Triggered SDMA

Triggered SDMA moves data that a running kernel produces without spending compute units on the
transfer. The host pre-arms copy-engine (SDMA) chains that wait on *gates*; the kernel opens a
gate when its data is written, and the copy engine copies the data to every peer and signals its
arrival there.

```
kernel writes a batch ──► publish(batch) opens the gate
                               │
          copy engine:  POLL_REGMEM(ready >= epoch) ─► COPY × segments ─► ATOMIC(peer arrival += 1)
                               │
peer kernel:  wait(batch) ◄────┘
```

The API is `iris.triggered`; see the [reference](../reference/host/triggered.md).

## Model

- A **segment** is one copy: a byte range, or a 2D sub-window (rows of a strided tensor). It is
  written at the same offset in each peer's buffer.
- A **batch** is an ordered group of segments behind one gate, with one arrival signal per peer.
  Batching amortizes the poll and the signal over many copies.
- A batch has **contributors**: the number of `publish` calls that complete it each epoch. One for
  a batch written by a single CTA; the number of CTAs for a batch they write together.
- A **schedule** lists a rank's batches in the order its producer completes them. The plan arms
  them in that order on each peer's queue.

`iris.triggered.schedule` builds schedules for common producers: `whole()` (one batch, published
once per program at the end), `chunks()` (row blocks completed in order) and `waves()` (one batch
per wave of a persistent tiled kernel, such as a GEMM). Any list of batches works.

```python
from iris.triggered import TriggeredPlan, TriggeredView, waves

ctx = iris.iris(heap_size, allocator_type="vmem_chunked")
out = ctx.zeros(world_size * M, N, dtype=torch.bfloat16)
schedules = [waves(out[r * M : (r + 1) * M], (BM, BN), num_programs, base=out) for r in range(world_size)]
plan = TriggeredPlan(ctx, out, schedules)          # collective: allocates gates, arms epoch 1

for step in range(steps):
    view = plan.begin_device_epoch()
    producer[(num_programs,)](..., view)           # TriggeredView.initialize(view).publish(batch)
    plan.end_device_epoch()                        # every batch arrived and drained
    ...
    plan.next_epoch()                              # barrier, re-arm, barrier
plan.destroy()
```

## The kernel author's contract

1. **Batches complete in schedule order.** All of a rank's chains to one peer share one queue and
   run strictly in arming order, so a batch completed early waits for every batch armed before it.
   Out-of-order completion is still correct, only slower. A persistent kernel whose programs each
   work through their tiles in order completes waves in order by construction.
2. **Each batch gets exactly `contributors` publishes per epoch,** each after the caller's writes
   to that batch. `publish` must be called by the whole CTA, outside divergent branches.
3. **A producer never waits on its own plan's arrivals before it has published everything it
   contributes to.** Its wait could depend on a chain parked behind one of its own unpublished
   batches.
4. **Segment sizes are fixed when the plan is created.**

Plans that share queues must begin, and be aborted, in the order they were armed. While chains
are armed on a queue, host `Iris.put()` and `quiet()` to that peer raise rather than queue behind
them.

## Memory ordering

Measured on 8× MI350X (gfx950) and 8× MI300X (gfx942).

**Producer.** Copy engines read memory behind the GPU's L2, so a batch's data must be written
back before its gate opens. A release fence only covers the issuing wave, so every wave of the
CTA must first complete its stores. `publish` does this:

```python
wait_cnt()                 # s_waitcnt vmcnt(0): this wave's stores are done
tl.debug_barrier()         # every wave's stores are done
tl.atomic_add(counter, 1, sem="acq_rel", scope="sys")   # the last contributor then:
tl.atomic_xchg(ready, epoch, sem="release", scope="sys")
```

| Release recipe                                  | Stale data at the peer (256 KiB and 4 MiB) |
| ----------------------------------------------- | ------------------------------------------ |
| `wait_cnt` + barrier + release (`gpu` or `sys`) | none in 350 epochs                         |
| release without the barrier                     | 80–100% of epochs                          |
| `.wt` store of the gate                         | every element, every epoch                 |
| plain store                                     | none, but the gate stays in L2 until kernel exit |

**Consumer.** `arrived` / `wait` read the arrival gate with an acquire at system scope; data read
after it in the same kernel is fresh (no stale elements over 9,600 tile-epochs).

## Gates and epochs

- Each batch has a 64-byte gate row: `ready`, `arrival`, `completion` (64-bit), `error`, and a
  contributor counter. The copy engine polls the low 32 bits of `ready` with `>=`.
- Epochs start at 1 and add 1 per `next_epoch()`. Gates are never reset, so a plan is limited to
  2³² − 1 epochs.
- `end_device_epoch()` checks completion with a kernel: this rank's batches published and arrived
  at every peer (each chain ends with that signal), and every other batch arrived here.

## Requirements and limits

- **Stable peer mappings.** Armed chains hold peer addresses, so the heap must not re-map peers
  while chains are armed. Create the context with `allocator_type="vmem_chunked"` (or `"vmem"`):
  both keep peer addresses fixed, and the chunked allocator maps new chunks into each peer's
  reserved range as the heap grows. The default torch allocator re-maps peer heaps on every
  allocation and is rejected.
- **One queue per peer.** Plans use the host-initiated copy-engine queue to each peer, shared with
  host `Iris.put()`. Kernel-initiated copies (`put(..., use_copy_engine=True)`) use a separate
  queue and are unaffected.
- **Ring capacity.** Armed chains may use at most half of each queue's 8 MiB ring.
- **Single node only.**
- **Linear segments are at most 1 GiB** (SDMA 4.4's count field is 30 bits). The schedule builders
  split larger ranges.
