# Collective Plans

A collective plan sends data that a running kernel produces to every other rank without spending
compute units on the transfer. The host prepares copy-engine (SDMA) transfers ahead of time; the
kernel marks each chunk of its output ready once it is written, and the copy engine copies the
chunk to every peer and signals its arrival there. This technique is often called *triggered SDMA*.

```
kernel writes a chunk ──► mark_ready(chunk) opens the chunk's gate
                               │
          copy engine:  POLL_REGMEM(ready >= round) ─► COPY × regions ─► ATOMIC(peer arrival += 1)
                               │
peer kernel:  wait(chunk) ◄────┘
```

The API is `iris.plan`; see the [reference](../reference/host/plan.md).

## Model

- A **region** is one copy: a byte range, or a 2D sub-window (rows of a strided tensor). It is
  written at the same offset in each peer's buffer. A region is independent of how the producer
  tiles its output: `tile_chunks()` makes one per tile, `chunks()` one per row block (more above
  1 GiB), and a hand-built schedule may cover part of a tile or several adjacent tiles.
- A **chunk** is an ordered group of regions sent together, with one arrival signal per peer.
  Grouping amortizes the poll and the signal over many copies.
- A chunk has **writers**: the number of `mark_ready` calls that complete it each round. One for
  a chunk written by a single CTA; the number of CTAs for a chunk they write together.
- A **schedule** lists the chunks one rank writes, in the order its producer completes them. The
  plan prepares them in that order on each peer's queue.
- A **round** is one use of the plan: `start()`, the producer marks every chunk ready, `wait()`.
  The plan is prepared once and reused for every round.

`iris.plan.schedule` builds schedules for common producers: `whole()` (one chunk, marked ready
once per program at the end), `chunks()` (row blocks completed in order) and `tile_chunks()`
(consecutive tiles of a tiled kernel, such as a GEMM). Any list of chunks works.

```python
from iris.plan import CollectivePlan, CollectiveView, tile_chunks

ctx = iris.iris(heap_size, allocator_type="vmem_chunked", engines=("sdma",))
out = ctx.zeros(world_size * M, N, dtype=torch.bfloat16)
mine = out[rank * M : (rank + 1) * M]
schedule = tile_chunks(mine, (BM, BN), tiles_per_chunk=num_programs, base=out)

with CollectivePlan(ctx, out, schedule) as plan:   # collective: allocates gates, prepares round 1
    for step in range(steps):
        handle = plan.start()
        producer[(num_programs,)](..., handle)     # CollectiveView.initialize(handle).mark_ready(chunk)
        plan.wait()                                # every chunk delivered everywhere
        ...                                        # read this round's data
        plan.next_round()                          # collective: barrier, prepare, barrier
```

## The kernel author's contract

1. **Chunks complete in schedule order.** All of a rank's copies to one peer share one queue and
   run strictly in the order they were prepared, so a chunk completed early waits for every chunk
   before it. Out-of-order completion is still correct, only slower. With `tiles_per_chunk` equal
   to the number of programs of a persistent kernel whose programs each work through their tiles
   in order, chunks complete in order by construction.
2. **Each chunk is marked ready exactly `writers` times per round,** each after the caller's
   writes to that chunk. `mark_ready` must be called by the whole CTA, outside divergent branches.
   `wait()` checks the count for this rank's chunks: it raises if a chunk got extra marks (which
   would otherwise open later rounds' gates early), and a timeout names the chunks still short.
3. **A producer never waits on its own plan's arrivals before it has marked ready everything it
   writes.** Its wait could depend on a copy queued behind one of its own unfinished chunks.
4. **Region sizes are fixed when the plan is created.**
5. **`arrived` and `wait` are called uniformly across the CTA:** every wave makes the same number
   of calls. Each is one atomic whose result reaches the other waves through a CTA barrier, so a
   polling loop must not exit on a per-wave condition (such as a clock read): waves would read the
   result before it is written, or deadlock. Count polls instead.
6. **`next_round()` means "done reading this round's data".** Once it returns, a peer may start the
   next round, and its copies can overwrite the buffer at any time. Work queued before the call is
   covered; nothing issued after it may read the buffer until the next `wait()`.

Plans that share queues must start, and close, in the order they were prepared. While copies are
prepared on a queue, host `Iris.put()` and `quiet()` to that peer raise rather than queue behind
them.

Creating a plan and `next_round()` prepare on every rank or on none: if one rank cannot (for
example its queue is full, or its schedule is invalid), every rank raises, and a failed
construction flushes what it prepared. After a failed `next_round()` the plan is either still done
or `FAILED`.

`close()` is collective and waits for every rank, since a peer's copies and polls can still touch
a rank's buffer and gates after its own work is done. Prepared copies cannot be withdrawn from the
queue, so closing a plan in the middle of a round (or before it starts) sends each rank's chunks
as they stand. Leaving a `with` block closes the plan, including on an exception.

## Memory ordering

Measured on 8× MI350X (gfx950) and 8× MI300X (gfx942).

**Producer.** Copy engines read memory behind the GPU's L2, so a chunk's data must be written
back before its gate opens. A release fence only covers the issuing wave, so every wave of the
CTA must first complete its stores. `mark_ready` does this:

```python
wait_cnt()                 # s_waitcnt vmcnt(0): this wave's stores are done
tl.debug_barrier()         # every wave's stores are done
tl.atomic_add(counter, 1, sem="acq_rel", scope="sys")   # the last writer then:
tl.atomic_xchg(ready, round, sem="release", scope="sys")
```

| Release recipe                                  | Stale data at the peer (256 KiB and 4 MiB) |
| ----------------------------------------------- | ------------------------------------------ |
| `wait_cnt` + barrier + release (`gpu` or `sys`) | none in 350 rounds                         |
| release without the barrier                     | 80–100% of rounds                          |
| `.wt` store of the gate                         | every element, every round                 |
| plain store                                     | none, but the gate stays in L2 until kernel exit |

**Consumer.** `arrived` / `wait` read the arrival gate with an acquire at system scope; data read
after it in the same kernel is fresh (no stale elements over 9,600 tile-rounds).

## Gates and rounds

- Each chunk has a 64-byte gate row: `ready`, `arrival` and a writer counter. The copy engine
  polls the low 32 bits of `ready` with `>=`.
- Rounds start at 1 and add 1 per `next_round()`. Gates are never reset, so a plan is limited to
  2³² − 1 rounds.
- `wait()` checks completion with a kernel: this rank's chunks marked ready and arrived at every
  peer (each copy chain ends with that signal), and every other chunk arrived here.

## Requirements and limits

- **Stable peer mappings.** Prepared copies hold peer addresses, so the heap must not re-map peers
  while copies are prepared. Create the context with `allocator_type="vmem_chunked"` (or `"vmem"`):
  both keep peer addresses fixed, and the chunked allocator maps new heap chunks into each peer's
  reserved range as the heap grows. The default torch allocator re-maps peer heaps on every
  allocation and is rejected.
- **SDMA engine.** Create the context with `engines=("sdma",)` (needs `pip install iris[sdma]`).
- **One queue per peer.** Plans use the host-initiated copy-engine queue to each peer, shared with
  host `Iris.put()` and with the plans of every Iris context in the process. Kernel-initiated
  copies (`put(..., use_copy_engine=True)`) use a separate queue and are unaffected.
- **Ring capacity.** Prepared copies may use at most half of each queue's 8 MiB ring.
- **Same offsets everywhere.** Every chunk is copied to the same offsets in each peer's buffer,
  so plans express all-gather and broadcast (ranks that send nothing pass an empty schedule).
- **Single node only.**
- **Linear regions are at most 1 GiB** (SDMA 4.4's count field is 30 bits). The schedule builders
  split larger ranges.
