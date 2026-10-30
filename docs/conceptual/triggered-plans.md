# Triggered Plans

A triggered plan sends data that a running kernel produces to every other rank without spending
compute units on the transfer. The host prepares copy-engine (SDMA) transfers ahead of time; the
kernel marks each chunk of its output ready once it is written, and the copy engine copies the
chunk to every peer and signals its arrival there: *triggered SDMA*, hence the name.

```
kernel writes a chunk ──► mark_ready(chunk) opens the chunk's gate
                               │
          copy engine:  POLL_REGMEM(ready >= round) ─► COPY × regions ─► ATOMIC(peer arrival += 1)
                               │
peer kernel:  wait(chunk) ◄────┘
```

The API is `iris.experimental.plan`; see the [reference](../reference/host/plan.md). It is
experimental: it may change between releases without deprecation.

## Model

- A **region** is one copy: a byte range, or a 2D sub-window (rows of a strided tensor). It is
  written at the same offset in each peer's buffer. A region is independent of how the producer
  tiles its output: `tile_chunks()` makes one per tile, `chunks()` one per row block (split into
  1 GiB pieces: a linear copy packet's byte count is a 30-bit field, so one copy moves at most
  1 GiB), and a hand-built schedule may cover part of a tile or several adjacent tiles.
- A **chunk** is an ordered group of regions sent together, with one arrival signal per peer.
  Grouping amortizes the poll and the signal over many copies.
- A chunk has **writers**: the number of `mark_ready` calls that complete it each round. One for
  a chunk written by a single CTA; the number of CTAs for a chunk they write together.
- A **schedule** lists the chunks one rank writes, in the order its producer completes them. The
  plan prepares them in that order on each peer's queue.
- A **round** is one use of the plan: `start()`, the producer marks every chunk ready, `wait()`.
  The plan is prepared once and reused for every round.

`iris.experimental.plan.schedule` builds schedules for common producers: `whole()` (one chunk, marked ready
once per program at the end), `chunks()` (row blocks completed in order) and `tile_chunks()`
(consecutive tiles of a tiled kernel, such as a GEMM). Any list of chunks works.

A schedule and its producer must agree on which tiles each chunk holds. `tile_chunks()` takes
the producer's step-to-tile mapping as `order`; pass the kernel's own mapping rather than a copy
of it, so the two cannot drift:

```python
@triton.jit
def tile_of(step, tiles_m, tiles_n, GROUP_M: tl.constexpr):   # plain integer arithmetic only
    ...

# Host: the same function, run in Python
schedule = tile_chunks(mine, (BM, BN), num_programs, order=lambda i: tile_of.fn(i, tiles_m, tiles_n, 8), base=out)

# Kernel: program p takes steps p, p + num_programs, ...
for step in range(tl.program_id(0), num_tiles, num_programs):
    tile = tile_of(step, tiles_m, tiles_n, 8)
    ...                                               # compute and store the tile
    cv.mark_ready(first_chunk + step // num_programs)
```

```python
from iris.experimental.plan import TriggeredPlan, TriggeredView, tile_chunks

ctx = iris.iris(heap_size, engines=("sdma",))
out = ctx.zeros(world_size * M, N, dtype=torch.bfloat16)
mine = out[rank * M : (rank + 1) * M]
schedule = tile_chunks(mine, (BM, BN), tiles_per_chunk=num_programs, base=out)

with TriggeredPlan(ctx, out, schedule) as plan:   # collective: allocates gates, prepares round 1
    for step in range(steps):
        handle = plan.start()
        producer[(num_programs,)](..., handle)     # TriggeredView.initialize(handle).mark_ready(chunk)
        plan.wait()                                # every chunk delivered everywhere
        ...                                        # read this round's data
        plan.next_round()                          # collective: prepare, barrier
```

For this in-place all-gather, `ctx.ccl.prepare_all_gather(out, schedule)` builds the same plan and
also checks that each rank's schedule stays inside its own rows of `out` (by default it sends them as
one chunk, `whole()`). See `examples/34_sdma_all_gather`.

### Gluon

Gluon kernels use `iris.experimental.plan.gluon.TriggeredView`. It reads the same handle and has
the same methods as the Triton `TriggeredView`, so the host side does not change:

```python
from iris.experimental.plan.gluon import TriggeredView

@gluon.jit
def producer(out, handle, ...):
    ...                                            # write this program's part of the chunk
    TriggeredView.initialize(handle).mark_ready(chunk)
```

`mark_ready` waits for vector-memory stores, including AMD buffer stores (`buffer_store`). Call it
after `gl.warp_specialize` returns, not inside a partition.

## The kernel author's contract

1. **Chunks complete in schedule order.** All of a rank's copies to one peer share one queue and
   run strictly in the order they were prepared, so a chunk completed early waits for every chunk
   before it. Out-of-order completion is still correct, only slower. With `tiles_per_chunk` equal
   to the number of programs of a persistent kernel whose programs each work through their tiles
   in order, chunks complete in order by construction.
2. **Each chunk is marked ready exactly `writers` times per round, by the programs that write its
   regions,** each after the caller's writes to that chunk. `mark_ready` must be called by the
   whole CTA, outside divergent branches.
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

Plans that share queues must start in the order they were prepared. While copies are prepared
on a queue, host `Iris.put()` and `quiet()` to that peer raise rather than queue behind them.

A chunk marked ready by programs that did not write all of its regions is sent before its data is
written. The mark count cannot catch this (the count is right, the tiles are wrong), which is why
`order` should come from the kernel's own mapping.

Creating a plan and `next_round()` prepare on every rank or on none: if one rank cannot (for
example its queue is full, or its schedule is invalid), every rank raises, and a failed
construction flushes what it prepared. After a failed `next_round()` the plan is either still done
or `FAILED`.

`close()` is collective and waits for every rank, since a peer's copies and polls can still touch
a rank's buffer and gates after its own work is done. Prepared copies cannot be withdrawn from the
queue, so closing a plan in the middle of a round (or before it starts) sends each rank's chunks
as they stand. Leaving a `with` block closes the plan, including on an exception.

Plans can be closed in any order, so `with` blocks nest. A plan whose copies are queued behind an
earlier plan's waiting copies cannot finish first: its `close()` returns with the plan `CLOSING`,
and closing the earlier plan finishes both.

## Memory ordering

**Producer.** Copy engines read memory behind the GPU's L2, so a chunk's data must be written
back to memory before its gate opens. Each writer does that with a release on the chunk's writer
counter, and the writers may run on different XCDs, each with its own L2. A release only orders
the stores of the thread that performs it, and `mark_ready` performs it on one thread for the whole
CTA, so the other waves' stores are ordered before it with a CTA barrier that waits for every
wave's stores, as HIP's `__syncthreads()` does. The last writer's increment follows every other
writer's, and it opens the gate only after its increment returns, so the gate itself needs no
release:

```python
wait_cnt()                 # s_waitcnt vmcnt(0): this wave's stores are done
tl.debug_barrier()         # every wave has got here, so every store of the CTA is done
tl.atomic_add(counter, 1, sem="release", scope="sys")   # write back this XCD's L2; the last writer then:
tl.atomic_xchg(ready, round, sem="relaxed", scope="sys")
```

Recent Triton versions already wait for outstanding stores in `tl.debug_barrier()`; `wait_cnt()`
keeps the barrier a memory barrier on versions where it only synchronizes execution. Without the
barrier, the copy engine sends stale data for most chunks.

**Consumer.** `arrived` / `wait` read the arrival gate with an acquire at system scope; data read
after it in the same kernel is fresh.

## Gates and rounds

- Each chunk has a 64-byte gate row: `ready`, `arrival` and a writer counter. The copy engine
  polls the low 32 bits of `ready` with `>=`.
- Rounds start at 1 and add 1 per `next_round()`. Gates are never reset, so a plan is limited to
  2³² − 1 rounds.
- `wait()` checks completion with a kernel: this rank's chunks marked ready and arrived at every
  peer (each copy chain ends with that signal), and every other chunk arrived here.

## Requirements and limits

- **SDMA engine.** Create the context with `engines=("sdma",)` (needs `pip install iris[sdma]`).
- **One queue per peer.** Plans use the host-initiated copy-engine queue to each peer, shared with
  host `Iris.put()` and with the plans of every Iris context in the process. Kernel-initiated
  copies (`put(..., use_copy_engine=True)`) use a separate queue and are unaffected.
- **Ring capacity.** Prepared copies may use at most half of each queue's 8 MiB ring. They hold
  their ring space until their gates open, and a submit that does not fit waits for space, so a
  full ring could hang the host before any kernel opens a gate. The other half covers wrap padding
  and work queued before them. Half is a conservative margin, not a measured bound.
- **Same offsets everywhere.** Every chunk is copied to the same offsets in each peer's buffer,
  so plans express all-gather and broadcast (ranks that send nothing pass an empty schedule). Two
  ranks may not send the same bytes: the plan checks this when it is created, which catches a
  schedule built from the whole buffer instead of the rank's own part.
- **Single node only.**
- **Linear regions are at most 1 GiB** (`COPY_LINEAR`'s byte count is 30 bits on SDMA 4.4). The
  schedule builders split larger ranges.

## Not supported yet

- **Other destinations.** A copy always lands at its source's offset in the same buffer on every
  peer: it cannot change offset, pitch or buffer (for example to gather into a staging buffer), and
  a chunk cannot go to only some peers. Reduce-scatter and two-shot all-reduce need both.
- **Waiting on a group.** A consumer waits on each chunk's arrival in turn; no single counter
  covers several chunks or several source ranks.
- **Overlapping rounds.** `wait()` must return before `next_round()` prepares the next round, and
  preparing synchronizes the ranks. The host's work per round grows with regions × peers, so a
  plan suits producers whose rounds are long next to that, rather than an op called many times
  per step.
