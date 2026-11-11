# Triggered Plans

A triggered plan sends data that a running kernel produces to other ranks without spending
compute units on the transfer. The host prepares copy-engine (SDMA) transfers ahead of time; the
kernel marks each chunk of its output ready once it is written, and the copy engine copies the
chunk to its peers and signals its arrival there: *triggered SDMA*, hence the name.

```
kernel writes a chunk ──► mark_ready(chunk): counter += 1
                               │
          copy engine:  POLL_REGMEM(counter >= writers × round) ─► COPY × regions ─► ATOMIC(peer arrival += 1)
                               │
peer kernel:  wait(chunk) ◄────┘
```

After a round's copies to a peer, the copy engine also adds 1 to the sending rank's completion
counter for that peer, which `wait()` and the device-side `wait_sent()` read.

The API is `iris.experimental.plan`; see the [reference](../reference/host/plan.md). It is
experimental: it may change between releases without deprecation.

## Model

- A **region** is one copy: a byte range, or a 2D sub-window (rows of a strided tensor). By
  default it is written at the same offset in each peer's buffer (see
  [Destinations, peers and groups](#destinations-peers-and-groups)). A region is independent of
  how the producer tiles its output: `tile_chunks()` makes one per tile, `chunks()` one per row
  block (split into 1 GiB pieces: a linear copy packet's byte count is a 30-bit field, so one copy
  moves at most 1 GiB), and a hand-built schedule may cover part of a tile or several adjacent
  tiles.
- A **chunk** is an ordered group of regions sent together, with one arrival signal per peer.
  Grouping amortizes the poll and the signal over many copies. A chunk goes to every other rank,
  or to its `peers`.
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
        plan.next_round()                          # collective: prepare, device barrier
```

`start()` sets the round in the handle with work queued on the current stream, so launch the kernels
that take the handle on that stream, or make their stream wait for it: a kernel on another stream
could still see the previous round, pass `wait` on that round's arrivals and read stale data.

For this in-place all-gather, `ctx.ccl.prepare_all_gather(out, schedule)` builds the same plan and
also checks that each rank's schedule stays inside its own rows of `out` (by default it sends them as
one chunk, `whole()`). See `examples/34_sdma_all_gather`.

### Overlapping rounds

Preparing a round costs host time that grows with regions × peers. On a plan created with
`prepare_ahead=True`, `prepare_next()` does it while the current round runs, so `next_round()` is
left with its round fence, a kernel on the current stream, and `wait(blocking=False)` queues the
wait on that stream too. A loop written this way never synchronizes the host with the GPU (see
`examples/34_sdma_all_gather`):

```python
with TriggeredPlan(ctx, out, schedule, prepare_ahead=True) as plan:
    for step in range(steps):
        handle = plan.start()
        producer[(num_programs,)](..., handle)
        plan.prepare_next()                        # host prepares round + 1 while the GPU works
        plan.wait(blocking=False)                  # queued: kernels after it see every chunk
        consumer[grid](out, ...)                   # read this round's data
        plan.next_round()                          # queued round fence: done reading
```

The next round's copies wait behind gates that no producer can open before every rank has passed
`next_round()`. A plan holds queue space for two rounds with `prepare_ahead`, so one round may use
half as much of each queue as without it.

A stream-ordered `wait()` cannot raise. What goes wrong is recorded on the GPU and raised, once,
by the next blocking `wait()` or by `close()`: a timeout (also when a chunk got too few marks, as
with a blocking `wait()`), extra marks, a peer that failed the round, or a round fence that a peer
never reached. Once an error is recorded, later stream-ordered waits return at once instead of each
spinning for the timeout, and the rank's next round fence marks the next round failed on its peers:
their `wait()` for it fails at once rather than timing out on that rank. The rank still waits at the
fence for its peers, so its next producer cannot overwrite a peer that is still reading, but only
for the plan's timeout, and not again for a peer it already timed out on. Past that point its copies
can land in the peer's buffer while the peer still reads, so a fence that times out on a peer also
marks the peer's next round failed: the peer's next `wait()` raises instead of accepting the data,
though what it was reading may already be overwritten. Do not rely on the fence for a peer that may
be slower than the timeout.

### Destinations, peers and groups

By default every chunk goes to every other rank and lands where it is, in their copy of the
buffer: an all-gather or a broadcast. Three options cover other patterns:

- **Destination.** `TriggeredPlan(..., destination=staging)` makes peers' chunks land in another
  symmetric tensor, and `Region(dst_offset_bytes=, dst_stride_bytes=)` places each region in it
  at its own offset and row pitch. The schedule builders take `dst=` (a view of the destination
  with the same shape as the source) together with `dst_base=` (the destination itself).
- **Peers.** `Chunk(peers=...)` sends a chunk to only some ranks; ranks it does not go to neither
  receive nor wait for it: there, `arrived()` and `wait()` on it return true at once. A chunk is
  never sent to its own rank.
- **Groups.** Chunks with the same `group` (a small integer chosen by the caller, shared across
  ranks) signal one arrival counter at each receiver, so a consumer waits for all of them at once
  with `wait_group(group)` instead of one `wait` per chunk. `wait(chunk)` on a chunk in a group
  waits for its whole group. Number groups from 0: each id up to the largest gets a counter.

A reduce-scatter's exchange: each rank sends block `p` of its partial result to rank `p` only,
into inbox slot `rank` there, and the reduction waits once for every source:

```python
nbytes = block * partial.element_size()
schedule = [
    Chunk((Region(p * nbytes, nbytes, dst_offset_bytes=rank * nbytes),), peers=(p,), group=0)
    for p in range(world_size)
    if p != rank
]
plan = TriggeredPlan(ctx, partial, schedule, destination=inbox)

@triton.jit
def reduce(inbox, ..., handle):
    TriggeredView.initialize(handle).wait_group(0)  # every other rank's block for this rank
    ...                                             # sum the inbox slots
```

An all-gather into a staging buffer with its own row pitch, for a GEMM that reads it as its `A`:

```python
mine = a_local                                      # (M, K) on the heap, this rank's rows
dst = staging[rank * M : (rank + 1) * M, :K]       # this rank's rows, in the staging pitch
schedule = tile_chunks(mine, (BM, BK), num_programs, dst=dst, dst_base=staging)
plan = TriggeredPlan(ctx, a_local, schedule, destination=staging)
...                                                 # each round, after the producer writes a_local:
dst.copy_(a_local)                                  # a chunk is never sent to its own rank
```

`ctx.ccl` also prepares in-place scatter, gather, broadcast and all-to-all plans (`prepare_scatter`,
`prepare_gather`, `prepare_broadcast`, `prepare_all_to_all`). A kernel driving any of them needs no
roles: it marks the chunks its rank `owns()` and waits for every chunk, since `wait()` returns at
once for a chunk not sent to it.

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
   Once the next round is prepared (`prepare_next()`), extra marks send its copies before the check
   can stop them.
   A kernel that is not written against `TriggeredView` may mark chunks ready itself; see
   [Driving the counters from another kernel](#driving-the-counters-from-another-kernel).
3. **A producer never waits on its own plan's arrivals before it has marked ready everything it
   writes.** Its wait could depend on a copy queued behind one of its own unfinished chunks.
   `wait_sent` waits for every chunk this rank sends, so it may only be called once all of them
   are marked ready; inside the producer, that means every writer CTA must be running at the same
   time (a persistent kernel), or the wait never ends.
4. **Region sizes are fixed when the plan is created.**
5. **`arrived`, `wait`, `sent` and `wait_sent` are called uniformly across the CTA:** every wave
   makes the same number of calls. Each is one atomic whose result reaches the other waves through
   a CTA barrier, so a polling loop must not exit on a per-wave condition (such as a clock read):
   waves would read the result before it is written, or deadlock. Count polls instead.
6. **`next_round()` means "done reading this round's data".** Its round fence is a kernel on the
   current stream: once every rank's stream has passed it, a peer may start the next round, and
   its copies can overwrite the buffer at any time. Work queued on that stream before the call is
   covered (work on other streams is not); nothing issued after it may read the buffer until the
   next `wait()`. The fence uses the plan's own counters and gives up after the plan's timeout,
   recording which rank never reached it.

Plans that share queues must start in the order they were prepared. While copies are prepared
on a queue, host `Iris.put()` and `quiet()` to that peer raise rather than queue behind them; a
round's copies stop counting once a blocking `wait()` has seen them land, or at `next_round()`.

A chunk marked ready by programs that did not write all of its regions is sent before its data is
written. The mark count cannot catch this (the count is right, the tiles are wrong), which is why
`order` should come from the kernel's own mapping.

Creating a plan prepares on every rank or on none: if one rank cannot (for example its queue is
full, or its schedule is invalid), every rank raises, and a failed construction flushes what it
prepared. The plan holds its queue space from then on, so preparing later rounds
(`prepare_next()` or `next_round()`) does not involve the other ranks. If it fails on one rank
anyway, that rank still takes part in the round fence and tells its peers there; it then raises,
and so does its peers' `wait()` for that round, rather than accept what the failed rank's
`close()` flushes.

`close()` is collective and waits for every rank, since a peer's copies and polls can still touch
a rank's buffer and gates after its own work is done. Prepared copies cannot be withdrawn from the
queue, so closing a plan in the middle of a round (or before it starts) sends each rank's chunks
as they stand. Leaving a `with` block closes the plan, including on an exception.

Plans can be closed in any order, so `with` blocks nest. A plan whose copies are queued behind an
earlier plan's waiting copies cannot finish first: its `close()` returns with the plan `CLOSING`,
and closing the earlier plan finishes both.

## Memory ordering

**Producer.** Copy engines read memory behind the GPU's L2, so a chunk's data must be written
back to memory before its gate opens. Each writer does that with a release on the chunk's counter,
and the writers may run on different XCDs, each with its own L2. A release only orders the stores
of the thread that performs it, and `mark_ready` performs it on one thread for the whole CTA, so
the other waves' stores are ordered before it with a CTA barrier that waits for every wave's
stores, as HIP's `__syncthreads()` does. The increment that brings the counter to
`writers × round` comes after every other writer's increment, each of which followed that writer's
write-back, so the copy engine sees every writer's data once it sees the count:

```python
wait_cnt()                 # s_waitcnt vmcnt(0): this wave's stores are done
tl.debug_barrier()         # every wave has got here, so every store of the CTA is done
tl.atomic_add(counter, 1, sem="release", scope="sys")   # write back this XCD's L2, then count
```

Recent Triton versions already wait for outstanding stores in `tl.debug_barrier()`; `wait_cnt()`
keeps the barrier a memory barrier on versions where it only synchronizes execution. Without the
barrier, the copy engine sends stale data for most chunks.

**Consumer.** `arrived` / `wait` read the arrival counter, and `sent` / `wait_sent` the completion
counters, with an acquire at system scope; data read after it in the same kernel is fresh.

### Driving the counters from another kernel

`plan.counters` is a contiguous int32 tensor with one counter per chunk id; adding 1 to a chunk's
counter is all `mark_ready` does. A kernel that signals per tile, such as a GEMM epilogue, can
therefore take this rank's counters (`plan.counters[plan.chunk_id(rank, 0):]`, indexed by the chunk's
position in the schedule) and increment them itself. The same rules apply: one increment per writer
per round, by the whole CTA after every wave's stores are done, with release semantics.

- **Scope.** A `gpu` (agent) scope release is enough on gfx942 and gfx950 (MI300X, MI350X): there it
  also writes back the XCD's L2 (`buffer_wbl2 sc1`; `sys` adds `sc0`), so the copy engine sees the
  data. Both were measured clean on both chips. Use `sys`, as `mark_ready` does, on other GPUs.
- **Barrier.** "After every wave's stores are done" needs `s_waitcnt vmcnt(0)` before the CTA
  barrier. `tl.debug_barrier()` emits it only on Triton versions that include triton-lang/triton#9025
  (Iris's pinned Triton does); on older ones it is a bare `s_barrier`, and the kernel must wait for
  its stores itself (`iris.mem.utils.wait_cnt()`), as `mark_ready` does.
- **Chunk numbering.** The kernel must increment the counter of the chunk that holds the tile it
  just wrote. tritonBLAS's `EpilogueConfig(counter_buffer=...)` indexes counters by tile position,
  not by the step at which the producer finishes the tile, so it only lines up with some schedules:
  `map_type="identity"` with `tile_chunks(view, block, 1)` in the default row-major order, and
  `map_type="row"` with `chunks(view, num_pid_m, writers=num_pid_n)` when M is a multiple of
  BLOCK_M. `launch_wave` counts by program id, which does not change in a persistent kernel. With a
  swizzled `order`, a tile's counter belongs to another chunk: the chunks are sent before their
  tiles are written, and the mark count cannot tell. The row-major pairings also mean a swizzled
  GEMM finishes chunks out of order, which blocks each queue behind its slowest chunk.

## Gates and rounds

- The gates are one int32 tensor on the symmetric heap: a counter and an arrival counter per
  chunk, a completion counter per peer, the round fence's counters, then an arrival counter per
  group. The copy engine polls a chunk's counter with `>=` against `writers × round`, so no writer
  has to tell which mark is the last one. A group's counter grows by the number of its chunks sent
  to the rank, per round. Since several senders add to it, it counts rounds correctly only because
  the round fence keeps every rank's round `r` copies ahead of any round `r + 1` copy.
- Every round, a rank's copies on each queue to a peer end with one signal on the sending rank's
  completion counter for that peer. `wait()` and `sent()` / `wait_sent()` read it: once it reaches
  the round times the number of queues (lanes) the rank's chunks use to that peer, every copy to
  that peer has landed, and the source may be overwritten.
- Rounds start at 1 and add 1 per `next_round()`. Counters are never reset, so a plan is limited
  to rounds where `writers × round` (and a group's count × round) stays at most 2³¹ − 1: over
  7 million rounds at 304 writers.
- `wait()` reads only this rank's gates: the completion counter of every peer it sends to, and the
  arrival counter of every chunk sent to it.

## Lanes

A rank's copies to one peer run in prepare order, so a chunk ready early waits behind earlier ones.
`sdma_lanes=N` gives each peer N independent host queues, and `Chunk.lane` (or `lane=` on the
builders) picks one; lane `i` is host queue (`channel`) `i`. Lanes share a copy engine, so they
remove blocking but add no bandwidth.

```python
ctx = iris.iris(heap_size, engines=("sdma",), sdma_lanes=2)
schedule = chunks(first, 4, base=out, lane=0) + chunks(second, 4, base=out, lane=1)
```

## Requirements and limits

- **SDMA engine.** Create the context with `engines=("sdma",)` (needs `pip install iris[sdma]`).
- **One queue per peer and lane.** Plans use the host-initiated copy-engine queue to each peer, shared with
  host `Iris.put()` and with the plans of every Iris context in the process. Kernel-initiated
  copies (`put(..., use_copy_engine=True)`) use a separate queue and are unaffected.
- **Ring capacity.** Prepared copies may use at most half of each queue's 8 MiB ring, and each plan
  holds room for one round of its copies (two with `prepare_ahead`). Copies hold their ring space
  until their gates open, and a submit that does not fit waits for space, so a full ring could hang
  the host before any kernel opens a gate. The other half covers wrap padding, work queued before
  them, and earlier rounds still draining. Half is a conservative margin, not a measured bound. A
  loop that never synchronizes can run ahead of the GPU; preparing then waits for earlier rounds'
  copies to drain, which they do, since their producers are already queued.
- **No two senders write the same bytes.** On each rank, the destination bytes that different
  senders' chunks land on must not overlap; when the destination is the buffer itself, they must
  also miss the bytes that rank sends. The plan checks this when it is created, which catches a
  schedule built from the whole buffer instead of the rank's own part. `destination` must be the
  buffer itself or memory apart from it.
- **Single node, or one rank per host over RDMA** (see [RDMA](rdma-gda.md#triggered-plans-over-rdma)).
- **Linear regions are at most 1 GiB** (`COPY_LINEAR`'s byte count is 30 bits on SDMA 4.4). The
  schedule builders split larger ranges.

## Not supported yet

- **Sending a chunk to its own rank.** A rank's own part of the result does not go through the
  copy engine; copy it locally if the destination is another tensor.
