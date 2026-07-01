"""Extract raw run output into flat typed rows, using only the stdlib and perfetto.
Each `parse_<source>` reads one source into its typed `<Source>Raw` essential facts;
`to_row` collapses those into the long-format `Row`s written to data.csv. Every view
(busy/idle, category totals, comms, compositions, A/B deltas) is derived from these
rows downstream, never stored. Run as a script over a bundle's data/ to build data.csv
(`python preprocess.py`, printing progress); the report notebook then just reads it."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# The reduction a row reports over a distribution. None (the common case) means the row
# is a direct measured value — a throughput, a count, a de-nested self-time, a track
# window — not a statistic. median == p50, so we keep `median`.
Statistic = Literal["mean", "median", "p99", "std"]
# The closed set of units a value can carry. A count says what it counts (tokens /
# requests / calls); a rate is a count over a second. The count/duration/rate split is a
# grouping of these (below) — kept in code, not a column (the unit determines it).
Unit = Literal["us", "ms", "s", "tokens", "requests", "calls", "tokens_per_s", "requests_per_s"]
DURATION_UNITS = frozenset({"us", "ms", "s"})
COUNT_UNITS = frozenset({"tokens", "requests", "calls"})
RATE_UNITS = frozenset({"tokens_per_s", "requests_per_s"})

# The grain of a row: what one row represents, named after the GROUP BY that produced it
# (dbt's term for a fact table's level of detail). Orthogonal to `statistic`: that reduces
# the value distribution (mean/p99), grain reduces the dimensions. "run" = a whole vLLM
# run; "kernel" = one profiler kernel; "track" = one (pid,tid) trace track (its window);
# "track_slice" = one (track, category, name) group (its self time + count); "track_pair" =
# one pair of tracks (their concurrent-busy overlap, the second track in pid2/tid2).
Grain = Literal["run", "kernel", "track", "track_slice", "track_pair"]


@dataclass(frozen=True)
class Row:
    """One essential fact for data.csv. `arm`/`container`/`command` are the caller-supplied
    arm dimensions stamped on every row (the run arm, its A/B variant, its group);
    `rank`/`pid`/`tid`/`cat`/`name` locate the entity (pid/tid = the track, cat = chrome
    category, name = op/kernel/metric/track); `grain` says what one row represents (its
    GROUP BY level), telling e.g. a track window apart from a slice self-time; `statistic`
    is the distribution reduction (mean/median/p99/std) or None for a direct value; `unit`
    the dimension. The notebook derives every view from these; none are stored."""

    # per-file: what the helpers build from
    rank: Optional[int]
    pid: Optional[int]
    tid: Optional[int]
    cat: Optional[str]
    name: str
    grain: Grain
    statistic: Optional[Statistic]
    unit: Unit
    value: Union[int, float, str]
    preprocessor: str
    source_file: str
    # the per-arm identity, stamped on every row by to_row (constant across the arm)
    arm: str = ""
    container: Optional[str] = None  # arm dimension: the A/B variant (baseline/exp/...)
    command: Optional[str] = None  # arm dimension: the group (bench/profile)
    # the SECOND track of a grain="track_pair" row (its overlap partner); None otherwise
    pid2: Optional[int] = None
    tid2: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --- vLLM ----------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmLatency:
    """The distribution stats vLLM reports for one latency metric (ms). None = absent."""

    mean: Optional[float]
    median: Optional[float]
    p99: Optional[float]
    std: Optional[float]


@dataclass(frozen=True)
class VllmRaw:
    """The essential vLLM --save-result facts (a typed mirror of the modeled fields).
    parse_vllm is the loss boundary: a JSON field not named here is dropped at parse."""

    output_throughput: Optional[float]
    total_token_throughput: Optional[float]
    request_throughput: Optional[float]
    ttft: VllmLatency  # time to first token
    tpot: VllmLatency  # time per output token
    num_prompts: Optional[int]
    completed: Optional[int]
    duration: Optional[float]
    total_output_tokens: Optional[int]


def parse_vllm(text: str) -> VllmRaw:
    """Read the modeled fields out of the --save-result JSON (the loss boundary)."""
    d = json.loads(text)

    def num(key: str) -> Optional[float]:
        v = d.get(key)
        return float(v) if v is not None else None

    def count(key: str) -> Optional[int]:
        v = d.get(key)
        return int(v) if v is not None else None

    def latency(metric: str) -> VllmLatency:
        return VllmLatency(
            num(f"mean_{metric}_ms"), num(f"median_{metric}_ms"), num(f"p99_{metric}_ms"), num(f"std_{metric}_ms")
        )

    return VllmRaw(
        output_throughput=num("output_throughput"),
        total_token_throughput=num("total_token_throughput"),
        request_throughput=num("request_throughput"),
        ttft=latency("ttft"),
        tpot=latency("tpot"),
        num_prompts=count("num_prompts"),
        completed=count("completed"),
        duration=num("duration"),
        total_output_tokens=count("total_output_tokens"),
    )


def _vllm_rows(parsed: VllmRaw, rank: Optional[int], source_file: str) -> List[Row]:
    """The e2e metrics — aggregate over the run, so no rank/track/category. Throughputs,
    token/prompt counts and duration are direct values (statistic None); the latencies are
    durations read out as mean/median/p99/std. (arm/container/command stamped by to_row.)"""
    rows: List[Row] = []

    def add(value: Optional[Union[int, float]], name: str, statistic: Optional[Statistic], unit: Unit) -> None:
        if value is not None:
            rows.append(Row(None, None, None, None, name, "run", statistic, unit, value, "parse_vllm", source_file))

    def add_latency(name: str, lat: VllmLatency) -> None:
        add(lat.mean, name, "mean", "ms")
        add(lat.median, name, "median", "ms")
        add(lat.p99, name, "p99", "ms")
        add(lat.std, name, "std", "ms")

    add(parsed.output_throughput, "output_throughput", None, "tokens_per_s")
    add(parsed.total_token_throughput, "total_throughput", None, "tokens_per_s")
    add(parsed.request_throughput, "request_throughput", None, "requests_per_s")
    add_latency("ttft", parsed.ttft)
    add_latency("tpot", parsed.tpot)
    add(parsed.num_prompts, "num_prompts", None, "requests")
    add(parsed.completed, "completed", None, "requests")
    add(parsed.duration, "duration", None, "s")
    add(parsed.total_output_tokens, "output_tokens", None, "tokens")
    return rows


# --- torch profiler key_averages table -----------------------------------------------


@dataclass(frozen=True)
class ProfileKernel:
    """One kernel row of the torch key_averages table — the columns we model (us)."""

    name: str
    calls: int
    cuda_time_avg_us: float
    self_cuda_us: float


@dataclass(frozen=True)
class ProfileRaw:
    """The essential kernels of a torch profiler key_averages table — a per-rank, per-name
    aggregate (the cheap fallback when full traces aren't kept). parse_profile is the loss
    boundary: unmodeled columns (Self CPU, CPU total, ...) are dropped at parse."""

    kernels: List[ProfileKernel]


def parse_profile(text: str) -> ProfileRaw:
    """Read each kernel's Name / # of Calls / CUDA time avg / Self CUDA out of the
    key_averages table (the loss boundary). Empty if the table can't be located."""
    lines = text.splitlines()
    dash_idxs = [i for i, ln in enumerate(lines) if _is_dash_row(ln)]
    if len(dash_idxs) < 2:
        return ProfileRaw([])
    top, mid = dash_idxs[0], dash_idxs[1]
    bottom = dash_idxs[2] if len(dash_idxs) >= 3 else len(lines)
    spans = _column_spans(lines[top])
    header_line = lines[mid - 1] if mid - 1 > top else lines[top + 1]
    col = {name: i for i, name in enumerate(_table_cells(header_line, spans))}
    i_name = col.get("Name", -1)
    i_self = col.get("Self CUDA", -1)
    i_avg = col.get("CUDA time avg", -1)
    i_calls = col.get("# of Calls", -1)
    kernels: List[ProfileKernel] = []
    if min(i_name, i_self, i_avg, i_calls) >= 0:
        for ln in lines[mid + 1 : bottom]:
            if not ln.strip() or _is_dash_row(ln):
                continue
            cells = _table_cells(ln, spans)
            if len(cells) <= max(i_name, i_self, i_avg, i_calls):
                continue
            name = cells[i_name]
            if not name:
                continue
            kernels.append(
                ProfileKernel(
                    name, _table_to_count(cells[i_calls]), _table_to_us(cells[i_avg]), _table_to_us(cells[i_self])
                )
            )
    return ProfileRaw(kernels)


def _profile_rows(parsed: ProfileRaw, rank: Optional[int], source_file: str) -> List[Row]:
    """Per kernel (rank-level, no track): call count, the mean per-call duration, and the
    exclusive self time (statistic None — it's a direct value). cat is None: the profiler
    table doesn't carry chrome categories. (arm/container/command stamped by to_row.)"""
    rows: List[Row] = []
    for k in parsed.kernels:
        rows.append(Row(rank, None, None, None, k.name, "kernel", None, "calls", k.calls, "parse_profile", source_file))
        rows.append(
            Row(
                rank, None, None, None, k.name, "kernel", "mean", "us", k.cuda_time_avg_us, "parse_profile", source_file
            )
        )
        rows.append(
            Row(rank, None, None, None, k.name, "kernel", None, "us", k.self_cuda_us, "parse_profile", source_file)
        )
    return rows


# --- perfetto trace -------------------------------------------------------------------

# Per (track, category, slice-name): exclusive self time (dur minus direct children via
# parent_id, so it's de-nested and non-overlapping) + count. This is the essential trace
# fact — busy/idle, category totals, comms and compositions all derive from it.
SLICE_FACTS_SQL = (
    "SELECT p.pid AS pid, t.tid AS tid, t.name AS tname, s.category AS cat, s.name AS name, "
    "COUNT(*) AS calls, "
    "SUM(s.dur - COALESCE(ch.child_dur, 0)) / 1000.0 AS self_us "
    "FROM slice s "
    "JOIN thread_track tt ON s.track_id = tt.id "
    "JOIN thread t ON tt.utid = t.utid "
    "LEFT JOIN process p ON t.upid = p.upid "
    "LEFT JOIN (SELECT parent_id, SUM(dur) AS child_dur FROM slice "
    "           WHERE dur > 0 AND parent_id IS NOT NULL GROUP BY parent_id) ch "
    "  ON ch.parent_id = s.id "
    "WHERE s.dur >= 0 "
    "GROUP BY p.pid, t.tid, s.category, s.name"
)

# Per (pid, tid) track: its wall span (first slice to last). idle = window - busy, both
# derived in the notebook (busy = the track's summed self time).
TRACK_WINDOW_SQL = (
    "SELECT p.pid AS pid, t.tid AS tid, t.name AS tname, "
    "(MAX(s.ts + s.dur) - MIN(s.ts)) / 1000.0 AS window_us "
    "FROM slice s "
    "JOIN thread_track tt ON s.track_id = tt.id "
    "JOIN thread t ON tt.utid = t.utid "
    "LEFT JOIN process p ON t.upid = p.upid "
    "WHERE s.dur >= 0 "
    "GROUP BY t.utid"
)

# Each GPU kernel slice's track + raw [ts, ts+dur) interval (nanoseconds), for the
# cross-track CONCURRENCY sweep. Concurrency is a track property (within a track slices are
# serial), so ALL cross-track interaction is captured EXACTLY by pairwise TRACK overlap:
# sweeping these intervals sums, for each pair of tracks, the wall both were busy. Because
# a track's slices never overlap each other, track overlap == the sum of slice-pairwise
# overlaps (we keep the exact duration, drop only per-slice identity). Name/role-agnostic:
# preprocess makes no comms/compute distinction - it emits the track-pair overlaps, and the
# notebook runs whatever overlap query it wants over them.
KERNEL_INTERVALS_SQL = (
    "SELECT p.pid AS pid, t.tid AS tid, s.ts AS ts, s.dur AS dur "
    "FROM slice s "
    "JOIN thread_track tt ON s.track_id = tt.id "
    "JOIN thread t ON tt.utid = t.utid "
    "LEFT JOIN process p ON t.upid = p.upid "
    "WHERE s.category = 'kernel' AND s.dur > 0"
)


@dataclass(frozen=True)
class TraceSlice:
    """One (track, category, name) group's de-nested self time + count."""

    pid: int
    tid: int
    tname: str
    cat: str
    name: str
    calls: int
    self_us: float


@dataclass(frozen=True)
class TraceTrack:
    """One (pid, tid) track's wall span (first slice to last)."""

    pid: int
    tid: int
    tname: str
    window_us: float


@dataclass(frozen=True)
class TracePair:
    """The exact wall two (pid, tid) tracks were busy CONCURRENTLY (canonical order,
    track a < track b) - the concurrency fact. Within a track slices are serial, so all
    cross-track overlap lives here; the notebook runs arbitrary overlap queries over it."""

    pid_a: int
    tid_a: int
    pid_b: int
    tid_b: int
    overlap_us: float


@dataclass(frozen=True)
class TraceRaw:
    """The essential trace facts: per-(track, category, name) self/count, per-track windows,
    and pairwise track overlaps (concurrency). busy/idle, category totals, comms,
    compositions and exposed/overlapped all DERIVE from these. Empty if perfetto is
    unavailable or the trace won't load."""

    slices: List[TraceSlice]
    tracks: List[TraceTrack]
    overlaps: List[TracePair]


_Track = Tuple[int, int]


def _track_pair_overlaps(intervals: List[Tuple[int, int, int, int]]) -> Dict[Tuple[_Track, _Track], int]:
    """Sweep kernel (pid, tid, ts, dur) intervals (nanoseconds) across tracks and sum, for
    each PAIR of tracks, the wall both were busy. Returns overlap ns keyed by
    (track_a, track_b) in canonical order (a < b). Exact: within a track slices are serial,
    so all cross-track interaction is captured by these pairwise overlaps."""
    events = []  # (time, delta, track)
    for pid, tid, ts, dur in intervals:
        track = (pid, tid)
        events.append((ts, 1, track))
        events.append((ts + dur, -1, track))
    events.sort(key=lambda e: e[0])
    active: Dict[_Track, int] = {}
    live: set[_Track] = set()
    overlaps: Dict[Tuple[_Track, _Track], int] = {}
    prev_t: Optional[int] = None
    for t, delta, track in events:
        if prev_t is not None and t > prev_t and len(live) >= 2:
            seg = t - prev_t
            ordered = sorted(live)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    key = (ordered[i], ordered[j])
                    overlaps[key] = overlaps.get(key, 0) + seg
        active[track] = active.get(track, 0) + delta
        if active[track] > 0:
            live.add(track)
        else:
            live.discard(track)
        prev_t = t
    return overlaps


def parse_trace(trace_path: str) -> TraceRaw:
    """Perfetto read of one *.pt.trace.json.gz into the essential facts. Empty TraceRaw
    if perfetto is unavailable or the trace won't load."""
    try:
        from perfetto.trace_processor import TraceProcessor
    except ImportError:
        return TraceRaw([], [], [])
    try:
        tp = TraceProcessor(trace=str(trace_path))
    except Exception:
        return TraceRaw([], [], [])
    try:
        slices: List[TraceSlice] = []
        for r in tp.query(SLICE_FACTS_SQL):
            if r.tid is None or r.cat is None:
                continue
            pid = int(r.pid) if r.pid is not None else -1
            slices.append(
                TraceSlice(pid, int(r.tid), r.tname or "", r.cat, r.name or "", int(r.calls), float(r.self_us))
            )
        kintervals: List[Tuple[int, int, int, int]] = []
        for r in tp.query(KERNEL_INTERVALS_SQL):
            if r.tid is None:
                continue
            pid = int(r.pid) if r.pid is not None else -1
            kintervals.append((pid, int(r.tid), int(r.ts), int(r.dur)))
        overlaps: List[TracePair] = []
        for (a, b), ns in _track_pair_overlaps(kintervals).items():
            overlaps.append(TracePair(a[0], a[1], b[0], b[1], ns / 1000.0))
        tracks: List[TraceTrack] = []
        for r in tp.query(TRACK_WINDOW_SQL):
            if r.tid is None:
                continue
            pid = int(r.pid) if r.pid is not None else -1
            tracks.append(TraceTrack(pid, int(r.tid), r.tname or "", float(r.window_us or 0.0)))
    except Exception:
        return TraceRaw([], [], [])
    finally:
        tp.close()
    return TraceRaw(slices, tracks, overlaps)


def _trace_rows(parsed: TraceRaw, rank: Optional[int], source_file: str) -> List[Row]:
    """The essential facts: per (track, category, name) self time + count (grain
    "track_slice"), each track's window (grain "track"), and each track-pair's concurrent
    overlap (grain "track_pair", partner in pid2/tid2). All direct values (statistic None);
    busy/idle, comms, composition and exposed/overlapped derive. `grain` keeps them apart.
    (arm/container/command stamped by to_row.)"""
    rows: List[Row] = []
    for s in parsed.slices:
        rows.append(
            Row(
                rank,
                s.pid,
                s.tid,
                s.cat,
                s.name,
                "track_slice",
                None,
                "us",
                round(s.self_us, 1),
                "parse_trace",
                source_file,
            )
        )
        rows.append(
            Row(rank, s.pid, s.tid, s.cat, s.name, "track_slice", None, "calls", s.calls, "parse_trace", source_file)
        )
    for t in parsed.tracks:
        rows.append(
            Row(
                rank,
                t.pid,
                t.tid,
                None,
                t.tname,
                "track",
                None,
                "us",
                round(t.window_us, 1),
                "parse_trace",
                source_file,
            )
        )
    for o in parsed.overlaps:
        rows.append(
            Row(
                rank,
                o.pid_a,
                o.tid_a,
                None,
                "overlap",
                "track_pair",
                None,
                "us",
                round(o.overlap_us, 1),
                "parse_trace",
                source_file,
                pid2=o.pid_b,
                tid2=o.tid_b,
            )
        )
    return rows


# --- project any parsed source into rows ----------------------------------------------

Parsed = Union[VllmRaw, ProfileRaw, TraceRaw]


def to_row(
    parsed: Parsed,
    arm: str,
    rank: Optional[int],
    source_file: str,
    container: Optional[str] = None,
    command: Optional[str] = None,
) -> List[Row]:
    """Collapse a parsed source's essential facts into flat cache rows — the single entry
    over the union of parse outputs. The helpers build each row from per-file context
    (rank, source_file); to_row then STAMPS the per-arm identity (arm, container, command)
    on all of them, so the three are applied uniformly in one place."""
    if isinstance(parsed, VllmRaw):
        rows = _vllm_rows(parsed, rank, source_file)
    elif isinstance(parsed, ProfileRaw):
        rows = _profile_rows(parsed, rank, source_file)
    elif isinstance(parsed, TraceRaw):
        rows = _trace_rows(parsed, rank, source_file)
    else:
        raise TypeError("to_row: unhandled parsed type %r" % type(parsed).__name__)
    return [dataclasses.replace(r, arm=arm, container=container, command=command) for r in rows]


# --- per-arm chaining -----------------------------------------------------------------


def preprocess(
    arm: str,
    vllm_files: List[str],
    profile_files: List[str],
    trace_files: List[str],
    container: Optional[str] = None,
    command: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Row]:
    """Essential-fact rows for one arm from its source files. In-memory: reads exactly the
    files handed in, chaining parse_X -> to_row per file. `container`/`command` are the
    arm's declared dimensions (caller-supplied, like `arm`); to_row stamps all three on
    every row. source_file is the path as handed in (the caller owns it), so a row traces
    back to its file. `progress(path)` is called before each file (trace parsing is slow, so
    the caller can report which file it's on)."""

    def note(f: str) -> None:
        if progress is not None:
            progress(f)

    rows: List[Row] = []
    for f in vllm_files:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_vllm(fh.read()), arm, None, f, container, command))
    for f in profile_files:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_profile(fh.read()), arm, _rank_of(f), f, container, command))
    for f in trace_files:
        note(f)
        rows.extend(to_row(parse_trace(f), arm, _rank_of(f), f, container, command))
    return rows


def _rank_of(path: str) -> Optional[int]:
    """rankN from 'rank3.123.pt.trace.json.gz' or 'profiler_out_3.txt' -> 3; None if none."""
    import re

    base = os.path.basename(path)
    m = re.search(r"rank(\d+)", base) or re.search(r"profiler_out_(\d+)", base)
    return int(m.group(1)) if m else None


def _is_dash_row(line: str) -> bool:
    s = line.strip()
    return len(s) > 0 and set(s) <= {"-", " "}


def _column_spans(dash_line: str) -> List[Any]:
    spans: List[Any] = []
    start = None
    for i, ch in enumerate(dash_line):
        if ch == "-":
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(dash_line)))
    return spans


def _table_cells(line: str, spans: List[Any]) -> List[str]:
    out: List[str] = []
    for idx, (s, _e) in enumerate(spans):
        end = len(line) if idx == len(spans) - 1 else spans[idx + 1][0]
        out.append(line[s:end].strip())
    return out


def _table_to_us(token: str) -> float:
    t = token.strip()
    if not t:
        return 0.0
    if t.endswith("us"):
        return float(t[:-2])
    if t.endswith("ms"):
        return float(t[:-2]) * 1_000.0
    if t.endswith("s"):
        return float(t[:-1]) * 1_000_000.0
    return float(t)


def _table_to_count(token: str) -> int:
    t = token.strip().replace(",", "")
    return int(float(t)) if t else 0


# --- run as a script: parse arm output dirs -> data.csv (the visible analysis step) ------


def _arm_meta(arm_dir: str) -> Dict[str, Any]:
    """container + command for an arm dir, from its arm.json (written by _serve.sh's
    dump_arm). Absent -> (None, None): rows still parse, but the report's container/command
    filters won't match, so a run that followed the instructions always has arm.json."""
    p = os.path.join(arm_dir, "arm.json")
    if os.path.exists(p):
        with open(p) as fh:
            j = json.load(fh)
        return {"container": j.get("container"), "command": j.get("command")}
    return {"container": None, "command": None}


def main(arm_dirs: List[str], out: str = "data.csv") -> int:
    """Parse each arm output dir into `out`, printing progress. This is THE analysis step -
    the heavy one (perfetto reads the GB traces); the report notebook just reads the CSV.
    Each dir is one arm (e.g. output/exp-profile/) holding arm.json + results/ + profile/,
    as the run scripts produce it. With no dirs given, defaults to output/*. eval arms are
    skipped (their results are gsm8k, not used by the report)."""
    import csv
    import glob

    if not arm_dirs:
        arm_dirs = sorted(glob.glob(os.path.join("output", "*")))
    rows: List[Row] = []
    n_arms = 0
    for d in arm_dirs:
        if not os.path.isdir(d):
            continue
        arm = os.path.basename(d.rstrip("/"))
        meta = _arm_meta(d)
        if meta["command"] == "eval":
            print(f"[{arm}] skipped (eval arm; gsm8k results are not in the report)", flush=True)
            continue
        vllm = glob.glob(os.path.join(d, "results", "*.json"))
        profile = glob.glob(os.path.join(d, "profile", "summary", "profiler_out_*.txt"))
        trace = glob.glob(os.path.join(d, "profile", "traces", "*.pt.trace.json.gz"))
        print(
            f"[{arm}] container={meta['container']} command={meta['command']}: "
            f"{len(vllm)} results, {len(profile)} profiler, {len(trace)} traces",
            flush=True,
        )
        rows.extend(
            preprocess(
                arm,
                vllm,
                profile,
                trace,
                container=meta["container"],
                command=meta["command"],
                progress=lambda f: print("    parsing " + os.path.basename(f), flush=True),
            )
        )
        n_arms += 1
    if not rows:
        print(
            "preprocess: no rows produced (no arm dirs matched, or they were empty). "
            "Pass the arm dirs, e.g. `python preprocess.py output/*`.",
            flush=True,
        )
        return 1
    fields = list(rows[0].to_dict().keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
    print(f"preprocess: wrote {out} ({len(rows)} rows across {n_arms} arms)", flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Parse arm output dirs into data.csv (the analysis step); the report "
        "notebook then just reads data.csv."
    )
    ap.add_argument(
        "arm_dirs",
        nargs="*",
        metavar="ARM_DIR",
        help="arm output dirs, each holding arm.json + results/ + profile/ (default: output/*)",
    )
    ap.add_argument("--out", default="data.csv", help="output CSV path (default: data.csv)")
    args = ap.parse_args()
    sys.exit(main(args.arm_dirs, args.out))
