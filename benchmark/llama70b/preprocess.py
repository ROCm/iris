"""Extract raw run output into flat long-format rows (data.csv), using only the stdlib and
perfetto. parse_<source> reads a source into its <Source>Raw facts; to_row(parsed, prov)
projects those into Rows. Run as a script over a bundle: python preprocess.py arms.json."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# The distribution reduction a row reports (None = a direct measured value).
Statistic = Literal["mean", "median", "p99", "std"]

# The unit a value carries, grouped for the notebook.
Unit = Literal["us", "ms", "s", "tokens", "requests", "calls", "tokens_per_s", "requests_per_s", "fraction"]
DURATION_UNITS = frozenset({"us", "ms", "s"})
COUNT_UNITS = frozenset({"tokens", "requests", "calls"})
RATE_UNITS = frozenset({"tokens_per_s", "requests_per_s"})

# What one row represents (its GROUP BY level).
Grain = Literal["run", "kernel", "track", "track_slice", "track_pair", "eval_sample"]

# The closed vocabulary of measured quantities a row can carry.
Variable = Literal[
    "output_throughput",
    "total_throughput",
    "request_throughput",  # vllm e2e
    "ttft",
    "tpot",
    "num_prompts",
    "completed",
    "duration",
    "output_tokens",
    "calls",
    "cuda_time_avg",
    "self_cuda",
    "self_cpu",  # profiler
    "self_time",
    "window",
    "overlap",  # trace
    "exact_match_strict",
    "exact_match_flexible",
    "correct",  # eval
]


@dataclass(frozen=True)
class Provenance:
    """A row's origin: the arm identity (arm/dockerfile/scripts/env) + the file trace
    (preprocessor/source_file). All required; Row.to_dict serializes it to the flat CSV columns."""

    arm: str
    dockerfile: str  # relative to the reproducer dir (Dockerfile.baseline)
    scripts: List[str]  # relative to the reproducer dir (./bench.sh)
    env: Dict[str, str]  # the operating point; empty = no overrides
    preprocessor: str  # which parse_* produced the row
    source_file: str  # the file it was parsed from


@dataclass(frozen=True)
class Row:
    """One flat long-format row: a parsed FACT (variable + value + unit/statistic) plus the
    ENTITY it is about (grain + id-vars rank/pid/tid/cat/name) plus its PROVENANCE. `name` is
    None for a run-level row. pid2/tid2 (track_pair partner) and text (eval sample output) are
    populated only for those grains."""

    variable: Variable
    value: float
    unit: Unit
    statistic: Optional[Statistic]
    grain: Grain
    rank: Optional[int]
    pid: Optional[int]
    tid: Optional[int]
    cat: Optional[str]
    name: Optional[str]
    prov: Provenance
    pid2: Optional[int] = None
    tid2: Optional[int] = None
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to the data.csv columns; prov is serialized here (script joined by '+',
        env as 'k=v;k=v')."""
        return {
            "variable": self.variable,
            "value": self.value,
            "unit": self.unit,
            "statistic": self.statistic,
            "grain": self.grain,
            "rank": self.rank,
            "pid": self.pid,
            "tid": self.tid,
            "cat": self.cat,
            "name": self.name,
            "arm": self.prov.arm,
            "dockerfile": self.prov.dockerfile,
            "script": _script_label(self.prov.scripts),
            "env": _env_label(self.prov.env),
            "preprocessor": self.prov.preprocessor,
            "source_file": self.prov.source_file,
            "pid2": self.pid2,
            "tid2": self.tid2,
            "text": self.text,
        }


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
    """The modeled vLLM --save-result facts; parse_vllm drops any JSON field not named here."""

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


def _vllm_rows(parsed: VllmRaw, prov: Provenance) -> List[Row]:
    """Run-level e2e metrics (name=None): throughputs/counts/duration as direct values,
    ttft/tpot read out as mean/median/p99/std."""
    rows: List[Row] = []

    def add(value: Optional[Union[int, float]], variable: Variable, statistic: Optional[Statistic], unit: Unit) -> None:
        if value is not None:
            rows.append(Row(variable, value, unit, statistic, "run", None, None, None, None, None, prov))

    def add_latency(variable: Variable, lat: VllmLatency) -> None:
        add(lat.mean, variable, "mean", "ms")
        add(lat.median, variable, "median", "ms")
        add(lat.p99, variable, "p99", "ms")
        add(lat.std, variable, "std", "ms")

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
    """One key_averages row (us). self_cpu_us is the device-vs-host tell: 0 = a device kernel,
    >0 = a host op (whose CUDA total double-counts its child kernel). The notebook keeps
    device kernels only, via the self_cpu variable."""

    name: str
    calls: int
    cuda_time_avg_us: float
    self_cuda_us: float
    self_cpu_us: float


@dataclass(frozen=True)
class ProfileRaw:
    """One rank's key_averages kernels (rank from the filename). parse_profile drops unmodeled columns."""

    kernels: List[ProfileKernel]
    rank: Optional[int]


def parse_profile(text: str, rank: Optional[int] = None) -> ProfileRaw:
    """Read each kernel's Name / # of Calls / CUDA time avg / Self CUDA / Self CPU out of the
    key_averages table (the loss boundary), for the given `rank` (the dump's GPU, from its
    filename). Empty (but rank-tagged) if the table can't be located."""
    lines = text.splitlines()
    dash_idxs = [i for i, ln in enumerate(lines) if _is_dash_row(ln)]
    if len(dash_idxs) < 2:
        return ProfileRaw([], rank)
    top, mid = dash_idxs[0], dash_idxs[1]
    bottom = dash_idxs[2] if len(dash_idxs) >= 3 else len(lines)
    spans = _column_spans(lines[top])
    header_line = lines[mid - 1] if mid - 1 > top else lines[top + 1]
    col = {name: i for i, name in enumerate(_table_cells(header_line, spans))}
    i_name = col.get("Name", -1)
    i_self = col.get("Self CUDA", -1)
    i_avg = col.get("CUDA time avg", -1)
    i_calls = col.get("# of Calls", -1)
    i_selfcpu = col.get("Self CPU", -1)  # the device-vs-host tell (0 => device kernel)
    kernels: List[ProfileKernel] = []
    if min(i_name, i_self, i_avg, i_calls) >= 0:
        for ln in lines[mid + 1 : bottom]:
            if not ln.strip() or _is_dash_row(ln):
                continue
            cells = _table_cells(ln, spans)
            if len(cells) <= max(i_name, i_self, i_avg, i_calls, i_selfcpu):
                continue
            name = cells[i_name]
            if not name:
                continue
            self_cpu = _table_to_us(cells[i_selfcpu]) if i_selfcpu >= 0 else 0.0
            kernels.append(
                ProfileKernel(
                    name,
                    _table_to_count(cells[i_calls]),
                    _table_to_us(cells[i_avg]),
                    _table_to_us(cells[i_self]),
                    self_cpu,
                )
            )
    return ProfileRaw(kernels, rank)


def _profile_rows(parsed: ProfileRaw, prov: Provenance) -> List[Row]:
    """One row per (kernel, variable): calls, cuda_time_avg (per-call mean), self_cuda, self_cpu.
    Entity is the kernel name; rank from parsed; cat None (no chrome category)."""
    rank = parsed.rank
    rows: List[Row] = []
    for k in parsed.kernels:
        rows.append(Row("calls", k.calls, "calls", None, "kernel", rank, None, None, None, k.name, prov))
        rows.append(
            Row("cuda_time_avg", k.cuda_time_avg_us, "us", "mean", "kernel", rank, None, None, None, k.name, prov)
        )
        rows.append(Row("self_cuda", k.self_cuda_us, "us", None, "kernel", rank, None, None, None, k.name, prov))
        rows.append(Row("self_cpu", k.self_cpu_us, "us", None, "kernel", rank, None, None, None, k.name, prov))
    return rows


# --- perfetto trace -------------------------------------------------------------------

# Per (track, category, name): de-nested exclusive self time (dur minus direct children) + count.
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

# Per (pid, tid) track: its wall span (first slice to last).
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

# Each GPU kernel slice's track + [ts, ts+dur) interval (ns), for the cross-track concurrency
# sweep (see _track_pair_overlaps).
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
    """The wall two (pid, tid) tracks were busy concurrently (canonical order, a < b)."""

    pid_a: int
    tid_a: int
    pid_b: int
    tid_b: int
    overlap_us: float


@dataclass(frozen=True)
class TraceRaw:
    """One rank's trace facts (rank from the filename): per-slice self/count, per-track windows,
    pairwise track overlaps. Empty (but rank-tagged) if perfetto is unavailable."""

    slices: List[TraceSlice]
    tracks: List[TraceTrack]
    overlaps: List[TracePair]
    rank: Optional[int]


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
    """Perfetto read of one *.pt.trace.json.gz into the essential facts, tagged with the trace's
    GPU `rank` (from its filename). Empty (but rank-tagged) if perfetto is unavailable or the
    trace won't load."""
    rank = _rank_of(trace_path)
    try:
        from perfetto.trace_processor import TraceProcessor
    except ImportError:
        return TraceRaw([], [], [], rank)
    # tp is closed in finally on every path; the guard skips close if TraceProcessor() raised.
    tp = None
    try:
        tp = TraceProcessor(trace=str(trace_path))
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
        return TraceRaw(slices, tracks, overlaps, rank)
    except Exception:
        return TraceRaw([], [], [], rank)
    finally:
        if tp is not None:
            tp.close()


def _trace_rows(parsed: TraceRaw, prov: Provenance) -> List[Row]:
    """One row per fact: self_time + calls per slice (grain track_slice), window per track
    (grain track), overlap per pair (grain track_pair, partner in pid2/tid2). rank from parsed;
    name is None for a pair or an unnamed track/slice."""
    rank = parsed.rank
    rows: List[Row] = []
    for s in parsed.slices:
        rows.append(
            Row(
                "self_time",
                round(s.self_us, 1),
                "us",
                None,
                "track_slice",
                rank,
                s.pid,
                s.tid,
                s.cat,
                s.name or None,
                prov,
            )
        )
        rows.append(
            Row("calls", s.calls, "calls", None, "track_slice", rank, s.pid, s.tid, s.cat, s.name or None, prov)
        )
    for t in parsed.tracks:
        rows.append(
            Row("window", round(t.window_us, 1), "us", None, "track", rank, t.pid, t.tid, None, t.tname or None, prov)
        )
    for o in parsed.overlaps:
        rows.append(
            Row(
                "overlap",
                round(o.overlap_us, 1),
                "us",
                None,
                "track_pair",
                rank,
                o.pid_a,
                o.tid_a,
                None,
                None,
                prov,
                pid2=o.pid_b,
                tid2=o.tid_b,
            )
        )
    return rows


# --- lm_eval harness results (the correctness gate: gsm8k accuracy) -------------------


@dataclass(frozen=True)
class EvalSample:
    """One lm_eval per-doc record (question, gold target, model output, exact_match 0/1).
    _eval_rows projects each to a grain=eval_sample Row so a wrong arm's output can be eyeballed."""

    question: str
    target: str
    output: str
    correct: float  # exact_match for this doc (1.0 right, 0.0 wrong)


# lm_eval names a gsm8k score "<metric>,<filter>"; resolve the external key to our closed vocab
# here (dropping _stderr/alias). Add an eval task by extending this map and the Variable union.
EvalMetric = Literal["exact_match_strict", "exact_match_flexible"]
_EVAL_METRIC: Dict[str, EvalMetric] = {
    "exact_match,strict-match": "exact_match_strict",
    "exact_match,flexible-extract": "exact_match_flexible",
}


@dataclass(frozen=True)
class EvalRaw:
    """One lm_eval run: `tasks` = task -> resolved EvalMetric -> score, `samples` = the
    --log_samples per-doc records. Both from parse_eval (two files, one type)."""

    tasks: Dict[str, Dict[EvalMetric, float]] = field(default_factory=dict)
    samples: List[EvalSample] = field(default_factory=list)


def parse_eval(results_text: str = "", samples_text: str = "") -> EvalRaw:
    """Parse an lm_eval run: results JSON (scores) and/or --log_samples JSONL (per-doc records);
    either may be empty. Metric keys are resolved via _EVAL_METRIC (unmodeled keys dropped); an
    unparseable sample line is skipped."""
    tasks: Dict[str, Dict[EvalMetric, float]] = {}
    if results_text.strip():
        results = json.loads(results_text).get("results", {})
        if isinstance(results, dict):
            for task, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                nums: Dict[EvalMetric, float] = {
                    _EVAL_METRIC[k]: float(v)
                    for k, v in metrics.items()
                    if k in _EVAL_METRIC and isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if nums:
                    tasks[task] = nums

    samples: List[EvalSample] = []
    for line in samples_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        doc = d.get("doc") or {}
        question = doc.get("question") if isinstance(doc, dict) else None
        resp = d.get("filtered_resps") or d.get("resps") or []
        output = resp[0] if isinstance(resp, list) and resp else resp
        em = d.get("exact_match")
        samples.append(
            EvalSample(
                question=str(question if question is not None else doc),
                target=str(d.get("target", "")),
                output=str(output if output is not None else ""),
                correct=float(em) if isinstance(em, (int, float)) and not isinstance(em, bool) else 0.0,
            )
        )
    return EvalRaw(tasks=tasks, samples=samples)


def _eval_rows(parsed: EvalRaw, prov: Provenance) -> List[Row]:
    """Scores -> grain=run rows (variable=EvalMetric, cat=task); samples -> grain=eval_sample
    rows (variable=correct, name=question, text=output). Eval is run-aggregate, so no rank."""
    rows: List[Row] = []
    for task, metrics in parsed.tasks.items():
        for metric, value in metrics.items():
            rows.append(Row(metric, value, "fraction", None, "run", None, None, None, task, None, prov))
    for s in parsed.samples:
        rows.append(
            Row(
                "correct",
                s.correct,
                "fraction",
                None,
                "eval_sample",
                None,
                None,
                None,
                None,
                s.question,
                prov,
                text=s.output,
            )
        )
    return rows


# --- project any parsed source into rows ----------------------------------------------

Parsed = Union[VllmRaw, ProfileRaw, TraceRaw, EvalRaw]


def to_row(parsed: Parsed, prov: Provenance) -> List[Row]:
    """A row is a parsed fact plus its provenance. Dispatch on the parsed type to its projection;
    rank (profiler/trace) rides on the parsed type, not a separate arg."""
    if isinstance(parsed, VllmRaw):
        return _vllm_rows(parsed, prov)
    elif isinstance(parsed, ProfileRaw):
        return _profile_rows(parsed, prov)
    elif isinstance(parsed, TraceRaw):
        return _trace_rows(parsed, prov)
    elif isinstance(parsed, EvalRaw):
        return _eval_rows(parsed, prov)
    raise TypeError("to_row: unhandled parsed type %r" % type(parsed).__name__)


# --- per-arm chaining -----------------------------------------------------------------


def preprocess(
    arm: str,
    dockerfile: str,
    scripts: List[str],
    env: Dict[str, str],
    vllm_files: List[str],
    profile_files: List[str],
    trace_files: List[str],
    eval_files: Optional[List[str]] = None,
    eval_sample_files: Optional[List[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Row]:
    """Rows for one arm from its source files: parse_X -> to_row per file, building each file's
    Provenance from the arm identity + the parser + the path. `progress(path)` is called per file."""

    def note(f: str) -> None:
        if progress is not None:
            progress(f)

    def prov(preprocessor: str, source_file: str) -> Provenance:
        return Provenance(arm, dockerfile, scripts, env, preprocessor, source_file)

    rows: List[Row] = []
    for f in vllm_files:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_vllm(fh.read()), prov("parse_vllm", f)))
    for f in profile_files:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_profile(fh.read(), _rank_of(f)), prov("parse_profile", f)))
    for f in trace_files:
        note(f)
        rows.extend(to_row(parse_trace(f), prov("parse_trace", f)))
    for f in eval_files or []:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_eval(results_text=fh.read()), prov("parse_eval", f)))
    for f in eval_sample_files or []:
        note(f)
        with open(f) as fh:
            rows.extend(to_row(parse_eval(samples_text=fh.read()), prov("parse_eval", f)))
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


# --- run as a script: read the manifest -> the analysis CSV (the visible step) -----------

# The analysis CSV name (the notebook reads it from here). One output; eval samples are
# grain="eval_sample" rows in it, not a second file.
OUTPUT_CSV = "data.csv"
# The analysis manifest: a JSON list of arm entries, one per arm, each an object:
#   {"dir": "data/<label>", "label": "<label>", "dockerfile": "Dockerfile.baseline",
#    "scripts": ["./bench.sh"], "env": {"WORKLOAD": "decode64", ...}}
MANIFEST = "arms.json"


@dataclass(frozen=True)
class ArmManifestEntry:
    """One arms.json entry, typed: the arm's raw-data `dir` + declared provenance. `dockerfile`
    is required, `scripts` non-empty, `env` may be empty."""

    dir: str
    label: str
    dockerfile: str
    scripts: List[str]
    env: Dict[str, str]


def parse_manifest(text: str) -> List[ArmManifestEntry]:
    """Parse arms.json into typed entries (the analysis input boundary): JSON in, ArmManifestEntry
    list out, ValueError on a bad entry."""
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"arms.json must be a JSON list of arm entries, got {type(data).__name__}")
    return [_parse_manifest_entry(e, i) for i, e in enumerate(data)]


def _parse_manifest_entry(obj: Any, i: int) -> ArmManifestEntry:
    """Validate one arms.json entry. `dir` required; `label` defaults to the dir basename;
    `dockerfile` a non-empty string, `scripts` a non-empty list, `env` a string->string dict
    (may be empty). ValueError (with the entry index) on any missing or bad field."""
    where = f"arms.json[{i}]"
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object, got {type(obj).__name__}")
    d = obj.get("dir")
    if not isinstance(d, str) or not d:
        raise ValueError(f"{where}.dir must be a non-empty string")
    label = obj.get("label") or os.path.basename(d.rstrip("/"))
    if not isinstance(label, str) or not label:
        raise ValueError(f"{where}.label must be a non-empty string")
    dockerfile = obj.get("dockerfile")
    if not isinstance(dockerfile, str) or not dockerfile:
        raise ValueError(f"{where}.dockerfile must be a non-empty string (every arm builds an image)")
    scripts = obj.get("scripts")
    if not isinstance(scripts, list) or not scripts or not all(isinstance(s, str) for s in scripts):
        raise ValueError(f"{where}.scripts must be a non-empty list of strings")
    env = obj.get("env")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError(f"{where}.env must be a string->string object")
    return ArmManifestEntry(dir=d, label=label, dockerfile=dockerfile, scripts=list(scripts), env=dict(env))


def _script_label(scripts: List[str]) -> str:
    """The arm's script series as the CSV `script` value, joined by `+`."""
    return "+".join(scripts)


def _env_label(env: Dict[str, str]) -> str:
    """An arm's env dict as the CSV `env` value — `k=v;k=v`, keys sorted (stable)."""
    return ";".join(f"{k}={v}" for k, v in sorted(env.items()))


# One folder per output kind under an arm dir; the folder name picks the parser. Shared by the
# CSV build (main) and --validate. eval nests under a model subdir, so it's globbed recursively.
def _vllm_files(d: str) -> List[str]:
    return glob.glob(os.path.join(d, "vllm", "*.json"))


def _profile_summary_files(d: str) -> List[str]:
    return glob.glob(os.path.join(d, "profile", "summary", "profiler_out_*.txt"))


def _trace_files(d: str) -> List[str]:
    return glob.glob(os.path.join(d, "profile", "traces", "*.pt.trace.json.gz"))


def _eval_files(d: str) -> List[str]:
    return glob.glob(os.path.join(d, "eval", "**", "results*.json"), recursive=True)


def _eval_sample_files(d: str) -> List[str]:
    # lm_eval --log_samples writes samples_<task>_<ts>.jsonl next to results*.json.
    return glob.glob(os.path.join(d, "eval", "**", "samples*.jsonl"), recursive=True)


# --validate parses each file (the same parse the CSV build does) to catch a corrupt one, then
# discards the rows. No arm context here, so a labeled placeholder provenance.
def _validation_prov(preprocessor: str, source_file: str) -> Provenance:
    return Provenance(
        arm="(validation)",
        dockerfile="(validation)",
        scripts=[],
        env={},
        preprocessor=preprocessor,
        source_file=source_file,
    )


def _rows_vllm(f: str) -> List[Row]:
    with open(f) as fh:
        return to_row(parse_vllm(fh.read()), _validation_prov("parse_vllm", f))


def _rows_profile(f: str) -> List[Row]:
    with open(f) as fh:
        return to_row(parse_profile(fh.read(), _rank_of(f)), _validation_prov("parse_profile", f))


def _rows_eval(f: str) -> List[Row]:
    with open(f) as fh:
        return to_row(parse_eval(fh.read()), _validation_prov("parse_eval", f))


# Kind -> (label, top dir, files locator, parse-one-file-to-rows). None for traces: too heavy
# for a fail-fast gate (perfetto over GBs), so they are presence-checked only.
_ParseOne = Optional[Callable[[str], List[Row]]]
_KINDS: List[Tuple[str, str, Callable[[str], List[str]], _ParseOne]] = [
    ("vllm result", "vllm", _vllm_files, _rows_vllm),
    ("profiler tables", "profile/summary", _profile_summary_files, _rows_profile),
    ("eval result", "eval", _eval_files, _rows_eval),
    ("traces", "profile/traces", _trace_files, None),
]


def validate_arm(arm_dir: str) -> List[str]:
    """Validate one arm's raw output; return a list of problems (empty = OK). Each present file
    is PARSED (parsing is the validation, so a corrupt result fails here, not just a missing one);
    traces are presence-only. A kind folder present but empty, or a file that won't parse / yields
    no rows, is a problem; no output at all means the arm died. Prints the gsm8k score for eval."""
    problems: List[str] = []
    produced = False
    for label, topdir, locate, rows_of in _KINDS:
        files = locate(arm_dir)
        if not files:
            if os.path.isdir(os.path.join(arm_dir, topdir)):
                problems.append(f"{topdir}/ exists but is empty ({label} missing) - workload did not complete")
            continue
        produced = True
        if rows_of is None:  # traces: presence only, not parsed
            print(f"  {label}: {len(files)} file(s) (presence only)", flush=True)
            continue
        n_rows, parse_failed = 0, False
        for f in files:
            try:
                n_rows += len(rows_of(f))
            except Exception as exc:  # noqa: BLE001 - any parse failure = invalid output
                problems.append(f"{label}: {os.path.basename(f)} did not parse ({exc})")
                parse_failed = True
        if not parse_failed and n_rows == 0:
            problems.append(f"{label}: files present but no usable data parsed")
        print(f"  {label}: {len(files)} file(s), {n_rows} row(s)", flush=True)
        if label == "eval result" and not parse_failed:
            _print_eval_score(files)
    if not produced and not problems:
        problems.append(f"no output under {arm_dir} - workload did not run or died before writing")
    return problems


def _print_eval_score(files: List[str]) -> None:
    """Print the gsm8k metrics from an eval arm's latest lm_eval result (log provenance)."""
    latest = sorted(files)[-1]
    try:
        with open(latest) as fh:
            raw = parse_eval(fh.read())
    except (OSError, ValueError) as exc:
        print(f"    (could not read {latest}: {exc})", flush=True)
        return
    for task, metrics in raw.tasks.items():
        for k, v in sorted(metrics.items()):
            print(f"    {task} {k}: {v:.4f}", flush=True)


def main(manifest: str = MANIFEST, out: str = OUTPUT_CSV) -> int:
    """Read arms.json and parse each arm's raw data/ into `out` (the CSV the notebook reads).
    Under each arm dir, one folder per output kind picks the parser (vllm/, profile/, eval/)."""
    import csv

    def prog(f: str) -> None:
        print("    parsing " + os.path.basename(f), flush=True)

    with open(manifest) as fh:
        entries = parse_manifest(fh.read())

    rows: List[Row] = []
    n_arms = 0
    for entry in entries:
        d = entry.dir
        arm = entry.label
        if not os.path.isdir(d):
            print(f"[{arm}] skipped (no dir {d})", flush=True)
            continue
        # Parse whatever folders the arm produced (the folder name picks the parser).
        vllm = _vllm_files(d)
        profile = _profile_summary_files(d)
        trace = _trace_files(d)
        ev = _eval_files(d)
        evs = _eval_sample_files(d)
        print(
            f"[{arm}] dockerfile={entry.dockerfile} script={_script_label(entry.scripts)} "
            f"env={_env_label(entry.env)}: {len(vllm)} vllm, {len(profile)} profiler, "
            f"{len(trace)} traces, {len(ev)} eval, {len(evs)} eval-samples",
            flush=True,
        )
        rows.extend(
            preprocess(
                arm,
                entry.dockerfile,
                entry.scripts,
                entry.env,
                vllm,
                profile,
                trace,
                eval_files=ev,
                eval_sample_files=evs,
                progress=prog,
            )
        )
        n_arms += 1
    if not rows:
        print(f"preprocess: no rows produced (no arm dirs in {manifest} matched, or they were empty).", flush=True)
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
        description="Read the analysis manifest (arms.json) into the analysis CSV (the analysis "
        "step); the report notebook then just reads the CSV."
    )
    ap.add_argument(
        "manifest",
        nargs="?",
        default=MANIFEST,
        metavar="ARMS_JSON",
        help=f"analysis manifest: a JSON list of {{dir, label, dockerfile, scripts, "
        f"env}} arm entries (default: {MANIFEST})",
    )
    ap.add_argument("--out", default=OUTPUT_CSV, help=f"output CSV path (default: {OUTPUT_CSV})")
    ap.add_argument(
        "--validate",
        metavar="ARM_DIR",
        help="validate ONE arm's raw output (presence/completeness) and exit; "
        "non-zero if the arm produced nothing or a partial result. Cheap "
        "(no trace parsing) - run.sh calls this per-arm to fail fast.",
    )
    args = ap.parse_args()
    if args.validate is not None:
        print(f"[validate] {args.validate}", flush=True)
        problems = validate_arm(args.validate)
        for p in problems:
            print(f"[validate] FATAL: {p}", file=sys.stderr, flush=True)
        if problems:
            sys.exit(1)
        print("[validate] ok", flush=True)
        sys.exit(0)
    sys.exit(main(args.manifest, args.out))
