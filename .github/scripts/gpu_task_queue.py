#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Run a job's GPU tasks concurrently, taking GPUs per task instead of per job.

The runner has fewer job slots (4) than GPUs (8), and a job that is waiting for
GPUs still holds its slot. With one GPU request per job, a 2-GPU job plus three
waiting 8-GPU jobs fill every slot and leave 6 GPUs idle while the small jobs
that could use them sit in GitHub's queue. Bundling several tasks into one job
and allocating per task lets any single job fill the node, and hands a task's
GPUs back the moment it finishes.

Usage:
    gpu_task_queue.py --install git unittests:8 unittests:4 unittests:2 unittests:1
    gpu_task_queue.py --install editable new_examples:8 new_examples:2
    gpu_task_queue.py --task 8 "GEMM All-Scatter" "bash .github/scripts/run_perf_benchmark.sh ..."

A positional task is <test_dir>:<num_ranks>. "new_examples" runs
run_new_examples.sh; anything else runs run_tests.sh on tests/<test_dir>.

--task RANKS NAME COMMAND runs COMMAND with bash, for jobs that are not a test
directory. Every task runs in a private copy of the checkout with GPU_DEVICES
set to its GPUs, and must pass them on (container_exec.sh --gpus).

Scheduling: largest task first, smaller tasks backfill whatever GPUs are free.
If the largest pending task has waited RESERVE_AFTER seconds, it posts a
reservation that stops every job (and the bash allocator) from starting new
work until it fits, so 8-GPU tasks cannot be starved by a stream of small ones.
A larger waiting task takes over a smaller one's reservation.

Shares state with gpu_allocator.sh: the same bitmap file and flock lock file.
"""

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

STATE_FILE = os.environ.get("GPU_STATE_FILE", "/tmp/iris_gpu_state")
LOCK_FILE = STATE_FILE + ".lock"
RESERVE_FILE = STATE_FILE + ".reserve"
MAX_GPUS = int(os.environ.get("MAX_GPUS", "8"))
POLL = float(os.environ.get("GPU_QUEUE_POLL", "5"))
RESERVE_AFTER = float(os.environ.get("GPU_QUEUE_RESERVE_AFTER", "300"))
# A reservation not refreshed for this long belongs to a dead job.
RESERVE_TTL = 600
HELD_FILE = os.environ.get(
    "GPU_QUEUE_HELD_FILE",
    os.path.join(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()), "iris_gpu_queue_held"),
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OWNER = "{}-{}-{}".format(os.environ.get("GITHUB_RUN_ID", "local"), os.environ.get("RUNNER_NAME", "host"), os.getpid())


def log(msg):
    print("[GPU-QUEUE] " + msg, flush=True)


class Task:
    def __init__(self, ranks, name, test_dir=None, install=None, shell=None):
        if not ranks.isdigit() or not 1 <= int(ranks) <= MAX_GPUS:
            raise SystemExit("bad task '{}': ranks must be 1-{}, got '{}'".format(name, MAX_GPUS, ranks))
        self.test_dir = test_dir
        self.ranks = int(ranks)
        self.install = install
        self.shell = shell
        self.name = name
        self.proc = None
        self.mask = 0
        self.gpus = ""
        self.tmpdir = None
        self.workdir = None
        self.logpath = None
        self.queued_at = time.time()
        self.started_at = None
        self.ended_at = None
        self.rc = None

    @classmethod
    def from_spec(cls, spec, install):
        """A positional <test_dir>:<ranks> task."""
        test_dir, _, ranks = spec.rpartition(":")
        if not test_dir or not ranks.isdigit() or not 1 <= int(ranks) <= MAX_GPUS:
            raise SystemExit("bad task '{}': expected <test_dir>:<1-{}>".format(spec, MAX_GPUS))
        return cls(ranks, "{} ({} ranks, {})".format(test_dir, ranks, install), test_dir=test_dir, install=install)

    def command(self):
        if self.shell is not None:
            return ["bash", "-c", self.shell]
        scripts = os.path.join(self.workdir, ".github", "scripts")
        if self.test_dir == "new_examples":
            return ["bash", os.path.join(scripts, "run_new_examples.sh"), str(self.ranks), self.install]
        return ["bash", os.path.join(scripts, "run_tests.sh"), self.test_dir, str(self.ranks), self.gpus, self.install]


class Allocator:
    """Same protocol as gpu_allocator.sh: decimal bitmap, bit N set = GPU N busy."""

    def __enter__(self):
        self.fd = open(LOCK_FILE, "a")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()

    def bitmap(self):
        try:
            with open(STATE_FILE) as f:
                value = int(f.read().strip())
            return value if 0 <= value < (1 << MAX_GPUS) else 0
        except (OSError, ValueError):
            return 0

    def write_bitmap(self, value):
        with open(STATE_FILE, "w") as f:
            f.write("{}\n".format(value))

    def reservation(self):
        """(owner, ranks) of a live reservation, or None."""
        try:
            with open(RESERVE_FILE) as f:
                owner, ranks, stamp = f.read().split()
            if time.time() - float(stamp) < RESERVE_TTL:
                return owner, int(ranks)
        except (OSError, ValueError):
            pass
        return None

    def reserve(self, ranks):
        with open(RESERVE_FILE, "w") as f:
            f.write("{} {} {}\n".format(OWNER, ranks, time.time()))

    def unreserve(self):
        res = self.reservation()
        if res and res[0] == OWNER:
            os.remove(RESERVE_FILE)


class Queue:
    def __init__(self, tasks):
        # Largest first; stable, so ties keep the order given on the command line.
        self.pending = sorted(tasks, key=lambda t: -t.ranks)
        self.running = []
        self.done = []
        self.head_waiting_since = None
        self.reserved_for = None

    def held_mask(self):
        mask = 0
        for t in self.running:
            mask |= t.mask
        return mask

    def record_held(self):
        # Read by the workflow's always() step, so GPUs are returned even if this
        # process is killed outright. Written under the allocator lock: after the
        # bitmap when acquiring, before it when releasing, so a kill in between can
        # leak GPUs (gpu-reset.yml) but never free ones another job now holds.
        with open(HELD_FILE, "w") as f:
            f.write("{}\n".format(self.held_mask()))

    def try_start(self):
        to_launch = []
        with Allocator() as alloc:
            bitmap = alloc.bitmap()
            res = alloc.reservation()
            foreign = res is not None and res[0] != OWNER
            if self.reserved_for is not None and (res is None or foreign):
                # Taken over by a larger request (or expired); queue behind it.
                self.reserved_for = None
            head = self.pending[0] if self.pending else None
            for task in list(self.pending):
                if foreign:
                    break
                if self.reserved_for is not None and task is not self.reserved_for:
                    continue
                free = [g for g in range(MAX_GPUS) if not (bitmap >> g) & 1]
                if len(free) < task.ranks:
                    continue
                for g in free[: task.ranks]:
                    task.mask |= 1 << g
                bitmap |= task.mask
                task.gpus = ",".join(str(g) for g in free[: task.ranks])
                self.pending.remove(task)
                self.running.append(task)
                to_launch.append(task)
                if task is self.reserved_for:
                    alloc.unreserve()
                    self.reserved_for = None
            alloc.write_bitmap(bitmap)
            self.record_held()

            # Anti-starvation: track how long the largest pending task has waited.
            # The head only changes when it launches, since nothing is added later.
            if head is None or head in to_launch:
                self.head_waiting_since = None
            else:
                if self.head_waiting_since is None:
                    self.head_waiting_since = time.time()
                waited = time.time() - self.head_waiting_since
                if self.reserved_for is head:
                    alloc.reserve(head.ranks)  # refresh
                elif self.reserved_for is None and waited >= RESERVE_AFTER and (not foreign or head.ranks > res[1]):
                    # A larger request takes over a smaller one's reservation: a small
                    # task often waits only because its own siblings fill the node,
                    # and must not lock out the 8-GPU task that backfill would starve.
                    alloc.reserve(head.ranks)
                    self.reserved_for = head
                    log(
                        "{} waited {:.0f}s; reserving the node until {} GPUs are free".format(
                            head.name, waited, head.ranks
                        )
                    )

        for task in to_launch:
            self.launch(task)

    def launch(self, task):
        # Private copy of the checkout: editable and pip installs write build
        # output into the source tree, which concurrent tasks would trample.
        # .git comes along because setuptools-scm needs it.
        task.tmpdir = tempfile.mkdtemp(prefix="iris-task-", dir=os.environ.get("RUNNER_TEMP"))
        task.workdir = os.path.join(task.tmpdir, "src")
        task.logpath = os.path.join(task.tmpdir, "task.log")
        shutil.copytree(WORKSPACE, task.workdir, symlinks=True, ignore=shutil.ignore_patterns("iris_overlay_*"))
        env = dict(os.environ, GPU_DEVICES=task.gpus)
        task.started_at = time.time()
        log("start {} on GPUs {} (waited {:.0f}s)".format(task.name, task.gpus, task.started_at - task.queued_at))
        with open(task.logpath, "w") as logf:
            task.proc = subprocess.Popen(
                task.command(), cwd=task.workdir, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
            )

    def reap(self):
        finished = [t for t in self.running if t.proc.poll() is not None]
        if not finished:
            return
        with Allocator() as alloc:
            bitmap = alloc.bitmap()
            for t in finished:
                bitmap &= ~t.mask
                self.running.remove(t)
            self.record_held()
            alloc.write_bitmap(bitmap)
        for t in finished:
            t.rc = t.proc.returncode
            t.ended_at = time.time()
            self.done.append(t)
            self.report(t)
            shutil.rmtree(t.tmpdir, ignore_errors=True)

    def report(self, t):
        mark = "✅" if t.rc == 0 else "❌"
        print(
            "::group::{} {} -- {:.0f}s on GPUs {}".format(mark, t.name, t.ended_at - t.started_at, t.gpus), flush=True
        )
        try:
            with open(t.logpath, errors="replace") as f:
                shutil.copyfileobj(f, sys.stdout)
        except OSError as e:
            print("(log unavailable: {})".format(e))
        print("::endgroup::", flush=True)
        if t.rc != 0:
            print("::error::{} failed with exit code {}".format(t.name, t.rc), flush=True)

    def shutdown(self):
        # GitHub follows a cancel's SIGINT with SIGTERM 7.5s later; don't let the
        # second one abort the cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        launched = [t for t in self.running if t.proc is not None]
        for t in launched:
            try:
                os.killpg(t.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 5
        for t in launched:
            try:
                t.proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(t.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        with Allocator() as alloc:
            mask = self.held_mask()
            stopped, self.running = self.running, []
            self.record_held()
            alloc.write_bitmap(alloc.bitmap() & ~mask)
            alloc.unreserve()
        # Show what the stopped tasks had printed: on a job timeout this is the
        # only record of where they were stuck.
        for t in launched:
            t.rc = t.proc.returncode
            t.ended_at = time.time()
            self.report(t)
        for t in stopped:
            if t.tmpdir:
                shutil.rmtree(t.tmpdir, ignore_errors=True)

    def run(self):
        last_status = time.time()
        while self.pending or self.running:
            self.reap()
            if self.pending:
                self.try_start()
            if time.time() - last_status >= 300:
                last_status = time.time()
                log(
                    "running: {}; pending: {}".format(
                        ", ".join(t.name for t in self.running) or "-", ", ".join(t.name for t in self.pending) or "-"
                    )
                )
            time.sleep(POLL)
        self.summary()
        return 0 if all(t.rc == 0 for t in self.done) else 1

    def summary(self):
        rows = ["| Task | GPUs | Waited | Ran | Result |", "|---|---|---|---|---|"]
        for t in self.done:
            rows.append(
                "| {} | {} | {:.0f}s | {:.0f}s | {} |".format(
                    t.name,
                    t.gpus,
                    t.started_at - t.queued_at,
                    t.ended_at - t.started_at,
                    "pass" if t.rc == 0 else "FAIL ({})".format(t.rc),
                )
            )
        print("\n".join(rows), flush=True)
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a") as f:
                f.write("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", choices=["git", "editable", "install"], help="required for TEST_DIR:RANKS tasks")
    parser.add_argument(
        "--task",
        nargs=3,
        action="append",
        default=[],
        metavar=("RANKS", "NAME", "COMMAND"),
        help="run COMMAND with bash on RANKS GPUs; repeatable",
    )
    parser.add_argument("tasks", nargs="*", metavar="TEST_DIR:RANKS")
    args = parser.parse_args()
    if not args.tasks and not args.task:
        parser.error("no tasks given")
    if args.tasks and not args.install:
        parser.error("--install is required for TEST_DIR:RANKS tasks")

    tasks = [Task(ranks, name, shell=command) for ranks, name, command in args.task]
    tasks += [Task.from_spec(spec, args.install) for spec in args.tasks]
    queue = Queue(tasks)

    def on_signal(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        return queue.run()
    finally:
        if queue.running:
            log("stopping {} running task(s) and returning their GPUs".format(len(queue.running)))
        queue.shutdown()


if __name__ == "__main__":
    sys.exit(main())
