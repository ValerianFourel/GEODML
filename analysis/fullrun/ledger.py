"""A finite task ledger on a shared filesystem, worked through by any number of allocations.

Layout of a ledger folder:
  tasks.jsonl            written once by ``plan``; one task per line: id, stage, argv, deps, cores, est_cpu_h, gpu
  claims/<id>.json       created with O_EXCL by the worker that runs the task (job id, host, pid, time)
  done/<id>.json         exit 0: finished (seconds, job, commit)
  checkpoint/<id>.jsonl  exit 4: deadline checkpoint (finished units saved); the claim is released, any worker resumes it
  failed/<id>.jsonl      any other exit; the claim is released; after MAX_FAILURES the task is no longer claimed
  released/<id>.<n>.json claims of ended allocations, released by ``reconcile`` after Slurm shows the allocation terminal
  logs/<id>.<job>.log    stdout and stderr of each attempt

Claims never expire on their own (AGENTS.md: network failure never expires ownership; reconcile first).
Argument placeholders filled by the worker: {PY} (its own interpreter), {WORKERS} (cores given to the task),
{MINUTES} (minutes before the worker's deadline), {JOB} (Slurm job id).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

MAX_FAILURES = 2
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


@dataclass
class Task:
    id: str
    stage: str
    argv: list
    deps: list
    cores: int = 1          # 0 = every core the worker has
    est_cpu_h: float = 0.1
    gpu: bool = False
    gpus: int = 0           # GPU tasks: devices the task needs (0 with gpu=True means all four: the embedding tasks)


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        for sub in ("claims", "done", "checkpoint", "failed", "released", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- tasks
    def write_tasks(self, tasks: list[Task]) -> None:
        path = self.root / "tasks.jsonl"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}: a ledger is planned once (start a new ledger folder instead)")
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        known = set(ids)
        for t in tasks:
            missing = set(t.deps) - known
            if missing:
                raise ValueError(f"{t.id} depends on unknown tasks {sorted(missing)}")
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t.__dict__) + "\n" for t in tasks))
        tmp.replace(path)

    def tasks(self) -> list[Task]:
        return [Task(**json.loads(line)) for line in (self.root / "tasks.jsonl").read_text().splitlines() if line.strip()]

    # ---------------------------------------------------------------- state
    def done(self, tid: str) -> bool:
        return (self.root / "done" / f"{tid}.json").exists()

    def claimed(self, tid: str) -> bool:
        return (self.root / "claims" / f"{tid}.json").exists()

    def failures(self, tid: str) -> int:
        path = self.root / "failed" / f"{tid}.jsonl"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def state(self, task: Task) -> str:
        if self.done(task.id):
            return "done"
        if self.claimed(task.id):
            return "running"
        if self.failures(task.id) >= MAX_FAILURES:
            return "failed"
        if all(self.done(d) for d in task.deps):
            return "ready"
        return "waiting"

    def claim(self, task: Task, job: str) -> bool:
        path = self.root / "claims" / f"{task.id}.json"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as stream:
            json.dump({"job": job, "host": socket.gethostname(), "pid": os.getpid(), "claimed_at": time.time()}, stream)
        if self.done(task.id):  # finished between the scan and the claim
            path.unlink()
            return False
        return True

    def _append(self, folder: str, tid: str, record: dict) -> None:
        with open(self.root / folder / f"{tid}.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")

    def finish(self, task: Task, job: str, rc: int, seconds: float, commit: str) -> str:
        record = {"job": job, "exit": rc, "seconds": round(seconds, 1), "commit": commit, "at": time.time()}
        if rc == 0:
            tmp = self.root / "done" / f".{task.id}.tmp"
            tmp.write_text(json.dumps(record))
            tmp.replace(self.root / "done" / f"{task.id}.json")
            outcome = "done"
        elif rc == 4:
            self._append("checkpoint", task.id, record)
            outcome = "checkpoint"
        else:
            self._append("failed", task.id, record)
            outcome = "failed"
        (self.root / "claims" / f"{task.id}.json").unlink(missing_ok=True)
        return outcome

    # ---------------------------------------------------------------- reconcile and status
    def reconcile(self, current_job: str, job_state=None) -> list[str]:
        """Release the claims of allocations Slurm reports terminal (never the current one, never unknown ones)."""
        job_state = job_state or slurm_state
        released = []
        for path in sorted((self.root / "claims").glob("*.json")):
            claim = json.loads(path.read_text())
            job = str(claim.get("job"))
            if job == str(current_job):
                continue
            state = job_state(job)
            if state and state.split()[0] in TERMINAL:
                n = len(list((self.root / "released").glob(f"{path.stem}.*.json")))
                target = self.root / "released" / f"{path.stem}.{n}.json"
                target.write_text(json.dumps({**claim, "released_at": time.time(), "slurm_state": state}))
                path.unlink()
                released.append(path.stem)
        return released

    def retry(self, ids: list[str] | None = None) -> list[str]:
        """Make failed tasks ready again (all failed ones, or ``ids``): their failure records move to failed-archive/,
        so each gets MAX_FAILURES fresh attempts. Running and done tasks are never touched."""
        archive = self.root / "failed-archive"
        archive.mkdir(exist_ok=True)
        reset = []
        for task in self.tasks():
            if ids is not None and task.id not in ids:
                continue
            path = self.root / "failed" / f"{task.id}.jsonl"
            if not path.exists() or self.done(task.id) or self.claimed(task.id):
                continue
            n = len(list(archive.glob(f"{task.id}.*.jsonl")))
            path.rename(archive / f"{task.id}.{n}.jsonl")
            reset.append(task.id)
        return reset

    def repin(self, old_code: str, new_code: str) -> list[str]:
        """Point every unfinished task at a new checkout (after a fix; done tasks keep their records and commits).
        Refused while any task is claimed: a running worker records its own checkout's commit."""
        if any((self.root / "claims").glob("*.json")):
            raise SystemExit("repin refused: tasks are claimed (wait for the allocations to end, then reconcile)")
        if not old_code or old_code == new_code:
            raise SystemExit("repin needs two different checkout paths")
        tasks, changed = self.tasks(), []
        for t in tasks:
            if self.done(t.id):
                continue
            argv = [a.replace(old_code, new_code) for a in t.argv]
            if argv != t.argv:
                t.argv = argv
                changed.append(t.id)
        path = self.root / "tasks.jsonl"
        n = len(list(self.root.glob("tasks.jsonl.before-repin-*")))
        (self.root / f"tasks.jsonl.before-repin-{n}").write_text(path.read_text())
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(t.__dict__) + "\n" for t in tasks))
        tmp.replace(path)
        with open(self.root / "repins.jsonl", "a") as f:
            f.write(json.dumps({"at": time.time(), "old": old_code, "new": new_code, "tasks": len(changed)}) + "\n")
        return changed

    def status(self) -> dict:
        tasks = self.tasks()
        states = {t.id: self.state(t) for t in tasks}
        by_stage = {}
        for t in tasks:
            by_stage.setdefault(t.stage, Counter())[states[t.id]] += 1
        seconds = 0.0
        for path in (self.root / "done").glob("*.json"):
            seconds += json.loads(path.read_text()).get("seconds", 0)
        remaining = sum(t.est_cpu_h for t in tasks if states[t.id] != "done")
        usage = shutil.disk_usage(self.root)
        vfs = os.statvfs(self.root)
        return {"tasks": len(tasks), "states": dict(Counter(states.values())), "by_stage": {k: dict(v) for k, v in by_stage.items()},
                "wall_hours_done": round(seconds / 3600, 2), "estimated_cpu_hours_remaining": round(remaining, 1),
                "checkpoints": len(list((self.root / "checkpoint").glob("*.jsonl"))),
                "failed_tasks": sorted(t for t, s in states.items() if s == "failed"),
                "ready_cpu": sum(1 for t in tasks if states[t.id] == "ready" and not t.gpu),
                "ready_gpu": sum(1 for t in tasks if states[t.id] == "ready" and t.gpu),
                "ready_cpu_cores": sum(t.cores or 76 for t in tasks if states[t.id] == "ready" and not t.gpu),
                "open_cpu": sum(1 for t in tasks if states[t.id] in ("ready", "waiting") and not t.gpu),
                "open_gpu": sum(1 for t in tasks if states[t.id] in ("ready", "waiting") and t.gpu),
                "running": sorted(t for t, s in states.items() if s == "running"),
                "disk_free_gb": round(usage.free / 1e9, 1), "inodes_free": vfs.f_favail}


def slurm_state(job: str) -> str:
    try:
        out = subprocess.run(["sacct", "-j", job, "-n", "-X", "-o", "State"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""


def git_commit() -> str:
    from analysis.scripts.page_readiness_ordering import git_commit as commit
    return commit()


# ---------------------------------------------------------------- worker

def fill(argv: list, values: dict) -> list:
    return [str(a).format(**values) for a in argv]


def set_aside_partials(argv: list) -> list[str]:
    """Before an attempt: move an earlier attempt's ``<output>.partial`` folder (the scripts write there and rename at the
    end; none resumes from it) to ``<output>.partial.interrupted-<time>``, kept for inspection, so a retry can start. The
    outputs are the values of --output, also inside ``bash -c`` strings."""
    import re
    outputs = [b for a, b in zip(argv, argv[1:]) if a == "--output"]
    for a in argv:
        if isinstance(a, str) and " " in a:
            outputs += re.findall(r"--output[ =]([^\s'\"]+)", a)
    moved = []
    for out in outputs:
        partial = Path(out + ".partial")
        if partial.exists():
            target = Path(f"{partial}.interrupted-{int(time.time())}")
            partial.rename(target)
            moved.append(str(target))
    return moved


def work(ledger: Ledger, *, job: str, cores: int, end_epoch: float, margin_minutes: float, stages: set | None,
         gpu: bool, min_start_minutes: float = 10.0, poll_seconds: float = 5.0, devices: int = 4,
         python: str | None = None, idle_minutes: float = 0.0) -> int:
    """Run ready tasks until none is ready or the deadline nears. Exit 0: nothing left that this worker may run;
    4: stopped at the deadline with ready tasks left (resubmit after reconciling). A GPU worker hands each task its
    own devices (CUDA_VISIBLE_DEVICES) from ``devices`` slots: 1-GPU statistics tasks run side by side, the embedding
    tasks (gpus 0 = all) take every slot. With ``idle_minutes`` > 0 a worker with nothing to start keeps polling for up
    to that long while another allocation runs a task its waiting tasks depend on (bounded: then it exits 0)."""
    commit = git_commit()
    running: dict = {}  # tid -> (process, task, cores, started, log, slots)
    used = 0
    free_slots = list(range(devices)) if gpu else []

    def minutes_left():
        return (end_epoch - time.time()) / 60 - margin_minutes

    def eligible(t: Task) -> bool:
        return t.gpu == gpu and (stages is None or t.stage in stages)

    deadline_hit = False
    idle_since = None
    while True:
        # collect finished tasks
        for tid in list(running):
            proc, task, n, started, log, slots = running[tid]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            outcome = ledger.finish(task, job, rc, time.time() - started, commit)
            print(json.dumps({"task": tid, "exit": rc, "outcome": outcome, "minutes": round((time.time() - started) / 60, 1)}), flush=True)
            used -= n
            free_slots = sorted(free_slots + slots)
            del running[tid]
        # admit new tasks
        admitted = False
        if minutes_left() < min_start_minutes:
            deadline_hit = True
        else:
            for task in ledger.tasks():
                if not eligible(task) or ledger.state(task) != "ready":
                    continue
                need = cores if task.cores == 0 else min(task.cores, cores)
                if used + need > cores:
                    continue
                slots = []
                if gpu:
                    want = devices if task.gpus in (0, None) else min(task.gpus, devices)
                    if len(free_slots) < want:
                        continue
                    slots = free_slots[:want]
                if not ledger.claim(task, job):
                    continue
                free_slots = [d for d in free_slots if d not in slots]
                values = {"PY": python or sys.executable, "WORKERS": need, "MINUTES": max(1, int(minutes_left())), "JOB": job}
                log = open(ledger.root / "logs" / f"{task.id}.{job}.log", "a")
                for moved in set_aside_partials(fill(task.argv, values)):
                    log.write(f"# earlier interrupted output set aside: {moved}\n")
                log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} job {job} commit {commit}\n# {' '.join(fill(task.argv, values))}\n")
                log.flush()
                env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "SLURM_CPUS_PER_TASK": str(need)}
                if gpu:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, slots))
                proc = subprocess.Popen(fill(task.argv, values), stdout=log, stderr=subprocess.STDOUT, env=env)
                running[task.id] = (proc, task, need, time.time(), log, slots)
                used += need
                admitted = True
                print(json.dumps({"task": task.id, "started": True, "cores": need, "devices": slots,
                                  "minutes_left": round(minutes_left())}), flush=True)
                if used >= cores:
                    break
        if running or admitted:
            idle_since = None
        if not running and not admitted:
            # read claims, then waiting, then ready: finish() writes "done" before removing the claim, so a dependency
            # finishing between any two reads leaves its dependants visible as waiting or as ready, never as neither
            others_running = any(ledger.claimed(t.id) for t in ledger.tasks())
            waiting = [t.id for t in ledger.tasks() if eligible(t) and ledger.state(t) in ("waiting", "running")]
            ready = [t.id for t in ledger.tasks() if eligible(t) and ledger.state(t) == "ready"]
            if deadline_hit and ready:
                print(json.dumps({"stop": "deadline", "ready_left": len(ready)}), flush=True)
                return 4
            if ready:              # became ready after the admission pass (e.g. another allocation just finished it)
                continue
            if waiting and not deadline_hit and idle_minutes > 0 and others_running:
                idle_since = idle_since or time.time()
                if time.time() - idle_since < 60 * idle_minutes:
                    time.sleep(poll_seconds)
                    continue
            print(json.dumps({"stop": "no task this worker may start", "waiting_on_dependencies_or_other_workers": len(waiting)}),
                  flush=True)
            return 0
        time.sleep(poll_seconds)


def supervise(ledger: Ledger, *, job: str, end_epoch: float, margin_minutes: float, min_start_minutes: float,
              idle_minutes: float, commands: dict, poll_seconds: float = 60.0, log=print) -> int:
    """One node, two kinds of worker (``commands``: {"cpu": argv, "gpu": argv}). A worker of a kind is (re)started
    whenever a task of that kind is ready and time is left, so neither kind is lost because the other kind's
    dependencies took long. The node is released when nothing is running here and either no task is ready or claimed
    anywhere, or for ``idle_minutes`` no task of this job has been running. Exit 4 if a worker stopped at the deadline
    with work left, else the first nonzero worker exit, else 0."""
    procs, codes, stopped = {}, [], set()
    last_busy = time.time()
    while True:
        for kind, proc in list(procs.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del procs[kind]
            codes.append(rc)
            if rc == 4:
                stopped.add(kind)
            log(json.dumps({"worker": kind, "exit": rc, "at": time.strftime("%H:%M:%S")}))
        tasks = ledger.tasks()
        claims = list((ledger.root / "claims").glob("*.json"))
        ours = 0
        for path in claims:
            try:
                ours += json.loads(path.read_text()).get("job") == job
            except (OSError, ValueError):
                continue
        if ours:
            last_busy = time.time()
        minutes_left = (end_epoch - time.time()) / 60 - margin_minutes
        states = {t.id: ledger.state(t) for t in tasks}
        ready = {k: any(states[t.id] == "ready" and t.gpu == (k == "gpu") for t in tasks) for k in commands}
        for kind, argv in commands.items():
            if kind in procs or kind in stopped or minutes_left < min_start_minutes or not ready[kind]:
                continue
            procs[kind] = subprocess.Popen(argv)
            log(json.dumps({"worker": kind, "started": True, "minutes_left": round(minutes_left), "at": time.strftime("%H:%M:%S")}))
        if not procs:
            open_tasks = any(s in ("ready", "waiting") for s in states.values())
            waiting_on_others = open_tasks and claims and minutes_left >= min_start_minutes
            if not waiting_on_others or time.time() - last_busy > 60 * idle_minutes:
                reason = ("deadline" if minutes_left < min_start_minutes else "nothing open" if not open_tasks
                          else "idle" if waiting_on_others else "nothing ready or claimed")
                log(json.dumps({"supervisor": "stop", "reason": reason, "open": open_tasks}))
                if stopped or (open_tasks and minutes_left < min_start_minutes):
                    return 4
                return next((c for c in codes if c not in (0, 4)), 0)
        time.sleep(poll_seconds)
