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


def work(ledger: Ledger, *, job: str, cores: int, end_epoch: float, margin_minutes: float, stages: set | None,
         gpu: bool, min_start_minutes: float = 10.0, poll_seconds: float = 5.0) -> int:
    """Run ready tasks until none is ready or the deadline nears. Exit 0: nothing left that this worker may run;
    4: stopped at the deadline with ready tasks left (resubmit after reconciling)."""
    commit = git_commit()
    running: dict = {}  # tid -> (process, task, cores, started, log)
    used = 0

    def minutes_left():
        return (end_epoch - time.time()) / 60 - margin_minutes

    def eligible(t: Task) -> bool:
        return t.gpu == gpu and (stages is None or t.stage in stages)

    deadline_hit = False
    while True:
        # collect finished tasks
        for tid in list(running):
            proc, task, n, started, log = running[tid]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            outcome = ledger.finish(task, job, rc, time.time() - started, commit)
            print(json.dumps({"task": tid, "exit": rc, "outcome": outcome, "minutes": round((time.time() - started) / 60, 1)}), flush=True)
            used -= n
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
                    if task.cores == 0 and used == 0:
                        pass
                    else:
                        continue
                if not ledger.claim(task, job):
                    continue
                values = {"PY": sys.executable, "WORKERS": need, "MINUTES": max(1, int(minutes_left())), "JOB": job}
                log = open(ledger.root / "logs" / f"{task.id}.{job}.log", "a")
                log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} job {job} commit {commit}\n# {' '.join(fill(task.argv, values))}\n")
                log.flush()
                env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "SLURM_CPUS_PER_TASK": str(need)}
                proc = subprocess.Popen(fill(task.argv, values), stdout=log, stderr=subprocess.STDOUT, env=env)
                running[task.id] = (proc, task, need, time.time(), log)
                used += need
                admitted = True
                print(json.dumps({"task": task.id, "started": True, "cores": need, "minutes_left": round(minutes_left())}), flush=True)
                if used >= cores:
                    break
        if not running and not admitted:
            ready = [t.id for t in ledger.tasks() if eligible(t) and ledger.state(t) == "ready"]
            if deadline_hit and ready:
                print(json.dumps({"stop": "deadline", "ready_left": len(ready)}), flush=True)
                return 4
            waiting = [t.id for t in ledger.tasks() if eligible(t) and ledger.state(t) in ("waiting", "running")]
            print(json.dumps({"stop": "no task this worker may start", "waiting_on_dependencies_or_other_workers": len(waiting)}),
                  flush=True)
            return 0
        time.sleep(poll_seconds)
