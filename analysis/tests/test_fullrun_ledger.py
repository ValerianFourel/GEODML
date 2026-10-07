"""The task ledger: dependencies, atomic claims, deadline checkpoints, finite failures, reconcile; the planned task list."""

import json
import sys
import threading
import time

import pytest

from analysis.fullrun import ledger as L
from analysis.fullrun import plan
from analysis.tests.test_funnel_study import pipeline  # noqa: F401


def task(tid, code, deps=(), cores=1, gpu=False):
    return L.Task(tid, "test", [sys.executable, "-c", code], list(deps), cores=cores, gpu=gpu)


def touch(path):
    return f"import pathlib; pathlib.Path({str(path)!r}).write_text('x')"


def run(ledger, job="j1", cores=4, hours=1.0, **kw):
    return L.work(ledger, job=job, cores=cores, end_epoch=time.time() + 3600 * hours, margin_minutes=0, stages=None,
                  gpu=kw.pop("gpu", False), min_start_minutes=kw.pop("min_start", 0.0), poll_seconds=0.05)


def test_dependencies_and_completion(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    led.write_tasks([task("a", touch(out / "a")), task("b", f"import pathlib,sys; assert pathlib.Path({str(out / 'a')!r}).exists(); "
                                                       + touch(out / "b"), ["a"]), task("g", "pass", gpu=True)])
    with pytest.raises(SystemExit):
        led.write_tasks([task("a", "pass")])                      # planned once
    assert run(led) == 0
    assert led.done("a") and led.done("b") and not led.done("g")   # a CPU worker never runs GPU tasks
    assert led.status()["states"] == {"done": 2, "ready": 1}
    assert run(led, gpu=True) == 0 and led.done("g")


def test_checkpoint_resumes_and_failures_are_finite(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    flag = tmp_path / "second"
    code = f"import pathlib,sys; p=pathlib.Path({str(flag)!r}); sys.exit(0 if p.exists() else (p.write_text('x'), 4)[1])"
    led.write_tasks([task("cp", code), task("bad", "import sys; sys.exit(3)")])
    assert run(led) == 0
    assert led.done("cp")                                         # exit 4 then rerun to completion
    assert (led.root / "checkpoint" / "cp.jsonl").exists()
    assert led.failures("bad") == L.MAX_FAILURES and led.state(led.tasks()[1]) == "failed"


def test_two_workers_never_run_a_task_twice(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    log = tmp_path / "runs.log"
    code = f"import time; time.sleep(0.2); open({str(log)!r}, 'a').write('x')"
    led.write_tasks([task(f"t{i}", code) for i in range(12)])
    threads = [threading.Thread(target=run, args=(led,), kwargs={"job": f"j{i}", "cores": 3}) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert log.read_text() == "x" * 12 and len(list((led.root / "done").glob("*.json"))) == 12


def test_deadline_stops_admission_with_exit_4(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([task("a", "pass")])
    assert L.work(led, job="j", cores=1, end_epoch=time.time() + 60, margin_minutes=0, stages=None, gpu=False,
                  min_start_minutes=5.0, poll_seconds=0.05) == 4
    assert not led.done("a")


def test_reconcile_releases_only_terminal_other_jobs(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([task("a", "pass"), task("b", "pass"), task("c", "pass")])
    for tid, job in (("a", "100"), ("b", "200"), ("c", "300")):
        assert led.claim(led.tasks()[["a", "b", "c"].index(tid)], job)
    states = {"100": "TIMEOUT", "200": "RUNNING", "300": ""}
    released = led.reconcile("999", job_state=states.get)
    assert released == ["a"] and led.claimed("b") and led.claimed("c")
    assert json.loads(next((led.root / "released").glob("a.*.json")).read_text())["slurm_state"] == "TIMEOUT"
    assert not led.claim(led.tasks()[1], "400")                   # a live claim is never taken over


def test_planned_task_list_is_consistent(tmp_path):
    cfg = {"code": "/c", "output": "/o", "sources": ["/d/l:llama4", "/d/q:qwen38"], "snapshots": {"duckduckgo": "/s/d", "searxng": "/s/s"},
           "axis_map": "/a.jsonl", "population": "/p.jsonl", "corpus": "/corpus", "features": "/f", "archive": "/A", "search_root": "/W",
           "gemma_run": "/g", "shards": 32}
    tasks = plan.build(cfg)
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks(tasks)                                        # unique ids, known dependencies
    ids = {t.id for t in tasks}
    assert sum(t.id.startswith("extract-funnel-") for t in tasks) == 32 and sum(t.gpu for t in tasks) == 4
    heavy = [t for t in tasks if t.id.startswith("funnel-confirmation-qwen38-reactive-p-c-main-bootstrap")]
    assert len(heavy) == 4 and all("--units" in t.argv for t in heavy)
    assert {"steelman-confirmation-supply", "steelman-exploration-census", "bundle"} <= ids
    bundle = next(t for t in tasks if t.id == "bundle")
    assert "steelman-confirmation-report" in bundle.deps and "funnel-report-exploration" in bundle.deps
    scripts = {a for t in tasks for a in t.argv if a.startswith("/c/analysis/")}
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    assert all((repo / s[len("/c/"):]).exists() for s in scripts), sorted(s for s in scripts if not (repo / s[len("/c/"):]).exists())
    print(json.dumps({"tasks": len(tasks), "cpu_h": round(sum(t.est_cpu_h for t in tasks), 1)}))


def test_worker_runs_the_real_pipeline_end_to_end(pipeline, tmp_path):  # noqa: F811
    """Extraction shards -> merge -> replay -> assemble -> every steelman part -> report, on two models, through the
    ledger and the worker (the funnel fits, GPU embeddings and intent analysis are tested on their own)."""
    inputs = pipeline["inputs"]
    from analysis.tests.test_funnel_study import snapshot_args
    snaps = snapshot_args(pipeline["tmp"])
    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    cfg = {"code": str(repo), "output": str(tmp_path / "fr"), "sources": [f"{inputs['root']}:llama4", f"{inputs['root']}:qwen38"],
           "snapshots": dict(s.split("=", 1) for s in snaps[1::2]), "axis_map": str(inputs["final"]),
           "population": str(inputs["population"]), "corpus": str(inputs["corpus"]), "features": str(pipeline["features"]),
           "archive": str(pipeline["tmp"]), "search_root": str(pipeline["tmp"]), "shards": 3, "splits": ["all"],
           "include_funnel": False, "draws": {"bootstrap": 4, "permutations": 3, "model_draws": 2, "mc": 8}}
    keep = lambda t: t.stage in ("extract", "merge", "steelman") or t.id in ("funnel-replay", "funnel-assemble") or t.id.startswith("steelman-")  # noqa: E731
    tasks = [t for t in plan.build(cfg) if keep(t)]
    # the assemble task reads the fixture's prompt projections and maps from the test folder layout
    for t in tasks:
        if t.id == "funnel-assemble":
            t.argv = [a.replace(f"{pipeline['tmp']}/final-audit/projections/qwen", str(pipeline["tmp"] / "prompts-qwen"))
                       .replace(f"{pipeline['tmp']}/final-audit/projections/mistral", str(pipeline["tmp"] / "prompts-mistral"))
                       .replace(f"{pipeline['tmp']}/maps/qwen", str(inputs["maps"]["qwen"]))
                       .replace(f"{pipeline['tmp']}/maps/mistral", str(inputs["maps"]["mistral"])) for a in t.argv]
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks(tasks)
    assert run(led, cores=8, hours=1) == 0
    status = led.status()
    failed = {p.stem: p.read_text()[-2000:] for p in (led.root / "logs").glob("*.log") if not led.done(p.name.split(".")[0])}
    assert status["states"] == {"done": len(tasks)}, (status, failed)
    results = (tmp_path / "fr" / "steelman-all" / "RESULTS.md").read_text()
    assert "llama4" in results and "qwen38" in results
