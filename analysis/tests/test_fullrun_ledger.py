"""The task ledger: dependencies, atomic claims, deadline checkpoints, finite failures, reconcile; the planned task list."""

import json
from pathlib import Path
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
    assert sum(t.id.startswith("extract-funnel-") for t in tasks) == 32 and sum(t.gpu for t in tasks) == 6
    by = {t.id: t for t in tasks}
    # the relocation check gates every new embedding and is passed to the intent analysis
    assert all("relocation" in by[f"embed-{w}-{v}"].deps for w in ("queries", "answers") for v in ("qwen", "mistral"))
    assert "relocation" in by["intent-analyze"].deps and "--relocation" in by["intent-analyze"].argv[2]
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


def test_gpu_stats_routes_the_heavy_fits_to_one_gpu_each():
    cfg = {"code": "/c", "output": "/o", "sources": ["/d/l:llama4", "/d/q:qwen38"], "snapshots": {"duckduckgo": "/s/d", "searxng": "/s/s"},
           "axis_map": "/a.jsonl", "population": "/p.jsonl", "corpus": "/corpus", "features": "/f", "archive": "/A", "search_root": "/W",
           "shards": 8, "gpu_stats": True}
    tasks = {t.id: t for t in plan.build(cfg)}
    unit = tasks["funnel-confirmation-qwen38-reactive-p-c-main-bootstrap-000"]
    assert unit.gpu and unit.gpus == 1 and unit.argv[unit.argv.index("--backend") + 1] == "cuda" and "bootstrap:0-200" in unit.argv
    assert tasks["steelman-confirmation-generator-llama4-parallel"].gpus == 1
    assert tasks["decisions-exploration"].gpus == 1
    # assembly, reports and the cheap parts stay on CPU workers but carry the backend for the cache key
    for tid in ("funnel-confirmation-qwen38-reactive-p-c-main", "steelman-confirmation-generator", "funnel-report-confirmation"):
        assert not tasks[tid].gpu and "cuda" in tasks[tid].argv
    assert not tasks["steelman-confirmation-supply"].gpu
    L.Ledger.__init__  # noqa: B018


def test_gpu_stats_per_estimator():
    cfg = {"code": "/c", "output": "/o", "sources": ["/d/l:llama4", "/d/q:qwen38"], "snapshots": {"duckduckgo": "/s/d", "searxng": "/s/s"},
           "axis_map": "/a.jsonl", "population": "/p.jsonl", "corpus": "/corpus", "features": "/f", "archive": "/A", "search_root": "/W",
           "shards": 4, "gpu_stats": ["generator", "fe"]}
    tasks = {t.id: t for t in plan.build(cfg)}
    assert tasks["steelman-exploration-generator-qwen38-reactive"].gpus == 1
    assert tasks["steelman-exploration-fe-llama4-parallel"].gpus == 1
    assert not tasks["funnel-exploration-qwen38-reactive-p-c-main-full"].gpu and "--backend" not in tasks["decisions-exploration"].argv
    assert "--backend" not in tasks["steelman-exploration-chain"].argv
    with pytest.raises(ValueError):
        plan.build({**cfg, "gpu_stats": ["chain"]})


def test_smoke_lists_are_valid(tmp_path):
    from analysis.fullrun import smoke
    cfg = {"code": "/c", "output": "/o", "sources": ["/d/l:llama4"], "snapshots": {"duckduckgo": "/s/d"}, "axis_map": "/a",
           "population": "/p", "corpus": "/c2", "archive": "/A"}
    for gpu in (False, True):
        tasks = smoke.build(cfg, gpu)
        L.Ledger(tmp_path / f"l{gpu}").write_tasks(tasks)
        assert all(t.gpu == gpu for t in tasks)
    assert any(t.gpus == 0 for t in smoke.build(cfg, True))       # the 4-GPU embedding


def test_chain_smoke_runs_end_to_end_on_the_fixture(pipeline, tmp_path):  # noqa: F811
    from analysis.fullrun import smoke
    from analysis.tests.test_funnel_study import snapshot_args
    inputs = pipeline["inputs"]
    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    cfg = {"code": str(repo), "output": str(tmp_path / "fr"), "sources": [f"{inputs['root']}:llama4", f"{inputs['root']}:qwen38"],
           "snapshots": dict(s.split("=", 1) for s in snapshot_args(pipeline["tmp"])[1::2]), "axis_map": str(inputs["final"]),
           "population": str(inputs["population"]), "corpus": str(inputs["corpus"]), "features": str(pipeline["features"]),
           "archive": str(pipeline["tmp"])}
    tasks = smoke.chain(cfg, shard="1/1")
    for t in tasks:   # the fixture keeps its prompt projections and maps in the test folder layout
        t.argv = [a.replace(f"{pipeline['tmp']}/final-audit/projections/qwen", str(pipeline["tmp"] / "prompts-qwen"))
                   .replace(f"{pipeline['tmp']}/final-audit/projections/mistral", str(pipeline["tmp"] / "prompts-mistral"))
                   .replace(f"{pipeline['tmp']}/maps/qwen", str(inputs["maps"]["qwen"]))
                   .replace(f"{pipeline['tmp']}/maps/mistral", str(inputs["maps"]["mistral"])) for a in t.argv]
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks(tasks)
    assert run(led, cores=8, hours=1) == 0
    failed = {p.name: p.read_text()[-1500:] for p in (led.root / "logs").glob("*.log") if not led.done(p.name.split(".")[0])}
    assert led.status()["states"] == {"done": len(tasks)}, failed
    assert (tmp_path / "fr/smoke-chain/steelman/RESULTS.md").exists()


def test_retry_resets_only_failed_tasks(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    flag = tmp_path / "fixed"
    led.write_tasks([task("flaky", f"import pathlib, sys; sys.exit(0 if pathlib.Path({str(flag)!r}).exists() else 3)"), task("ok", "pass")])
    assert run(led) == 0
    assert led.state(led.tasks()[0]) == "failed" and led.done("ok")
    flag.write_text("x")                                     # "the fix"
    assert led.retry() == ["flaky"] and led.state(led.tasks()[0]) == "ready"
    s = led.status()
    assert s["ready_cpu"] == 1 and s["ready_gpu"] == 0 and s["open_cpu"] == 1
    assert run(led) == 0 and led.done("flaky")
    assert len(list((led.root / "failed-archive").glob("flaky.*.jsonl"))) == 1
    assert led.retry() == []


def test_idle_worker_waits_for_a_dependency_running_elsewhere(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([task("dep", "pass"), task("after", "pass", ["dep"])])
    assert led.claim(led.tasks()[0], "other")                # another allocation runs the dependency
    t0 = time.time()
    rc = L.work(led, job="j1", cores=2, end_epoch=time.time() + 3600, margin_minutes=0, stages=None, gpu=False,
                min_start_minutes=0, poll_seconds=0.05, idle_minutes=0.01)
    assert rc == 0 and 0.5 <= time.time() - t0 < 10 and not led.done("after")   # bounded wait, then exit 0

    def finish_elsewhere():
        time.sleep(0.3)
        led.finish(led.tasks()[0], "other", 0, 1.0, "c")
    th = threading.Thread(target=finish_elsewhere)
    led.claim(led.tasks()[0], "other")
    th.start()
    rc = L.work(led, job="j1", cores=2, end_epoch=time.time() + 3600, margin_minutes=0, stages=None, gpu=False,
                min_start_minutes=0, poll_seconds=0.05, idle_minutes=1)
    th.join()
    assert rc == 0 and led.done("after")                     # picked up as soon as the dependency finished


def test_repin_points_unfinished_tasks_at_the_new_checkout(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([L.Task("a", "s", ["py", "/c/old/x.py"], []), L.Task("b", "s", ["py", "/c/old/y.py", "--out", "/o"], ["a"])])
    led.finish(led.tasks()[0], "j", 0, 1.0, "c")
    led.claim(led.tasks()[1], "j2")
    with pytest.raises(SystemExit):
        led.repin("/c/old", "/c/new")                        # refused while a task is claimed
    (led.root / "claims" / "b.json").unlink()
    assert led.repin("/c/old", "/c/new") == ["b"]
    a, b = led.tasks()
    assert a.argv == ["py", "/c/old/x.py"] and b.argv == ["py", "/c/new/y.py", "--out", "/o"]
    assert (led.root / "tasks.jsonl.before-repin-0").exists() and (led.root / "repins.jsonl").exists()


def test_mixed_node_supervisor_runs_cpu_then_gpu_then_cpu(tmp_path):
    """Workers that exit as soon as they are idle (the failure the supervisor exists for): the GPU task becomes ready
    only after a CPU task, and the last CPU task only after the GPU task; one node must still finish all three."""
    led = L.Ledger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    led.write_tasks([task("a", "import time; time.sleep(0.5); " + touch(out / "a")),
                     L.Task("g", "test", [sys.executable, "-c", touch(out / "g")], ["a"], gpu=True, gpus=1),
                     task("b", touch(out / "b"), ["g"])])
    root = str(Path(L.__file__).resolve().parents[2])
    end = time.time() + 3600
    def worker(*extra):
        return [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from analysis.fullrun.__main__ import main; "
                "raise SystemExit(main(sys.argv[2:]))", root, "worker", "--ledger", str(led.root), "--job", "node1",
                "--end-epoch", str(end), "--margin-minutes", "0", "--min-start-minutes", "0", "--cores", "2", *extra]
    rc = L.supervise(led, job="node1", end_epoch=end, margin_minutes=0, min_start_minutes=0, idle_minutes=0.05,
                     commands={"cpu": worker(), "gpu": worker("--gpu", "--devices", "1")}, poll_seconds=0.1, log=lambda m: None)
    assert rc == 0 and all(led.done(t) for t in ("a", "g", "b"))


def test_mixed_supervisor_releases_an_idle_node(tmp_path):
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([task("dep", "pass"), task("after", "pass", ["dep"])])
    led.claim(led.tasks()[0], "another-node")                      # runs elsewhere and never finishes here
    t0 = time.time()
    rc = L.supervise(led, job="node1", end_epoch=time.time() + 3600, margin_minutes=0, min_start_minutes=0, idle_minutes=0.02,
                     commands={"cpu": [sys.executable, "-c", "pass"]}, poll_seconds=0.1, log=lambda m: None)
    assert rc == 0 and time.time() - t0 < 10 and not led.done("after")
