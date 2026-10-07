"""Real-data equivalence gates of the PyTorch backend (gate 4 of the GPU plan), on the Mac exploration tables.

  python -m analysis.fullrun.gpu_validation funnel    --backend torch-cpu --task "qwen38 · Reactive|P|C|main" --units 3
  python -m analysis.fullrun.gpu_validation generator --backend torch-cpu --stratum "qwen38 · Reactive"
  python -m analysis.fullrun.gpu_validation fe-compare --torch-fe DIR/fe.json

Each mode compares with the published CPU results (cached funnel fits of the exploration report, steelman-v1) and
appends a section to the report (default analysis/fullrun/validation/REPORT.md). Thresholds are those of the plan:
funnel coefficients |Δ| < 1e-4; Δ_gen |Δ| < 1e-4, interval endpoints |Δ| < 2e-4, same C2 verdict; FE |Δ| < 1e-6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

INPUTS = Path.home() / "Hamburg/geodml-inputs"
REPORT = Path(__file__).with_name("validation") / "REPORT.md"
FUNNEL_CACHE = INPUTS / "funnel-analysis-exploration-v1.cache/86fb830867005588"   # commit 362a9e9, 100 draws, exploration
STEELMAN = INPUTS / "steelman-v1"


def write(section: str, rows: list[tuple], passed: bool, report: Path, notes: str = "") -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"## {section} — {'PASS' if passed else 'FAIL'} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    if notes:
        lines += [notes, ""]
    lines += ["| quantity | CPU | torch | abs diff | threshold | ok |", "|---|---|---|---|---|---|"]
    for name, a, b, thr in rows:
        d = abs(a - b) if a is not None and b is not None else float("nan")
        lines.append(f"| {name} | {a:.10g} | {b:.10g} | {d:.3g} | {thr:g} | {'yes' if d < thr else 'NO'} |")
    with open(report, "a") as stream:
        stream.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


def funnel(args) -> bool:
    from analysis.interpretability.pipeline import funnel_models as fm
    from analysis.interpretability.pipeline import intent_stages as stages
    from analysis.interpretability.pipeline import torch_fits as tf
    from analysis.scripts import funnel_study as study
    data = study.load_assembled(INPUTS / "funnel-assembled-exploration-v1")
    ans = study.answer_arrays(data)
    ans.split = study.split_mask(data, "exploration")
    stats = study.standardisation(data, ans)
    stratum, stage_name, spec = args.task.split("|", 2)[0], "|".join(args.task.split("|")[1:3]), args.task.split("|")[3]
    task = study.stage_task(data, ans, stratum, stage_name, spec)
    task.design.stats = dict(stats)
    X = task.design.matrix(ans.x[task.row_answer])
    cpu = fm._load_parts(FUNNEL_CACHE / f"{hashlib.sha256(args.task.encode()).hexdigest()[:16]}.parts.jsonl")
    t0 = time.time()
    problem = tf.problem_for(task.kind, X, task.data, args.backend)
    full = fm._problem_fit(problem, task)
    seconds = {"full": round(time.time() - t0, 1)}
    k = len(task.design.features)
    rows = [(f"full {n}", cpu["full-0"]["x"][j], float(full.x[j]), 1e-4) for j, n in enumerate(task.design.names)]
    objective = ("full objective (pass needs torch <= cpu + 1e-12)", cpu["full-0"]["fun"], float(full.fun), 1e-6)
    draws = stages.keyword_draws(ans.keywords, 100, 20261007)
    for i in range(args.units):
        t0 = time.time()
        res = fm._problem_fit(problem, task, weight=draws[i][fm._group_keyword(task, ans.keyword)], start=np.asarray(cpu["full-0"]["x"]))
        seconds[f"bootstrap-{i}"] = round(time.time() - t0, 1)
        rows += [(f"bootstrap-{i} {n}", cpu[f"bootstrap-{i}"][j], float(res.x[j]), 1e-4) for j, n in enumerate(task.design.names)]
    passed = all(abs(a - b) < t for _, a, b, t in rows) and full.fun <= cpu["full-0"]["fun"] + 1e-12
    rows.append(objective)
    write(f"funnel {args.task} ({args.backend})", rows, passed, args.report,
          f"Rows {X.shape[0]:,} × {X.shape[1]} features; choice sets {getattr(task.data, 'sets', getattr(task.data, 'groups', 0)):,}; "
          f"seconds {json.dumps(seconds)}; Newton iterations (full) {getattr(full, 'nit', '?')}.")
    return passed


def generator(args) -> bool:
    from analysis.interpretability.pipeline import funnel_models as fm
    from analysis.steelman import decide
    from analysis.steelman import generator as gen
    from analysis.steelman import tables as T
    fm.set_backend(args.backend)
    t = T.load()
    run_args = SimpleNamespace(bootstrap=200, model_draws=100, mc=200, seed=20261007, workers=1, stratum=args.stratum)
    t0 = time.time()
    out = gen.run(t, run_args, cache_root=Path(args.cache), deadline=time.monotonic() + 1e9)
    seconds = round(time.time() - t0, 1)
    torch_entry = out["strata"][args.stratum]
    cpu_entry = json.loads((STEELMAN / "generator.json").read_text())["strata"][args.stratum]
    rows = []
    for variant in ("main", "no_slot", "score"):
        a, b = cpu_entry["delta"][variant], torch_entry["delta"][variant]
        rows.append((f"{variant} delta_gen", a["delta_gen"]["estimate"], b["delta_gen"]["estimate"], 1e-4))
        for key in ("ci90", "ci95"):
            for side in (0, 1):
                rows.append((f"{variant} delta_gen {key}[{side}]", a["delta_gen"][key][side], b["delta_gen"][key][side], 2e-4))
        rows.append((f"{variant} model_check", a["model_check"]["estimate"], b["model_check"]["estimate"], 1e-4))
    for key, coefs in cpu_entry["coefficients"].items():
        for name, v in coefs.items():
            if name == "slot_effects":
                continue
            rows.append((f"{key} {name}", v["estimate"], torch_entry["coefficients"][key][name]["estimate"], 1e-4))
    same_verdict = (cpu_entry["delta"]["main"]["tost_inside_sesoi"] == torch_entry["delta"]["main"]["tost_inside_sesoi"])
    passed = all(abs(a - b) < thr for _, a, b, thr in rows) and same_verdict and torch_entry["delta"]["main"]["failed_replicates"] == 0
    write(f"steelman generator {args.stratum} ({args.backend})", rows, passed, args.report,
          f"100 keyword draws, 200 Monte-Carlo draws per answer; {seconds} s on this machine; TOST verdict CPU "
          f"{cpu_entry['delta']['main']['tost_inside_sesoi']} / torch {torch_entry['delta']['main']['tost_inside_sesoi']}.")
    _ = decide
    return passed


def fe_compare(args) -> bool:
    cpu = json.loads((STEELMAN / "fe.json").read_text())["strata"]
    got = json.loads(Path(args.torch_fe).read_text())["strata"]
    rows = []
    for stratum, outcomes in cpu.items():
        for outcome, r in outcomes.items():
            if "skipped" in r:
                continue
            for name, v in r["coefficients_per_sd"].items():
                g = got[stratum][outcome]["coefficients_per_sd"][name]
                rows.append((f"{stratum} {outcome} {name}", v["estimate"], g["estimate"], 1e-6))
                rows.append((f"{stratum} {outcome} {name} ci95 lo", v["ci95"][0], g["ci95"][0], 1e-6))
                rows.append((f"{stratum} {outcome} {name} ci95 hi", v["ci95"][1], g["ci95"][1], 1e-6))
    passed = all(abs(a - b) < thr for _, a, b, thr in rows)
    write("steelman two-way fixed effects (all strata)", rows, passed, args.report,
          f"torch file {args.torch_fe}; seconds {json.loads(Path(args.torch_fe).read_text()).get('seconds')}.")
    return passed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("funnel", "generator", "fe-compare"))
    p.add_argument("--backend", default="torch-cpu")
    p.add_argument("--task", default="qwen38 · Reactive|P|C|main")
    p.add_argument("--units", type=int, default=3)
    p.add_argument("--stratum", default="qwen38 · Reactive")
    p.add_argument("--cache", default="/tmp/gpu-validation-generator-cache")
    p.add_argument("--torch-fe")
    p.add_argument("--report", type=Path, default=REPORT)
    a = p.parse_args(argv)
    ok = {"funnel": funnel, "generator": generator, "fe-compare": fe_compare}[a.mode](a)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
