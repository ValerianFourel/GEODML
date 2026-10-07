"""The full run as a finite task list (PREREG addendum B3; plan in analysis/docs/handoff/2026-10-08_full-run-plan_handoff.md).

Every task is a command line of an existing, tested entry point. The list is written once into a ledger folder and
worked through by any number of CPU allocations (``worker``) and GPU allocations (``worker --gpu``). Paths come from a
JSON config (see ``analysis/fullrun/horeka-config.example.json``); nothing is hard-coded here.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ledger import Task

STRATA = ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive")
STAGES = ("R|U", "R0|U", "P|C", "K|P")
SPECS = ("main", "visible", "complete")
SPLITS = ("exploration", "confirmation")
UNIT_BLOCK = 50          # bootstrap and permutation units per range task of a heavy funnel fit
STEELMAN_SINGLE = ("chain", "followup", "pairs", "census", "supply", "ablation", "queries")


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.replace(" · ", "-")).strip("-").lower().replace("--", "-")


def build(cfg: dict) -> list[Task]:
    code, out = cfg["code"], cfg["output"]
    py = "{PY}"
    script = lambda name: f"{code}/analysis/scripts/{name}"  # noqa: E731
    sources = [a for s in cfg["sources"] for a in ("--source", s)]
    snaps = [a for e, p in cfg["snapshots"].items() for a in ("--snapshot", f"{e}={p}")]
    archive = cfg["archive"]
    n = int(cfg.get("shards", 32))
    gpu_stats = bool(cfg.get("gpu_stats", False))   # heavy fits on one GPU each with the PyTorch backend (torch_fits)
    backend = ["--backend", "cuda"] if gpu_stats else []
    heavy = dict(gpu=True, gpus=1, cores=8) if gpu_stats else {}
    unit_block = 200 if gpu_stats else UNIT_BLOCK
    draws = {"bootstrap": 200, "permutations": 200, "secondary_bootstrap": 100, "decision_draws": 100, "model_draws": 100, "mc": 200,
             **cfg.get("draws", {})}  # production values are the defaults; tests shrink them
    tasks: list[Task] = []
    add = lambda *a, **k: tasks.append(Task(*a, **k))  # noqa: E731

    # ---------------------------------------------------------------- extraction, sharded by keyword hash
    for k in range(1, n + 1):
        add(f"extract-funnel-{k:02d}", "extract",
            [py, "-u", script("funnel_study.py"), "extract", *sources, *snaps, "--final-axis-map", cfg["axis_map"],
             "--population-prompts", cfg["population"], "--workers", "{WORKERS}", "--prompt-shard", f"{k}/{n}",
             "--output", f"{out}/shards/funnel-{k:02d}"], [], cores=8, est_cpu_h=0.6)
        add(f"extract-trace-{k:02d}", "extract",
            [py, "-u", script("intent_stages_study.py"), "trace-extract", *sources, "--final-axis-map", cfg["axis_map"],
             "--corpus-package", cfg["corpus"], "--population-prompts", cfg["population"], "--workers", "{WORKERS}",
             "--prompt-shard", f"{k}/{n}", "--output", f"{out}/shards/trace-{k:02d}"], [], cores=8, est_cpu_h=0.5)
    add("merge-funnel", "merge", [py, "-u", "-m", "analysis.fullrun", "merge", "funnel", "--output", f"{out}/funnel-extract",
                                  *[f"{out}/shards/funnel-{k:02d}" for k in range(1, n + 1)]],
        [f"extract-funnel-{k:02d}" for k in range(1, n + 1)], est_cpu_h=0.2)
    add("merge-trace", "merge", [py, "-u", "-m", "analysis.fullrun", "merge", "trace", "--output", f"{out}/trace-extract",
                                 *[f"{out}/shards/trace-{k:02d}" for k in range(1, n + 1)]],
        [f"extract-trace-{k:02d}" for k in range(1, n + 1)], est_cpu_h=0.2)
    add("funnel-replay", "prepare", [py, "-u", script("funnel_study.py"), "replay", *snaps, "--population-prompts", cfg["population"],
                                     "--output", f"{out}/funnel-replay"], [], est_cpu_h=0.2)
    add("funnel-assemble", "prepare",
        [py, "-u", script("funnel_study.py"), "assemble", "--extract", f"{out}/funnel-extract", "--features", cfg["features"],
         "--replay", f"{out}/funnel-replay", "--corpus-package", cfg["corpus"], "--qwen-prompts", f"{archive}/final-audit/projections/qwen",
         "--mistral-prompts", f"{archive}/final-audit/projections/mistral", "--qwen-map", f"{archive}/maps/qwen",
         "--mistral-map", f"{archive}/maps/mistral", "--output", f"{out}/funnel-assembled"],
        ["merge-funnel", "funnel-replay"], est_cpu_h=0.5)
    roots = [a for s in cfg["sources"] for a in ("--dataset-root", s.rpartition(":")[0])]
    add("intent-replay", "prepare",
        ["bash", "-c", f'"$0" -u {script("intent_stages_study.py")} replay --trace-extract {out}/trace-extract '
         f'--corpus-package {cfg["corpus"]} {" ".join(roots)} --search-root {cfg["search_root"]} --workers "$1" '
         f'--output {out}/intent-replay || test -f {out}/intent-replay.fidelity-failed.json', py, "{WORKERS}"],
        ["merge-trace"], cores=8, est_cpu_h=1.0)
    if cfg.get("gemma_run"):
        add("gemma-extract", "prepare", [py, "-u", script("intent_stages_study.py"), "gemma-extract", "--gemma-run", cfg["gemma_run"],
                                         "--corpus-package", cfg["corpus"], "--workers", "{WORKERS}", "--output", f"{out}/gemma-extract"],
            [], cores=8, est_cpu_h=1.0)
    datasets = [a for s in cfg["sources"] for a in ("--dataset", f"{s.rpartition(':')[2]}={s.rpartition(':')[0]}")]
    add("answers-export", "prepare", [py, "-u", script("answer_readiness.py"), "export", *datasets, "--axis-map", cfg["axis_map"],
                                      "--output", f"{out}/answers-export"], [], est_cpu_h=0.5)

    # ---------------------------------------------------------------- GPU: both LLM2Vec views of queries and answers
    embed = f"{code}/analysis/docs/horeka-fullrun-embed.sh"
    # relocation check first (as horeka-page-readiness-lib.sh:ensure_relocation): a fresh re-embedding of 512 archived
    # prompts must reproduce the archived axis before any new text is embedded
    add("relocation-sample", "prepare", [py, "-u", script("page_readiness_ordering.py"), "sample-prompts", "--prompts",
                                         f"{archive}/final-audit/compliant-candidates.jsonl", "--count", "512",
                                         "--output", f"{out}/prompt-sample"], [], est_cpu_h=0.05)
    for view in ("qwen", "mistral"):
        add(f"embed-sample-{view}", "gpu", ["bash", embed, view, f"{out}/prompt-sample/pages.jsonl.gz", f"{out}/embed-sample-{view}"],
            ["relocation-sample"], cores=0, est_cpu_h=0.1, gpu=True)
    add("relocation", "prepare", [py, "-u", script("page_readiness_ordering.py"), "relocate", "--final-axis-map", cfg["axis_map"],
                                  "--qwen-projections", f"{archive}/final-audit/merged/qwen",
                                  "--mistral-projections", f"{archive}/final-audit/merged/mistral", "--battery", f"{archive}/battery",
                                  "--fresh-qwen", f"{out}/embed-sample-qwen.merged", "--fresh-mistral", f"{out}/embed-sample-mistral.merged",
                                  "--output", f"{out}/relocation-fresh.json"], ["embed-sample-qwen", "embed-sample-mistral"], est_cpu_h=0.05)
    for view in ("qwen", "mistral"):
        add(f"embed-queries-{view}", "gpu", ["bash", embed, view, f"{out}/trace-extract/queries.jsonl.gz", f"{out}/embed-queries-{view}"],
            ["merge-trace", "relocation"], cores=0, est_cpu_h=0.5, gpu=True)
        add(f"embed-answers-{view}", "gpu", ["bash", embed, view, f"{out}/answers-export/answers.jsonl.gz", f"{out}/embed-answers-{view}"],
            ["answers-export", "relocation"], cores=0, est_cpu_h=1.0, gpu=True)
    add("answers-analyze", "analysis",
        [py, "-u", script("answer_readiness.py"), "analyze", "--export", f"{out}/answers-export", "--qwen", f"{out}/embed-answers-qwen.merged",
         "--mistral", f"{out}/embed-answers-mistral.merged", "--battery", f"{archive}/battery", "--axis-map", cfg["axis_map"],
         "--output", f"{out}/answers-analysis"], ["embed-answers-qwen", "embed-answers-mistral"], est_cpu_h=0.3)
    intent_deps = ["merge-trace", "intent-replay", "embed-queries-qwen", "embed-queries-mistral", "embed-answers-qwen",
                   "embed-answers-mistral", "answers-export", "relocation"] + (["gemma-extract"] if cfg.get("gemma_run") else [])
    add("intent-analyze", "analysis",
        ["bash", "-c", f'R=; [ -f {out}/intent-replay/manifest.json ] && R="--replay {out}/intent-replay"; '
         f'G=; [ -f {out}/gemma-extract/manifest.json ] && G="--gemma {out}/gemma-extract"; '
         f'"$0" -u {script("intent_stages_study.py")} analyze --trace-extract {out}/trace-extract $R --corpus-package {cfg["corpus"]} '
         f'--qwen-prompts {archive}/final-audit/projections/qwen --mistral-prompts {archive}/final-audit/projections/mistral '
         f'--qwen-map {archive}/maps/qwen --mistral-map {archive}/maps/mistral --battery {archive}/battery '
         f'--final-axis-map {cfg["axis_map"]} --queries-qwen {out}/embed-queries-qwen.merged '
         f'--queries-mistral {out}/embed-queries-mistral.merged --answers-index {out}/answers-export/observations.jsonl.gz '
         f'--answers-id-field answer_id --answers-qwen {out}/embed-answers-qwen.merged '
         f'--answers-mistral {out}/embed-answers-mistral.merged $G --relocation {out}/relocation-fresh.json --bootstrap {draws["bootstrap"]} --permutations {draws["permutations"]} --workers "$1" '
         f'--output {out}/intent-stages', py, "{WORKERS}"], intent_deps, cores=0, est_cpu_h=8.0)

    # ---------------------------------------------------------------- per keyword split
    splits = tuple(cfg.get("splits", SPLITS))
    for split in splits:
        fdir = f"{out}/funnel-analysis-{split}"
        base = [py, "-u", script("funnel_study.py"), "analyze", "--assembled", f"{out}/funnel-assembled", "--output", fdir,
                "--specs", ",".join(SPECS), "--split", split, "--workers", "{WORKERS}", "--stop-after-minutes", "{MINUTES}",
                "--bootstrap", str(draws["bootstrap"]), "--permutations", str(draws["permutations"]),
                "--secondary-bootstrap", str(draws["secondary_bootstrap"]), *backend]
        funnel_ids = []
        for stratum in STRATA:
            for stage in STAGES:
                for spec in SPECS:
                    name = f"{stratum}|{stage}|{spec}"
                    tid = f"funnel-{split}-{slug(stratum)}-{slug(stage)}-{spec}"
                    if stage == "P|C" and spec == "main":
                        unit_ids = []
                        add(f"{tid}-full", "funnel", [*base, "--task", name, "--units", "full"], ["funnel-assemble"],
                            **({"cores": 4} | heavy), est_cpu_h=0.2)
                        unit_ids.append(f"{tid}-full")
                        for kind, count in (("drop", 0), ("bootstrap", draws["bootstrap"]), ("permutation", draws["permutations"])):
                            ranges = [("drop", None)] if kind == "drop" else [(f"{kind}:{lo}-{min(lo + unit_block, count)}", lo)
                                                                              for lo in range(0, count, unit_block)]
                            for spec_units, lo in ranges:
                                uid = f"{tid}-{kind}" + ("" if lo is None else f"-{lo:03d}")
                                add(uid, "funnel", [*base, "--task", name, "--units", spec_units], [f"{tid}-full"],
                                    **({"cores": 16} | heavy), est_cpu_h=3.5 if kind != "drop" else 0.6)
                                unit_ids.append(uid)
                        add(tid, "funnel", [*base, "--task", name], unit_ids, cores=4, est_cpu_h=0.2)
                    else:
                        add(tid, "funnel", [*base, "--task", name], ["funnel-assemble"], **({"cores": 16} | heavy),
                            est_cpu_h=6.0 if stage == "P|C" else 2.0 if spec == "main" else 1.0)
                    funnel_ids.append(tid)
        add(f"funnel-report-{split}", "report",
            [py, "-u", script("funnel_study.py"), "report", "--assembled", f"{out}/funnel-assembled", "--output", fdir,
             "--specs", ",".join(SPECS), "--split", split, "--bootstrap", str(draws["bootstrap"]),
             "--permutations", str(draws["permutations"]), "--secondary-bootstrap", str(draws["secondary_bootstrap"]),
             "--report", f"{out}/funnel-report-{split}", *backend], funnel_ids, est_cpu_h=0.2)
        add(f"decisions-{split}", "analysis",
            [py, "-u", script("funnel_importance.py"), "--assembled", f"{out}/funnel-assembled", "--split", split, "--bootstrap", str(draws["decision_draws"]),
             "--permutations", str(draws["decision_draws"]), "--workers", "{WORKERS}", "--stop-after-minutes", "{MINUTES}", "--output", f"{out}/decisions-{split}",
             *backend], ["funnel-assemble"], **({"cores": 16} | heavy), est_cpu_h=10.0)

        sdir = f"{out}/steelman-{split}"
        steel = [py, "-u", "-m", "analysis.steelman", "PART", "--assembled", f"{out}/funnel-assembled", "--extract", f"{out}/funnel-extract",
                 "--replay", f"{out}/funnel-replay", "--trace-extract", f"{out}/trace-extract", "--features", cfg["features"],
                 "--prompts", cfg["population"], "--split", split, "--input-root", out, "--output", sdir,
                 "--workers", "{WORKERS}", "--stop-after-minutes", "{MINUTES}", "--bootstrap", str(draws["bootstrap"]),
                 "--permutations", str(draws["permutations"]), "--model-draws", str(draws["model_draws"]), "--mc", str(draws["mc"]),
                 *backend]
        part = lambda name, *extra: [a if a != "PART" else name for a in steel] + list(extra)  # noqa: E731
        prep = ["funnel-assemble", "merge-trace"]
        steel_ids = []
        for name in STEELMAN_SINGLE:
            add(f"steelman-{split}-{name}", "steelman", part(name), prep, est_cpu_h=0.5)
            steel_ids.append(f"steelman-{split}-{name}")
        add(f"steelman-{split}-lexical", "steelman", part("lexical", "--selection-draws", "0"), prep, est_cpu_h=0.5)
        steel_ids.append(f"steelman-{split}-lexical")
        for name, est in (("generator", 12.0), ("fe", 4.0)):
            ids = []
            for stratum in STRATA:
                sid = f"steelman-{split}-{name}-{slug(stratum)}"
                add(sid, "steelman", part(name, "--stratum", stratum), prep, **({"cores": 16} | heavy), est_cpu_h=est)
                ids.append(sid)
            add(f"steelman-{split}-{name}", "steelman", part(name), ids, cores=4, est_cpu_h=0.3)
            steel_ids.append(f"steelman-{split}-{name}")
        add(f"steelman-{split}-report", "report", [py, "-u", "-m", "analysis.steelman", "report", "--output", sdir], steel_ids, est_cpu_h=0.1)
        if not cfg.get("include_funnel", True):  # tests: drop the funnel and decisions tasks of this split
            tasks[:] = [t for t in tasks if not (t.id.startswith(f"funnel-{split}") or t.id == f"decisions-{split}")]

    # ---------------------------------------------------------------- the bundle
    present = {t.id for t in tasks}
    final_deps = [t.id for t in tasks if t.stage == "report"] + [x for x in ("decisions-exploration", "decisions-confirmation",
                                                                             "intent-analyze", "answers-analyze") if x in present]
    inputs = []
    for split in splits:
        inputs += [f"funnel-{split}={out}/funnel-report-{split}", f"decisions-{split}={out}/decisions-{split}",
                   f"steelman-{split}={out}/steelman-{split}"]
    inputs += [f"intent-stages={out}/intent-stages", f"answers-analysis={out}/answers-analysis", f"funnel-extract={out}/funnel-extract",
               f"trace-extract={out}/trace-extract", f"funnel-assembled={out}/funnel-assembled"]
    add("bundle", "report", ["bash", "-c", f'"$0" -u {script("paper_results.py")} ' + " ".join(f"--input {i}" for i in inputs)
                             + f' --output {out}/paper-results && tar -C {out} --exclude="*.cache" -czf {out}/paper-results.tar.gz paper-results '
                             + " ".join(f"steelman-{s}" for s in splits), py], final_deps, est_cpu_h=0.1)
    return tasks


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    for key in ("code", "output", "sources", "snapshots", "axis_map", "population", "corpus", "features", "archive", "search_root"):
        if key not in cfg:
            raise ValueError(f"config lacks {key}")
    return cfg
