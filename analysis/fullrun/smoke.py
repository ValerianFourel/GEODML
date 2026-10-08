"""Smoke task lists: run the batch scripts' real code paths on HoreKa in a short interactive allocation.

CPU (``smoke --config C --ledger L``): environment and imports in the container; the full-run test suites; an exit-4
checkpoint resumed by the same worker; one real keyword-hash shard (1 of 500) of each extraction; the merge of that
shard; the prompt-text replay. GPU (``smoke --gpu``): four 1-GPU tasks that must each see their own device; CUDA
equivalence of the PyTorch fits with the CPU and torch-cpu results; the real relocation sample and one real 4-GPU
embedding of it (validated profile). Outputs go under ``<config output>/smoke``; nothing of the full run is touched.
"""

from __future__ import annotations

from .ledger import Task

ENV_CHECK = ("import os, numpy, scipy, pandas, pyarrow, sklearn, huggingface_hub; import analysis.fullrun.ledger, "
             "analysis.steelman.tables, analysis.scripts.funnel_study, analysis.scripts.intent_stages_study; "
             "print('env ok', 'commit', os.environ.get('GEODML_GIT_COMMIT'), 'cpus', os.cpu_count(), 'workers', os.environ.get('SLURM_CPUS_PER_TASK'))")
CHECKPOINT = ("import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
              "sys.exit(0) if p.exists() else (p.write_text('first attempt'), sys.exit(4))")
SLOT = ("import os, torch; d = os.environ.get('CUDA_VISIBLE_DEVICES'); n = torch.cuda.device_count(); "
        "print('device', d, 'visible', n, torch.cuda.get_device_name(0)); raise SystemExit(0 if n == 1 else 2)")


FILTER = ("import json, sys; from analysis.scripts.page_readiness_ordering import prompt_shard; "
          "rows = [json.loads(l) for l in open(sys.argv[1])]; keep = [r for r in rows if prompt_shard(r.get('keyword') or '', sys.argv[3])]; "
          "import pathlib; pathlib.Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True); "
          "open(sys.argv[2], 'w').write(''.join(json.dumps(r) + chr(10) for r in keep)); print('prompts in shard', len(keep))")


def chain(cfg: dict, shard: str = "1/500") -> list[Task]:
    """The CPU chain of the full run on real data, one keyword-hash shard (1 of 500, both models): extraction, merges,
    replay, assembly, one funnel stage model (unit ranges and assembly), the generator decisions, every steelman part
    and its report, all with tiny draw counts. Not covered here: the answer export (reads every cell), the intent-stages
    analysis and Gemma extract (need the full embeddings), the Hub import (login node)."""
    code, out = cfg["code"], cfg["output"] + "/smoke-chain"
    py, script = "{PY}", (lambda name: f"{code}/analysis/scripts/{name}")
    sources = [a for s in cfg["sources"] for a in ("--source", s)]
    snaps = [a for e, p in cfg["snapshots"].items() for a in ("--snapshot", f"{e}={p}")]
    archive = cfg["archive"]
    pop = f"{out}/population-1of500.jsonl"
    t = [Task("chain-population", "smoke", [py, "-c", FILTER, cfg["population"], pop, shard], [], cores=1),
         Task("chain-extract-funnel", "smoke", [py, "-u", script("funnel_study.py"), "extract", *sources, *snaps, "--final-axis-map",
                                                cfg["axis_map"], "--population-prompts", cfg["population"], "--workers", "{WORKERS}",
                                                "--prompt-shard", shard, "--output", f"{out}/shard-funnel"], [], cores=16),
         Task("chain-extract-trace", "smoke", [py, "-u", script("intent_stages_study.py"), "trace-extract", *sources, "--final-axis-map",
                                               cfg["axis_map"], "--corpus-package", cfg["corpus"], "--population-prompts", cfg["population"],
                                               "--workers", "{WORKERS}", "--prompt-shard", shard, "--output", f"{out}/shard-trace"], [], cores=16),
         Task("chain-merge-funnel", "smoke", [py, "-u", "-m", "analysis.fullrun", "merge", "funnel", "--output", f"{out}/funnel-extract",
                                              f"{out}/shard-funnel"], ["chain-extract-funnel"], cores=1),
         Task("chain-merge-trace", "smoke", [py, "-u", "-m", "analysis.fullrun", "merge", "trace", "--output", f"{out}/trace-extract",
                                             f"{out}/shard-trace"], ["chain-extract-trace"], cores=1),
         Task("chain-replay", "smoke", [py, "-u", script("funnel_study.py"), "replay", *snaps, "--population-prompts", pop,
                                        "--output", f"{out}/funnel-replay"], ["chain-population"], cores=1),
         Task("chain-assemble", "smoke", [py, "-u", script("funnel_study.py"), "assemble", "--extract", f"{out}/funnel-extract",
                                          "--features", cfg["features"], "--replay", f"{out}/funnel-replay", "--corpus-package", cfg["corpus"],
                                          "--qwen-prompts", f"{archive}/final-audit/projections/qwen",
                                          "--mistral-prompts", f"{archive}/final-audit/projections/mistral", "--qwen-map", f"{archive}/maps/qwen",
                                          "--mistral-map", f"{archive}/maps/mistral", "--output", f"{out}/funnel-assembled"],
              ["chain-merge-funnel", "chain-replay"], cores=4)]
    fbase = [py, "-u", script("funnel_study.py"), "analyze", "--assembled", f"{out}/funnel-assembled", "--output", f"{out}/funnel-analysis",
             "--specs", "main", "--split", "all", "--bootstrap", "4", "--permutations", "3", "--secondary-bootstrap", "2",
             "--workers", "{WORKERS}", "--stop-after-minutes", "{MINUTES}", "--task", "qwen38 · Reactive|P|C|main"]
    t += [Task("chain-funnel-full", "smoke", [*fbase, "--units", "full"], ["chain-assemble"], cores=4),
          Task("chain-funnel-boot", "smoke", [*fbase, "--units", "bootstrap:0-4"], ["chain-funnel-full"], cores=4),
          Task("chain-funnel-perm", "smoke", [*fbase, "--units", "permutation:0-3"], ["chain-funnel-full"], cores=4),
          Task("chain-funnel-drop", "smoke", [*fbase, "--units", "drop"], ["chain-funnel-full"], cores=4),
          Task("chain-funnel-assemble", "smoke", fbase, ["chain-funnel-boot", "chain-funnel-perm", "chain-funnel-drop"], cores=4),
          Task("chain-decisions", "smoke", [py, "-u", script("funnel_importance.py"), "--assembled", f"{out}/funnel-assembled", "--split", "all",
                                            "--bootstrap", "4", "--permutations", "4", "--min-units", "5", "--workers", "{WORKERS}",
                                            "--output", f"{out}/decisions"], ["chain-assemble"], cores=8)]
    steel = [py, "-u", "-m", "analysis.steelman", "PART", "--assembled", f"{out}/funnel-assembled", "--extract", f"{out}/funnel-extract",
             "--replay", f"{out}/funnel-replay", "--trace-extract", f"{out}/trace-extract", "--features", cfg["features"], "--prompts", pop,
             "--split", "all", "--input-root", out, "--output", f"{out}/steelman", "--workers", "{WORKERS}", "--bootstrap", "10",
             "--permutations", "5", "--model-draws", "4", "--mc", "20"]
    part = lambda name, *extra: [a if a != "PART" else name for a in steel] + list(extra)  # noqa: E731
    deps = ["chain-assemble", "chain-merge-trace"]
    ids = []
    for name in ("chain", "followup", "pairs", "census", "supply", "ablation", "queries"):
        t.append(Task(f"chain-steelman-{name}", "smoke", part(name), deps, cores=1)); ids.append(f"chain-steelman-{name}")
    t.append(Task("chain-steelman-lexical", "smoke", part("lexical", "--selection-draws", "2"), deps, cores=4)); ids.append("chain-steelman-lexical")
    for name in ("generator", "fe"):
        sids = []
        for stratum in ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive"):
            tid = f"chain-steelman-{name}-{stratum.replace(' · ', '-').lower()}"
            t.append(Task(tid, "smoke", part(name, "--stratum", stratum), deps, cores=4)); sids.append(tid)
        t.append(Task(f"chain-steelman-{name}", "smoke", part(name), sids, cores=2)); ids.append(f"chain-steelman-{name}")
    t.append(Task("chain-steelman-report", "smoke", [py, "-u", "-m", "analysis.steelman", "report", "--output", f"{out}/steelman"], ids, cores=1))
    return t


def build(cfg: dict, gpu: bool) -> list[Task]:
    code, out = cfg["code"], cfg["output"] + "/smoke"
    py, script = "{PY}", (lambda name: f"{code}/analysis/scripts/{name}")
    sources = [a for s in cfg["sources"] for a in ("--source", s)]
    snaps = [a for e, p in cfg["snapshots"].items() for a in ("--snapshot", f"{e}={p}")]
    tests = f"{code}/analysis/tests"
    if not gpu:
        return [
            Task("smoke-cpu-env", "smoke", [py, "-c", ENV_CHECK], [], cores=1),
            Task("smoke-cpu-tests", "smoke", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{tests}/test_fullrun_ledger.py",
                                              f"{tests}/test_fullrun_merge.py", f"{tests}/test_fullrun_gpu_slots.py",
                                              f"{tests}/test_steelman_chain.py", f"{tests}/test_steelman_horeka.py"], [], cores=8),
            Task("smoke-cpu-checkpoint", "smoke", [py, "-c", CHECKPOINT, f"{out}/checkpoint-marker"], [], cores=1),
            Task("smoke-extract-funnel", "smoke", [py, "-u", script("funnel_study.py"), "extract", *sources, *snaps,
                                                   "--final-axis-map", cfg["axis_map"], "--population-prompts", cfg["population"],
                                                   "--workers", "{WORKERS}", "--prompt-shard", "1/500", "--output", f"{out}/funnel-1of500"],
                 [], cores=16),
            Task("smoke-extract-trace", "smoke", [py, "-u", script("intent_stages_study.py"), "trace-extract", *sources,
                                                  "--final-axis-map", cfg["axis_map"], "--corpus-package", cfg["corpus"],
                                                  "--population-prompts", cfg["population"], "--workers", "{WORKERS}",
                                                  "--prompt-shard", "1/500", "--output", f"{out}/trace-1of500"], [], cores=16),
            Task("smoke-merge", "smoke", [py, "-u", "-m", "analysis.fullrun", "merge", "funnel", "--output", f"{out}/funnel-merged",
                                          f"{out}/funnel-1of500"], ["smoke-extract-funnel"], cores=1),
            Task("smoke-replay", "smoke", [py, "-u", script("funnel_study.py"), "replay", *snaps, "--population-prompts", cfg["population"],
                                           "--output", f"{out}/funnel-replay"], [], cores=1),
        ]
    archive = cfg["archive"]
    embed = f"{code}/analysis/docs/horeka-fullrun-embed.sh"
    return [
        *[Task(f"smoke-gpu-slot-{i}", "smoke", [py, "-c", SLOT], [], cores=2, gpu=True, gpus=1) for i in range(4)],
        Task("smoke-gpu-equivalence", "smoke", [py, "-u", "-m", "analysis.fullrun.gpu_validation", "cuda-smoke"],
             [f"smoke-gpu-slot-{i}" for i in range(4)], cores=4, gpu=True, gpus=1),
        Task("smoke-relocation-sample", "smoke", [py, "-u", script("page_readiness_ordering.py"), "sample-prompts", "--prompts",
                                                  f"{archive}/final-audit/compliant-candidates.jsonl", "--count", "512",
                                                  "--output", f"{out}/prompt-sample"], [], cores=1, gpu=True, gpus=1),
        Task("smoke-embed-sample-qwen", "smoke", ["bash", embed, "qwen", f"{out}/prompt-sample/pages.jsonl.gz", f"{out}/embed-sample-qwen"],
             ["smoke-relocation-sample", "smoke-gpu-equivalence"], cores=0, gpu=True, gpus=0),
    ]
