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
CHECKPOINT = ("import pathlib, sys; p = pathlib.Path(sys.argv[1]); "
              "sys.exit(0) if p.exists() else (p.write_text('first attempt'), sys.exit(4))")
SLOT = ("import os, torch; d = os.environ.get('CUDA_VISIBLE_DEVICES'); n = torch.cuda.device_count(); "
        "print('device', d, 'visible', n, torch.cuda.get_device_name(0)); raise SystemExit(0 if n == 1 else 2)")


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
