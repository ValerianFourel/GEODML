"""Compare CPU analysis and judge preparation with a local Git baseline.

Run as a module with --baseline-ref and, optionally, --repetitions.
Synthetic fixtures measure Python and local file work, not GPU throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from analysis.interpretability.pipeline import (
    acl_arr_document_analysis as analysis,
    agentic_judging as judging,
)
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    iter_experiment_tasks,
)
from analysis.tests.test_acl_arr_document_analysis import AclArrAnalysisTests
from analysis.tests.test_agentic_judge_transcript import _recorded_result

ROOT = Path(__file__).resolve().parents[2]


def _baseline(name: str, commit: str) -> ModuleType:
    path = f"analysis/interpretability/pipeline/{name}.py"
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    module = ModuleType(f"analysis.interpretability.pipeline._baseline_{name}")
    module.__file__ = str(ROOT / path)
    sys.modules[module.__name__] = module
    exec(compile(source, f"{commit}:{path}", "exec"), module.__dict__)
    return module


def _analysis_fixture(count: int):
    plan, rerank, answers, judgments, mappings = AclArrAnalysisTests()._fixture()
    prompts = tuple(replace(plan.prompts[0], prompt_id=f"p-{index}") for index in range(count))
    plan = replace(plan, prompts=prompts, assignments=tuple(
        replace(plan.assignments[0], prompt_id=prompt.prompt_id, assignment_id=f"a-{index}")
        for index, prompt in enumerate(prompts)
    ))
    original = {(row["pipeline"], row["condition"]): row for row in rerank + answers}
    judge_by_id = {row["judge_task_id"]: row for row in judgments}
    judge_by_answer = {
        row["source_generation_task_id"]: judge_by_id[row["judge_task_id"]]
        for row in mappings
    }
    new_rerank, new_answers, new_judgments, new_mappings = [], [], [], []
    for task in iter_experiment_tasks(plan):
        source = original[task.pipeline, task.condition]
        row = {**source, "task_id": task.task_id, "prompt_id": task.prompt_id,
               "assignment_id": task.assignment_id}
        if task.pipeline == "rerank":
            new_rerank.append(row)
        else:
            new_answers.append(row)
            judge_id = f"judge-{task.task_id}"
            new_judgments.append({**judge_by_answer[source["task_id"]], "judge_task_id": judge_id})
            new_mappings.append({"judge_task_id": judge_id, "source_generation_task_id": task.task_id})
    return plan, (new_rerank, new_answers, new_judgments, new_mappings)


def _measure(call):
    gc.collect()
    started = time.perf_counter()
    value = asdict(call())
    return value, time.perf_counter() - started


def _compare(before, after, repetitions: int):
    times = {"before": [], "after": []}
    for repetition in range(repetitions):
        order = (("before", before), ("after", after))
        outputs = {}
        for name, call in order if repetition % 2 == 0 else reversed(order):
            outputs[name], elapsed = _measure(call)
            times[name].append(elapsed)
        assert outputs["before"] == outputs["after"], "serialized outputs differ"
    peak_bytes = {}
    for name, call in (("before", before), ("after", after)):
        gc.collect()
        tracemalloc.start()
        value = call()
        peak_bytes[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del value
    return {
        "outputs_equal": True,
        "seconds": times,
        "median_seconds": {name: statistics.median(values) for name, values in times.items()},
        "peak_python_bytes": peak_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline_ref}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    old_analysis = _baseline("acl_arr_document_analysis", commit)
    old_judging = _baseline("agentic_judging", commit)
    plan, rows = _analysis_fixture(1000)
    report = {
        "baseline_commit": commit, "scientific_result": False, "gpu_performance_measured": False,
        "analysis_prompts": 1000, "judge_results": 100,
        "analysis": _compare(
            lambda: old_analysis.analyze_acl_arr_outcomes(*rows, plan=plan, allow_fake=True),
            lambda: analysis.analyze_acl_arr_outcomes(*rows, plan=plan, allow_fake=True),
            args.repetitions,
        ),
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source_path, prompt, _ = _recorded_result(root)
        source = json.loads(source_path.read_bytes())
        paths = []
        for index in range(100):
            path = root / "results" / f"item-{index}.json"
            path.write_text(json.dumps({**source, "cell_id": f"cell-{index}"}), encoding="utf-8")
            paths.append(path)
        kwargs = {
            "prompt_rows": [prompt], "generator_model_by_root": {str(root): "generator"},
            "bulk_model": judging.AgenticJudgeModel("bulk", "judge", "a" * 40),
            "validation_model": judging.AgenticJudgeModel("validation", "validator", "b" * 40),
            "recorded_conversation": True,
        }
        calls = {
            "before": lambda: old_judging.build_agentic_judge_plan(paths, **kwargs),
            "after": lambda: judging.build_agentic_judge_plan(paths, **kwargs),
        }
        report["judging"] = _compare(calls["before"], calls["after"], args.repetitions)
        counts = {}
        original_open = Path.open
        result_paths = set(paths)
        for name, call in calls.items():
            reads = 0

            def counted_open(path, *positional, **keywords):
                nonlocal reads
                if path in result_paths:
                    reads += 1
                return original_open(path, *positional, **keywords)

            with patch.object(Path, "open", counted_open):
                call()
            counts[name] = reads
        report["judging"]["result_file_reads"] = counts
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
