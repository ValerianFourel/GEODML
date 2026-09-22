from analysis.scripts.report_agentic_500_pilot_results import (
    _judge_claim_arguments,
    build_report,
    render_text,
)
from analysis.scripts.run_acl_arr_vllm import _parser as inference_parser

METHODS = ("Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1")
ENGINES = ("duckduckgo", "searxng")
CONDITIONS = ("natural", "ablated", "shuffled")
MODELS = {
    "qwen38": "Qwen/Qwen3.8-27B",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}


def test_report_reconstructs_the_exact_worker_claim_contract():
    worker = inference_parser().parse_args(
        [
            "agentic-judge",
            "--tasks",
            "tasks.jsonl",
            "--output-dir",
            "output",
            "--judge-manifest",
            "run_manifest.json",
            "--judge-role",
            "bulk",
            "--disable-thinking",
            "--max-attempts",
            "3",
            "--request-timeout",
            "120",
        ]
    )
    report = _judge_claim_arguments()

    for field in (
        "judge_role",
        "disable_thinking",
        "fake",
        "pilot_only",
        "max_attempts",
        "request_timeout",
    ):
        assert getattr(report, field) == getattr(worker, field)
        assert type(getattr(report, field)) is type(getattr(worker, field))


def frozen_cells():
    return [
        {
            "cell_id": f"{prompt}-{method}-{engine}-{condition}",
            "prompt_id": prompt,
            "prompt_sha256": f"hash-{prompt}",
            "method": method,
            "engine": engine,
            "condition": condition,
        }
        for prompt in ("prompt-0", "prompt-1")
        for method in METHODS
        for engine in ENGINES
        for condition in CONDITIONS
    ]


def judge_mappings(cells):
    rows = []
    for alias, model_id in MODELS.items():
        for cell in cells:
            rows.append(
                {
                    "judge_task_id": f"judge-{alias}-{cell['cell_id']}",
                    "source_cell_id": cell["cell_id"],
                    "prompt_id": cell["prompt_id"],
                    "generator_model_id": model_id,
                    "method": cell["method"],
                    "engine": cell["engine"],
                    "condition": cell["condition"],
                }
            )
    return rows


def test_report_shows_exact_generation_and_nemotron_coverage():
    cells = frozen_cells()
    mappings = judge_mappings(cells)
    qwen = {row["cell_id"] for row in cells}
    llama = {
        row["cell_id"]
        for row in cells
        if row["prompt_id"] == "prompt-0"
        or row["method"] == "Parallel-Expansion-v1"
    }
    llama_failure = next(
        row["cell_id"]
        for row in cells
        if row["prompt_id"] == "prompt-1"
        and row["method"] == "Reactive-Snippet-Loop-v1"
    )
    judge_states = {}
    for row in mappings:
        if row["prompt_id"] == "prompt-0":
            judge_states[row["judge_task_id"]] = "completed"
    prompt_1 = [row for row in mappings if row["prompt_id"] == "prompt-1"]
    for row in prompt_1[:6]:
        judge_states[row["judge_task_id"]] = "completed"
    judge_states[prompt_1[6]["judge_task_id"]] = "failed"
    judge_states[prompt_1[7]["judge_task_id"]] = "busy"

    report = build_report(
        generation_tasks=cells,
        generator_completed={"qwen38": qwen, "llama4": llama},
        generator_failed={"qwen38": set(), "llama4": {llama_failure}},
        judge_mappings=mappings,
        judge_states=judge_states,
        generator_models=MODELS,
        expected_prompt_count=2,
    )

    assert report["plan"] == {
        "prompts": 2,
        "cells_per_model": 24,
        "generator_tasks": 48,
        "judge_tasks": 48,
        "cells_per_prompt_per_model": 12,
    }
    assert report["generation"]["overall"] == {
        "expected": 48,
        "completed": 42,
        "failed": 1,
        "missing": 5,
        "percent_completed": 87.5,
    }
    assert report["generation"]["models"]["qwen38"]["prompts"] == {
        "complete": 2,
        "partial": 0,
        "untouched": 0,
    }
    assert report["generation"]["models"]["llama4"]["prompts"] == {
        "complete": 1,
        "partial": 1,
        "untouched": 0,
    }
    assert report["generation"]["models"]["llama4"]["by_method"][
        "Reactive-Snippet-Loop-v1"
    ] == {
        "expected": 12,
        "completed": 6,
        "failed": 1,
        "missing": 5,
        "percent_completed": 50.0,
    }
    assert report["nemotron"]["overall"] == {
        "expected": 48,
        "completed": 30,
        "failed": 1,
        "busy": 1,
        "missing": 16,
        "unresolved": 18,
        "percent_completed": 62.5,
    }
    assert report["nemotron"]["prompts"] == {
        "complete": 1,
        "partial": 1,
        "untouched": 0,
    }
    assert report["experiment_complete"] is False

    text = render_text(report)
    assert "GENERATION 42/48 (87.50%) failed=1 missing=5" in text
    assert "NEMOTRON 30/48 (62.50%) failed=1 busy=1 missing=16" in text
    assert "PROMPTS fully_judged=1/2 partial=1 untouched=0" in text
    assert "EXPERIMENT_COMPLETE=NO" in text


def test_report_rejects_an_incomplete_factorial_plan():
    cells = frozen_cells()[:-1]

    try:
        build_report(
            generation_tasks=cells,
            generator_completed={"qwen38": set(), "llama4": set()},
            generator_failed={"qwen38": set(), "llama4": set()},
            judge_mappings=[],
            judge_states={},
            generator_models=MODELS,
            expected_prompt_count=2,
        )
    except ValueError as error:
        assert "factorial" in str(error)
    else:
        raise AssertionError("an incomplete prompt factorial must be rejected")
