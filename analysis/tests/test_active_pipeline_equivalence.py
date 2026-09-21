"""Output fingerprints captured before cleanup at 5b7b6563b4204876e00bca101f5dbd80966b6eef."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from analysis.interpretability.pipeline.acl_arr_document_analysis import (
    analyze_acl_arr_outcomes,
)
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    iter_experiment_tasks,
)
from analysis.interpretability.pipeline.agentic_judging import (
    render_agentic_judge_prompt,
)
from analysis.interpretability.pipeline.agentic_search import (
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
)
from analysis.scripts import run_agentic_search_integration_smoke as generation
from analysis.scripts.run_acl_arr_vllm import _prepare_primary
from analysis.tests import test_acl_arr_document_analysis as analysis_tests
from analysis.tests.test_agentic_judge_transcript import _plan, _recorded_result

EXPECTED = {
    "analysis": "1422ea80ef97e200a246f6af9e04f20049c0c6e71babd653296bee5f80367a82",
    "primary_tasks": "7722d6c112b7b757a955ff6f21246a147d3999875148aeb42754f0d622c6305d",
    "primary_requests": "f1c449b5731411170bf1370001bb9026303b6b85a8fa541e24888b7cf17a2d29",
    "selected_prompts": "947caf05b8e98b8226d656fac4582d4c22f6a02b7c5a873acb6a8d86dbba479a",
    "cells": "3bb124364ab562f5e0591af3509475319971ea84c7a3becea4edf95d2319741b",
    "smoke_cells": "5527feae86ba2aa4eebdc9395b0be9c5d2fa1d4958964bf719c1b55c0b908c05",
    "shards": "7b81f02b350f68f4f3797b37838ebf15d00a3ea41248ad84ab7be12ad307d9b1",
    "selected_cells": "f105fce26a3fdca889ac87182a58f66fc7a708f79873a3f0e4587203f9936c87",
    "judge_Parallel-Expansion-v1_False": "762d99549bb69d7b8fe6f8186f69a438f4a6ab231c26bb18f62b6d941adbb9ae",
    "judge_prompt_Parallel-Expansion-v1_False": "2c4b991d27f8e4bb62efddda8738a2a4a711f44d425e49b95ddd118cf5a78ba4",
    "judge_Parallel-Expansion-v1_True": "593436cbeb8eed7379ab71c36232d631c2e9229220820aa9871e40a58babb980",
    "judge_prompt_Parallel-Expansion-v1_True": "ae89d3981ce21d4bd2291fbe2e6f79f5f39808c5359c7f5a61731815ca536840",
    "judge_Reactive-Snippet-Loop-v1_False": "04ef537af1f213356bffd697cd7bc7f42b54f162ab7c3e920b91c82f98129e41",
    "judge_prompt_Reactive-Snippet-Loop-v1_False": "2c4b991d27f8e4bb62efddda8738a2a4a711f44d425e49b95ddd118cf5a78ba4",
    "judge_Reactive-Snippet-Loop-v1_True": "22de8155d21829b21e83fcd30fe154451442f2428dd6caced2ea7ca0f4ee1da7",
    "judge_prompt_Reactive-Snippet-Loop-v1_True": "78e022b9aa3ac43b01023c2458eb94eb9b2e7868d4f5bc95358691809c34c99f",
}


def artifact_fingerprints(root: Path) -> dict[str, str]:
    plan, rerank, answers, judgments, mappings = analysis_tests.AclArrAnalysisTests()._fixture()
    artifacts = {
        "analysis": asdict(analyze_acl_arr_outcomes(
            rerank, answers, judgments, mappings, plan=plan, allow_fake=True,
        )),
        "primary_tasks": [asdict(task) for task in iter_experiment_tasks(plan)],
    }
    requests = []
    for task in iter_experiment_tasks(plan):
        request = _prepare_primary(
            task, prompt=plan.prompts[0], assignment=plan.assignments[0],
            document_set=plan.document_sets[0], model=plan.models[0],
        )
        requests.append({key: value for key, value in request.items() if key != "validator"})
    artifacts["primary_requests"] = requests

    prompts = [
        {"candidate_id": f"p-{index}", "question": f"Quel outil choisir {index} ?",
         "keyword": "logiciel", "axis_bin": index % 3}
        for index in range(12)
    ]
    prompt_path = root / "prompts.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    selected = generation._load_calibration_prompts(
        prompt_path, prompt_path, prompt_count=8, seed=20260912,
    )
    cells = generation._cells(selected)
    artifacts["selected_prompts"] = [asdict(prompt) for prompt in selected]
    artifacts["cells"] = [{"cell_id": cell.cell_id, **cell.core} for cell in cells]
    artifacts["smoke_cells"] = [
        {"cell_id": cell.cell_id, **cell.core} for cell in generation._cells()
    ]
    artifacts["shards"] = [
        [prompt.prompt_id for prompt in generation._prompt_shard(
            selected, shard_index=index, shard_count=3,
        )] for index in range(3)
    ]
    cell_path = root / "cells.jsonl"
    cell_path.write_text("".join(
        json.dumps({"cell_id": cell.cell_id}) + "\n" for cell in reversed(cells[::7])
    ))
    artifacts["selected_cells"] = [
        cell.cell_id for cell in generation._select_cells(cells, cell_path)
    ]

    for method in (ParallelExpansionV1, ReactiveSnippetLoopV1):
        result_root = root / method.method_id
        result_path, prompt, _ = _recorded_result(result_root, method)
        result = json.loads(result_path.read_bytes())
        result["trace"] = "../traces/cell.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        for recorded in (False, True):
            judge_plan = _plan(result_root, result_path, prompt, recorded_conversation=recorded)
            value = asdict(judge_plan)
            for mapping in value["mappings"]:
                mapping["source_result_path"] = "results/cell.json"
            artifacts[f"judge_{method.method_id}_{recorded}"] = value
            artifacts[f"judge_prompt_{method.method_id}_{recorded}"] = [
                render_agentic_judge_prompt(task) for task in judge_plan.bulk_tasks
            ]

    return {
        name: hashlib.sha256(json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        for name, value in artifacts.items()
    }


def test_active_pipeline_matches_pre_cleanup_outputs(tmp_path):
    assert artifact_fingerprints(tmp_path) == EXPECTED
