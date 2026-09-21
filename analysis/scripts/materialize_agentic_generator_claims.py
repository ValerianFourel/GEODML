#!/usr/bin/env python3
"""Materialize a complete shared generator registry into the four historical shards."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.inference_claims import (
    InferenceClaimStore,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    SmokeInputs,
    _generator_claim_identity,
    _validate_shared_generator_bundle,
    describe_queue,
    run_smoke,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_from_plan(plan: dict, *, output: Path, shard_index: int) -> SmokeInputs:
    study = plan["original_study"]
    model = study["llama"]
    return SmokeInputs(
        output=output,
        base_url="http://127.0.0.1:8010/v1",
        model_id=model["model_id"],
        model_revision=model["model_revision"],
        cross_encoder_snapshot=Path(study["cross_encoder_snapshot"]),
        cross_encoder_revision=study["cross_encoder_revision"],
        search_snapshots={
            name: Path(value["path"])
            for name, value in study["search_snapshots"].items()
        },
        seed=study["inference_seed"],
        max_tokens=model["final_max_tokens"],
        query_max_tokens=model["query_max_tokens"],
        final_max_tokens=model["final_max_tokens"],
        request_concurrency=model["request_concurrency"],
        cell_concurrency=model["cell_concurrency"],
        disable_thinking=model["disable_thinking"],
        prompts_jsonl=Path(study["prompt_sources"]["prompts_jsonl"]["path"]),
        selection_records_jsonl=Path(
            study["prompt_sources"]["selection_records_jsonl"]["path"]
        ),
        prompt_count=500,
        prompt_selection_seed=study["prompt_selection_seed"],
        prompt_shard_index=shard_index,
        prompt_shard_count=4,
        production_conditions=True,
        shared_claim_root=Path(model["claim_root"]),
    )


def require_complete(inputs: SmokeInputs) -> int:
    description = describe_queue(inputs)
    config, cells = description["config"], description["cells"]
    target_urls, legacy_prompt = (
        description["target_urls"],
        description["legacy_prompt"],
    )
    method_hash = _sha256(
        REPOSITORY_ROOT / "analysis/interpretability/pipeline/agentic_search.py"
    )
    store = InferenceClaimStore(inputs.shared_claim_root)
    incomplete = []
    for cell in cells:
        identity = _generator_claim_identity(
            cell,
            inputs,
            config,
            legacy_prompt,
            target_urls,
            method_source_sha256=method_hash,
        )
        prompt = cell.prompt if cell.prompt is not None else legacy_prompt
        state, _ = store.inspect(
            identity,
            validate=lambda value, cell=cell, prompt=prompt: (
                _validate_shared_generator_bundle(cell, prompt, value)
            ),
            validate_failure=lambda value, cell=cell, prompt=prompt: (
                _validate_shared_generator_bundle(cell, prompt, value, failed=True)
            ),
        )
        if state != "completed":
            incomplete.append((cell.cell_id, state))
    if incomplete:
        states: dict[str, int] = {}
        for _, state in incomplete:
            states[state] = states.get(state, 0) + 1
        raise RuntimeError(
            "shared generator barrier is closed: " + json.dumps(states, sort_keys=True)
        )
    return len(cells)


def materialize(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format_version") != "agentic-adaptive-500-plan-v1":
        raise ValueError("unsupported adaptive plan")
    expected = plan["original_study"]["generation_tasks"]
    tasks_path = Path(expected["path"])
    if _sha256(tasks_path) != expected["sha256"] or expected["rows"] != 6000:
        raise ValueError("frozen generation task identity changed")
    root = Path(plan["original_study"]["llama"]["materialized_root"])
    manifests = []
    for shard_index in range(4):
        inputs = inputs_from_plan(
            plan, output=root / f"shard-{shard_index}", shard_index=shard_index
        )
        if require_complete(inputs) != 1500:
            raise AssertionError("historical shard must contain 1500 cells")
        manifest = asyncio.run(run_smoke(inputs))
        if manifest["status"] != "complete" or manifest["completed_count"] != 1500:
            raise RuntimeError("claim materialization did not produce a complete shard")
        manifests.append(str((inputs.output / "run_manifest.json").resolve()))
    return {"status": "complete", "completed": 6000, "manifests": manifests}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.plan.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
