"""Tests for the agentic-search integration smoke runner."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from analysis.interpretability.pipeline.agentic_search import (
    AgentExecutionError,
    ContextCompactor,
    ExperimentalCondition,
    IdentityConditionHook,
    LexicalOverlapScorer,
    LLMRequest,
    MemoizingSnippetScorer,
    ParallelExpansionV1,
    Snippet,
    StaticSearchAdapter,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    CalibrationPrompt,
    FrozenSnapshotSearchAdapter,
    SmokeInputs,
    SmokeConditionHook,
    TargetUrlConditionHook,
    VllmAgentGenerator,
    _build_target_urls,
    _cells,
    _evidence_id_resume_config,
    _legacy_resume_config,
    _load_calibration_prompts,
    _prompt_shard,
    _prepare_config,
    _previous_resume_config,
    _repair_resume_config,
    _serial_resume_config,
    run_smoke,
)


class FrozenSnapshotSearchAdapterTests(unittest.TestCase):
    def test_previous_resume_config_migrates_only_the_exact_predecessor(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            current = {
                "git_commit": "new-commit",
                "disable_thinking": True,
                "request_concurrency": 1,
                "execution_policy": {
                    "max_tokens_by_purpose": {
                        "parallel_query_expansion": 256,
                        "parallel_final": 4096,
                        "reactive_action": 4096,
                        "reactive_forced_finish": 4096,
                    },
                    "ranking_reference_mode": "evidence-id-v1",
                    "final_answer_max_characters": 1200,
                    "final_attempt_repair_mode": "evidence-projection-v1",
                    "structured_output_schema_mode": "xgrammar-structural-v1",
                    "maximum_active_cells": 1,
                },
            }
            current_hash = hashlib.sha256(json.dumps(
                current,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            previous = _previous_resume_config(current)
            self.assertIsNotNone(previous)
            previous_hash = hashlib.sha256(json.dumps(
                previous,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            path.write_text(json.dumps({
                **previous,
                "config_sha256": previous_hash,
            }), encoding="utf-8")

            compatible, migrated = _prepare_config(
                path,
                current,
                current_hash,
            )

            self.assertIsNotNone(migrated)
            self.assertEqual(migrated["config_sha256"], previous_hash)
            self.assertIn(previous_hash, {
                source["config_sha256"] for source in compatible
            })
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {**current, "config_sha256": current_hash},
            )

            evidence_id = _evidence_id_resume_config(current)
            self.assertIsNotNone(evidence_id)
            self.assertEqual(
                evidence_id["execution_policy"]["max_tokens_by_purpose"],
                {
                    "parallel_query_expansion": 256,
                    "parallel_final": 2048,
                    "reactive_action": 2048,
                    "reactive_forced_finish": 2048,
                },
            )
            evidence_id_hash = hashlib.sha256(json.dumps(
                evidence_id,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            path.write_text(json.dumps({
                **evidence_id,
                "config_sha256": evidence_id_hash,
            }), encoding="utf-8")

            compatible, migrated = _prepare_config(
                path,
                current,
                current_hash,
            )

            self.assertEqual(migrated["config_sha256"], evidence_id_hash)
            self.assertIn(evidence_id_hash, {
                source["config_sha256"] for source in compatible
            })

            repair = _repair_resume_config(current)
            self.assertIsNotNone(repair)
            self.assertEqual(repair["request_concurrency"], 4)
            self.assertEqual(
                repair["execution_policy"]["maximum_active_cells"],
                4,
            )
            self.assertNotIn(
                "final_attempt_repair_mode",
                repair["execution_policy"],
            )

            serial = _serial_resume_config(current)
            self.assertIsNotNone(serial)
            self.assertEqual(
                serial["git_commit"],
                "6a2469958df8355234982a45608c220780b855a0",
            )
            self.assertEqual(serial["request_concurrency"], 1)
            self.assertNotIn(
                "structured_output_schema_mode",
                serial["execution_policy"],
            )
            serial_hash = hashlib.sha256(json.dumps(
                serial,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            path.write_text(json.dumps({
                **serial,
                "config_sha256": serial_hash,
            }), encoding="utf-8")

            compatible, migrated = _prepare_config(
                path,
                current,
                current_hash,
            )

            self.assertEqual(migrated["config_sha256"], serial_hash)
            self.assertIn(serial_hash, {
                source["config_sha256"] for source in compatible
            })

    def test_global_1024_budget_reproduces_parallel_final_truncation(self) -> None:
        client = _FakeClientContext(truncate_final_at=1024)

        async def execute() -> None:
            generator = VllmAgentGenerator(
                client,
                seed=11,
                query_max_tokens=1024,
                final_max_tokens=1024,
                request_semaphore=asyncio.Semaphore(1),
            )
            method = _parallel_method(generator)
            with self.assertRaisesRegex(
                AgentExecutionError,
                "schema validation failed after 3 attempts",
            ):
                await method.run(
                    "What is Berlin's population?",
                    ExperimentalCondition.NATURAL,
                )

        asyncio.run(execute())

        self.assertEqual(
            [
                max_tokens
                for purpose, max_tokens in client.calls
                if purpose == "parallel_final"
            ],
            [1024, 1024, 1024],
        )

    def test_final_purpose_gets_larger_budget_without_inflating_queries(self) -> None:
        client = _FakeClientContext(truncate_final_at=1024)

        async def execute():
            generator = VllmAgentGenerator(
                client,
                seed=11,
                query_max_tokens=256,
                final_max_tokens=2048,
                request_semaphore=asyncio.Semaphore(1),
            )
            result = await _parallel_method(generator).run(
                "What is Berlin's population?",
                ExperimentalCondition.NATURAL,
            )
            for purpose in ("reactive_action", "reactive_forced_finish"):
                await generator.generate(LLMRequest(
                    purpose=purpose,
                    prompt=(
                        "OBSERVATIONS:\n[]"
                        if purpose == "reactive_action"
                        else "test"
                    ),
                    response_schema={},
                ))
            with self.assertRaisesRegex(ValueError, "unsupported LLM request purpose"):
                await generator.generate(LLMRequest(
                    purpose="unknown",
                    prompt="test",
                    response_schema={},
                ))
            return result

        result = asyncio.run(execute())

        self.assertEqual(result.answer, "Parallel answer")
        self.assertEqual(client.calls, [
            ("parallel_query_expansion", 256),
            ("parallel_final", 2048),
            ("reactive_action", 2048),
            ("reactive_forced_finish", 2048),
        ])

    def test_calibration_prompt_selection_covers_all_axis_bins(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "pilot-prompts.jsonl"
            records_path = root / "selection-records.jsonl"
            prompt_rows = []
            record_rows = []
            for axis_bin in range(20):
                for copy in range(2):
                    candidate_id = f"prompt-{axis_bin}-{copy}"
                    question = f"Question for axis bin {axis_bin}, copy {copy}?"
                    prompt_rows.append({
                        "candidate_id": candidate_id,
                        "keyword": f"keyword-{axis_bin}",
                        "question": question,
                        "question_sha256": hashlib.sha256(
                            question.encode("utf-8")
                        ).hexdigest(),
                    })
                    record_rows.append({
                        "candidate_id": candidate_id,
                        "axis_bin": axis_bin,
                    })
            prompts_path.write_text(
                "".join(json.dumps(row) + "\n" for row in prompt_rows),
                encoding="utf-8",
            )
            records_path.write_text(
                "".join(json.dumps(row) + "\n" for row in record_rows),
                encoding="utf-8",
            )

            selected = _load_calibration_prompts(
                prompts_path,
                records_path,
                prompt_count=25,
                seed=20260912,
            )
            repeated = _load_calibration_prompts(
                prompts_path,
                records_path,
                prompt_count=25,
                seed=20260912,
            )
            cells = _cells(selected)

        counts = Counter(prompt.axis_bin for prompt in selected)
        self.assertEqual(len(selected), 25)
        self.assertEqual(set(counts), set(range(20)))
        self.assertEqual(sorted(counts.values()), [1] * 15 + [2] * 5)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(cells), 300)
        self.assertEqual(len({cell.cell_id for cell in cells}), 300)
        self.assertEqual(
            {cell.prompt_id for cell in cells},
            {prompt.prompt_id for prompt in selected},
        )

    def test_two_prompt_shards_are_disjoint_and_exhaustive(self) -> None:
        prompts = tuple(
            CalibrationPrompt(
                prompt_id=f"prompt-{index:03d}",
                prompt=f"Question {index}?",
                question_sha256=f"{index:064x}",
                axis_bin=index % 20,
                keyword=f"keyword-{index}",
            )
            for index in range(500)
        )

        left = _prompt_shard(prompts, shard_index=0, shard_count=2)
        right = _prompt_shard(prompts, shard_index=1, shard_count=2)
        left_ids = {prompt.prompt_id for prompt in left}
        right_ids = {prompt.prompt_id for prompt in right}

        self.assertEqual(len(left), 250)
        self.assertEqual(len(right), 250)
        self.assertFalse(left_ids & right_ids)
        self.assertEqual(
            left_ids | right_ids,
            {prompt.prompt_id for prompt in prompts},
        )

    def test_four_production_shards_have_disjoint_prompt_and_cell_ids(self) -> None:
        prompts = tuple(
            CalibrationPrompt(
                prompt_id=f"prompt-{index:03d}",
                prompt=f"Question {index}?",
                question_sha256=f"{index:064x}",
                axis_bin=index % 20,
                keyword=f"keyword-{index}",
            )
            for index in range(500)
        )

        shards = [
            _prompt_shard(prompts, shard_index=index, shard_count=4)
            for index in range(4)
        ]
        prompt_id_sets = [
            {prompt.prompt_id for prompt in shard}
            for shard in shards
        ]
        cell_id_sets = [
            {cell.cell_id for cell in _cells(shard)}
            for shard in shards
        ]

        self.assertEqual([len(shard) for shard in shards], [125] * 4)
        self.assertEqual([len(cell_ids) for cell_ids in cell_id_sets], [1500] * 4)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(prompt_id_sets[left] & prompt_id_sets[right])
                self.assertFalse(cell_id_sets[left] & cell_id_sets[right])
        self.assertEqual(
            set().union(*prompt_id_sets),
            {prompt.prompt_id for prompt in prompts},
        )

    def test_twenty_five_prompt_calibration_runs_and_resumes_300_cells(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "pilot-prompts.jsonl"
            records_path = root / "selection-records.jsonl"
            prompts = []
            records = []
            for axis_bin in range(20):
                for copy in range(2):
                    candidate_id = f"prompt-{axis_bin}-{copy}"
                    question = f"What is the Berlin population for bin {axis_bin}?"
                    prompts.append({
                        "candidate_id": candidate_id,
                        "keyword": "Berlin population",
                        "question": question,
                        "question_sha256": hashlib.sha256(
                            question.encode("utf-8")
                        ).hexdigest(),
                    })
                    records.append({
                        "candidate_id": candidate_id,
                        "axis_bin": axis_bin,
                    })
            prompts_path.write_text(
                "".join(json.dumps(row) + "\n" for row in prompts),
                encoding="utf-8",
            )
            records_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            snapshots = {}
            for engine in ("duckduckgo", "searxng"):
                path = root / f"{engine}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({
                            "keyword": "Berlin population",
                            "position": index + 1,
                            "title": f"Berlin source {index}",
                            "url": f"https://{engine}.test/{index}",
                            "snippet": f"Berlin population evidence {index}",
                        }) + "\n"
                        for index in range(20)
                    ),
                    encoding="utf-8",
                )
                snapshots[engine] = path
            cross_encoder = root / ("e" * 40)
            cross_encoder.mkdir()
            inputs = SmokeInputs(
                output=root / "output",
                base_url="http://127.0.0.1:8010/v1",
                model_id="Qwen/Qwen3.8-27B",
                model_revision="a" * 40,
                cross_encoder_snapshot=cross_encoder,
                cross_encoder_revision="e" * 40,
                search_snapshots=snapshots,
                seed=11,
                max_tokens=1024,
                query_max_tokens=256,
                request_concurrency=4,
                disable_thinking=True,
                prompts_jsonl=prompts_path,
                selection_records_jsonl=records_path,
                prompt_count=25,
                prompt_selection_seed=20260912,
            )
            client = _FakeClientContext()
            manifest = asyncio.run(run_smoke(
                inputs,
                client_context=client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            resume_client = _FakeClientContext()
            resumed = asyncio.run(run_smoke(
                inputs,
                client_context=resume_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            result_count = len(list((inputs.output / "results").glob("*.json")))
            current_config_record = json.loads(
                (inputs.output / "config.json").read_text(encoding="utf-8")
            )
            current_config = dict(current_config_record)
            current_config.pop("config_sha256")
            legacy_config = _legacy_resume_config(current_config)
            self.assertIsNotNone(legacy_config)
            legacy_config_hash = hashlib.sha256(json.dumps(
                legacy_config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            removed_results = sorted(
                (inputs.output / "results").glob("*.json")
            )[-14:]
            retained_path = sorted(
                (inputs.output / "results").glob("*.json")
            )[0]
            retained_bytes = retained_path.read_bytes()
            for result_path in removed_results:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                Path(result["trace"]).unlink()
                result_path.unlink()
            (inputs.output / "config.json").write_text(
                json.dumps(
                    {**legacy_config, "config_sha256": legacy_config_hash},
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            legacy_manifest = {
                **legacy_config,
                "config_sha256": legacy_config_hash,
                "status": "checkpointed",
                "completed_count": 286,
                "remaining_count": 14,
                "compactor_cache_this_invocation": manifest[
                    "compactor_cache_this_invocation"
                ],
            }
            (inputs.output / "run_manifest.json").write_text(
                json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            migration_client = _FakeClientContext()
            migrated = asyncio.run(run_smoke(
                inputs,
                client_context=migration_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            retained_after_migration = retained_path.read_bytes()

        self.assertEqual(
            manifest["format_version"],
            "agentic-search-execution-calibration-v2",
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["prompt_count"], 25)
        self.assertEqual(manifest["cell_count"], 300)
        self.assertEqual(manifest["completed_count"], 300)
        self.assertEqual(manifest["remaining_count"], 0)
        self.assertNotIn("slurm_job_id", manifest)
        self.assertEqual(sorted(manifest["axis_bin_counts"].values()), [1] * 15 + [2] * 5)
        self.assertEqual(result_count, 300)
        self.assertEqual(manifest["peak_active_cells_this_invocation"], 4)
        self.assertEqual(resumed, manifest)
        self.assertEqual(resume_client.call_count, 0)
        self.assertEqual(migration_client.call_count, 28)
        self.assertEqual(migrated["status"], "complete")
        self.assertEqual(migrated["resume_migration"]["preserved_completed_count"], 286)
        self.assertEqual(retained_after_migration, retained_bytes)

    def test_smoke_request_concurrency_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "from 1 to 4"):
            SmokeInputs(
                output=Path("output"),
                base_url="http://127.0.0.1:8010/v1",
                model_id="test/model",
                model_revision="a" * 40,
                cross_encoder_snapshot=Path("cross-encoder"),
                cross_encoder_revision="e" * 40,
                search_snapshots={},
                seed=11,
                max_tokens=128,
                request_concurrency=5,
            )

    def test_cell_concurrency_keeps_request_concurrency_bounded(self) -> None:
        with TemporaryDirectory() as directory:
            inputs = replace(
                _smoke_inputs(Path(directory)),
                cell_concurrency=12,
            )
            client = _FakeClientContext(delay_seconds=0.001)

            manifest = asyncio.run(run_smoke(
                inputs,
                client_context=client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))

        self.assertEqual(manifest["request_concurrency"], 4)
        self.assertEqual(manifest["cell_concurrency"], 12)
        self.assertEqual(
            manifest["execution_policy"]["maximum_active_cells"],
            12,
        )
        self.assertEqual(manifest["peak_active_cells_this_invocation"], 12)
        self.assertEqual(client.peak_active_calls, 4)

    def test_exhausted_cell_does_not_cancel_peers_and_only_it_resumes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _smoke_inputs(root)
            client = _FakeClientContext(
                delay_seconds=0.001,
                failed_final_passes=2,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "1 agentic-search cells failed after bounded retry",
            ):
                asyncio.run(run_smoke(
                    inputs,
                    client_context=client,
                    compactor=ContextCompactor(
                        MemoizingSnippetScorer(LexicalOverlapScorer())
                    ),
                ))
            failed_manifest = json.loads(
                (inputs.output / "run_manifest.json").read_text(encoding="utf-8")
            )
            failed_trace_count = len(list(
                (inputs.output / "failed_traces").glob("*/*.json")
            ))
            retained_path = next((inputs.output / "results").glob("*.json"))
            retained_bytes = retained_path.read_bytes()
            resume_client = _FakeClientContext()
            completed_manifest = asyncio.run(run_smoke(
                inputs,
                client_context=resume_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            retained_after_resume = retained_path.read_bytes()

        self.assertEqual(failed_manifest["status"], "complete_with_failures")
        self.assertEqual(failed_manifest["completed_count"], 11)
        self.assertEqual(failed_manifest["remaining_count"], 1)
        self.assertEqual(len(failed_manifest["failed_cell_ids"]), 1)
        self.assertEqual(failed_manifest["failed_cell_retry_passes_completed"], 1)
        self.assertEqual(failed_manifest["peak_active_cells_this_invocation"], 4)
        self.assertEqual(client.failed_final_calls, 6)
        self.assertEqual(failed_trace_count, 1)
        self.assertEqual(completed_manifest["status"], "complete")
        self.assertEqual(completed_manifest["completed_count"], 12)
        self.assertEqual(resume_client.call_count, 2)
        self.assertEqual(retained_after_resume, retained_bytes)

    def test_cluster_launcher_forces_offline_model_resolution(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])
        self.assertIn("--query-max-tokens 256", launcher)
        self.assertIn("--final-max-tokens 2048", launcher)
        self.assertIn("--disable-thinking", launcher)
        self.assertIn("SEARCH_AGENTIC_CELL_CONCURRENCY", launcher)

    def test_qwen25_launcher_pins_model_and_yarn_profile(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_qwen25_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])
        self.assertIn("Qwen/Qwen2.5-72B-Instruct", launcher)
        self.assertIn("495f39366efef23836d0cfae4fbe635880d2be31", launcher)
        self.assertIn('"rope_type": "yarn"', launcher)
        self.assertIn("AGENTIC_QWEN25_12_CELL_SMOKE=PASS", launcher)

    def test_llama4_launcher_pins_model_and_offline_resolution(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])
        self.assertIn("meta-llama/Llama-4-Scout-17B-16E-Instruct", launcher)
        self.assertIn("92f3b1597a195b523d8d9e5700e57e4fbb8f20d3", launcher)
        self.assertIn("AGENTIC_LLAMA4_12_CELL_SMOKE=PASS", launcher)
        self.assertIn("SEARCH_AGENTIC_CELL_CONCURRENCY", launcher)

    def test_calibration_launcher_runs_three_models_and_900_cells(self) -> None:
        scripts = Path(__file__).resolve().parents[1] / "scripts/slurm/jupiter"
        launcher = (
            scripts / "run_agentic_search_25_prompt_calibration.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("SEARCH_AGENTIC_PROMPT_COUNT=25", launcher)
        self.assertIn("SEARCH_AGENTIC_EXPECTED_CELL_COUNT=300", launcher)
        self.assertIn("SEARCH_AGENTIC_REQUEST_CONCURRENCY", launcher)
        self.assertIn("run_agentic_search_qwen38_smoke.sh", launcher)
        self.assertIn("run_agentic_search_qwen25_smoke.sh", launcher)
        self.assertIn("run_agentic_search_llama4_smoke.sh", launcher)
        self.assertIn('"cell_count": 900', launcher)
        self.assertIn('"scientific_result": False', launcher)
        self.assertIn('"baseline_started": False', launcher)
        self.assertIn('"judge_started": False', launcher)

        final_budgets = {
            "run_agentic_search_qwen38_smoke.sh": 2048,
            "run_agentic_search_qwen25_smoke.sh": 2048,
            "run_agentic_search_llama4_smoke.sh": 4096,
        }
        for name, final_budget in final_budgets.items():
            model_launcher = (scripts / name).read_text(encoding="utf-8")
            self.assertIn("--prompts-jsonl", model_launcher)
            self.assertIn("--selection-records-jsonl", model_launcher)
            self.assertIn("SEARCH_AGENTIC_EXPECTED_CELL_COUNT", model_launcher)
            self.assertIn("--query-max-tokens 256", model_launcher)
            self.assertIn(
                f"--final-max-tokens {final_budget}",
                model_launcher,
            )
        self.assertIn(
            'legacy["git_commit"] = "b561f1aaf54971a89fa2dabb7f3f9d32770ce8cc"',
            launcher,
        )

    def test_nemotron_launcher_pins_model_and_disables_thinking(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_nemotron_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")
        first_git_call = launcher.index("git rev-parse HEAD")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])
        self.assertIn("module load git", launcher[:first_git_call])
        self.assertIn(
            'source "${ACL_ARR_VENV:?}/bin/activate"',
            launcher[:first_git_call],
        )
        self.assertIn("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", launcher)
        self.assertIn("2dc98e2afe4face0e4ce40972a915c45368bd34a", launcher)
        self.assertIn("--disable-thinking", launcher)
        self.assertIn("AGENTIC_NEMOTRON_12_CELL_SMOKE=PASS", launcher)
        self.assertIn('SEARCH_AGENTIC_REQUEST_CONCURRENCY:-1', launcher)

    def test_search_is_bounded_relevant_and_audited(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.jsonl"
            rows = [
                {
                    "keyword": "Berlin population",
                    "position": 1,
                    "title": "Berlin census",
                    "url": "https://example.test/berlin",
                    "snippet": "Population figures for Berlin",
                },
                {
                    "keyword": "weather",
                    "position": 1,
                    "title": "Forecast",
                    "url": "https://example.test/weather",
                    "snippet": "Rain tomorrow",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            adapter = FrozenSnapshotSearchAdapter("duckduckgo", path)
            result = asyncio.run(adapter.search("Berlin population", 1))

        self.assertEqual(result.engine, "duckduckgo")
        self.assertEqual(len(result.snippets), 1)
        self.assertEqual(result.snippets[0].url, "https://example.test/berlin")
        self.assertEqual(result.raw_payload["selection"], "deterministic-lexical-v1")
        self.assertEqual(len(result.raw_payload["snapshot_sha256"]), 64)

    def test_search_excludes_null_position_rows_and_audits_them(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.jsonl"
            rows = [
                {
                    "keyword": "Berlin population",
                    "position": None,
                    "title": None,
                    "url": None,
                    "snippet": None,
                },
                {
                    "keyword": "Berlin population",
                    "position": 1,
                    "title": "Berlin census",
                    "url": "https://example.test/berlin",
                    "snippet": "Population figures for Berlin",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            adapter = FrozenSnapshotSearchAdapter("searxng", path)
            result = asyncio.run(adapter.search("Berlin population", 20))

        self.assertEqual(
            [snippet.url for snippet in result.snippets],
            ["https://example.test/berlin"],
        )
        self.assertEqual(result.raw_payload["snapshot_rows"], {
            "total": 2,
            "usable": 1,
            "excluded": 1,
            "exclusion_reasons": {"invalid_position": 1},
        })

    def test_smoke_conditions_do_not_introduce_evidence(self) -> None:
        rows = [
            Snippet(f"https://example.test/{index}", f"Title {index}", "Text")
            for index in range(8)
        ]
        hook = SmokeConditionHook(7)

        natural = list(hook.apply(ExperimentalCondition.NATURAL, rows))
        ablated = list(hook.apply(ExperimentalCondition.ABLATED, rows))
        shuffled_once = list(hook.apply(ExperimentalCondition.SHUFFLED, rows))
        shuffled_twice = list(hook.apply(ExperimentalCondition.SHUFFLED, rows))

        self.assertEqual(natural, rows)
        self.assertEqual(ablated, [rows[index] for index in (1, 2, 3, 5, 6, 7)])
        self.assertEqual(shuffled_once, shuffled_twice)
        self.assertEqual(set(shuffled_once), set(rows))
        self.assertNotEqual(shuffled_once, rows)

    def test_production_conditions_remove_only_the_frozen_target(self) -> None:
        rows = [
            Snippet(f"https://example.test/{index}", f"Title {index}", "Text")
            for index in range(8)
        ]
        target = rows[3].url

        natural_hook = TargetUrlConditionHook(target_url=target, seed=7)
        ablated_hook = TargetUrlConditionHook(target_url=target, seed=7)
        shuffled_hook = TargetUrlConditionHook(target_url=target, seed=7)
        natural = list(natural_hook.apply(ExperimentalCondition.NATURAL, rows))
        ablated = list(ablated_hook.apply(ExperimentalCondition.ABLATED, rows))
        shuffled = list(shuffled_hook.apply(ExperimentalCondition.SHUFFLED, rows))

        self.assertEqual(natural, rows)
        self.assertEqual(set(rows) - set(ablated), {rows[3]})
        self.assertEqual(set(shuffled), set(rows))
        self.assertEqual(len(shuffled), len(rows))
        self.assertEqual(ablated_hook.calls[0]["target_removed_count"], 1)

    def test_target_urls_are_balanced_from_each_frozen_engine(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            adapters = {}
            for engine in ("duckduckgo", "searxng"):
                path = root / f"{engine}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({
                            "keyword": "baseline query",
                            "position": index + 1,
                            "title": f"Title {index}",
                            "url": f"https://{engine}.test/{index}",
                            "snippet": f"Evidence {index}",
                        }) + "\n"
                        for index in range(2)
                    ),
                    encoding="utf-8",
                )
                adapters[engine] = FrozenSnapshotSearchAdapter(engine, path)
            prompts = tuple(
                CalibrationPrompt(
                    prompt_id=f"prompt-{index}",
                    prompt=f"Question {index}?",
                    question_sha256=f"{index:064x}",
                    axis_bin=index,
                    keyword="baseline query",
                )
                for index in range(4)
            )

            targets = _build_target_urls(prompts, adapters, seed=17)

        self.assertEqual(len(targets), 8)
        for engine in adapters:
            counts = Counter(
                targets[(prompt.prompt_id, engine)] for prompt in prompts
            )
            self.assertEqual(sorted(counts.values()), [2, 2])

    def test_target_urls_fall_back_to_frozen_lexical_retrieval(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "duckduckgo.jsonl"
            path.write_text(
                "".join(
                    json.dumps({
                        "keyword": "family medical coverage",
                        "position": index + 1,
                        "title": f"Family insurance source {index}",
                        "url": f"https://insurance.test/{index}",
                        "snippet": "Health insurance options for families",
                    }) + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            adapter = FrozenSnapshotSearchAdapter("duckduckgo", path)
            prompt = CalibrationPrompt(
                prompt_id="missing-exact-keyword",
                prompt="Which family health insurance plan should I choose?",
                question_sha256="f" * 64,
                axis_bin=0,
                keyword="best health insurance for families",
            )
            selection_audit = {}

            targets = _build_target_urls(
                (prompt,),
                {"duckduckgo": adapter},
                seed=17,
                selection_audit=selection_audit,
            )

        self.assertIn(
            targets[(prompt.prompt_id, "duckduckgo")],
            {"https://insurance.test/0", "https://insurance.test/1"},
        )
        self.assertEqual(selection_audit, {
            "duckduckgo": {"deterministic_lexical_fallback": 1},
        })

    def test_complete_twelve_cell_flow_writes_auditable_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = {}
            for engine in ("duckduckgo", "searxng"):
                path = root / f"{engine}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({
                            "keyword": "Berlin population",
                            "position": index + 1,
                            "title": f"Berlin source {index}",
                            "url": f"https://{engine}.test/{index}",
                            "snippet": f"Berlin population evidence {index}",
                        })
                        + "\n"
                        for index in range(20)
                    ),
                    encoding="utf-8",
                )
                snapshots[engine] = path
            cross_encoder = root / ("e" * 40)
            cross_encoder.mkdir()
            inputs = SmokeInputs(
                output=root / "output",
                base_url="http://127.0.0.1:8010/v1",
                model_id="test/model",
                model_revision="a" * 40,
                cross_encoder_snapshot=cross_encoder,
                cross_encoder_revision="e" * 40,
                search_snapshots=snapshots,
                seed=11,
                max_tokens=128,
                request_concurrency=4,
            )
            client = _FakeClientContext(delay_seconds=0.01)
            manifest = asyncio.run(run_smoke(
                inputs,
                client_context=client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))

            results = list((inputs.output / "results").glob("*.json"))
            traces = list((inputs.output / "traces").glob("*.json"))
            diagnostics = list((inputs.output / "diagnostics").glob("*.json"))
            diagnostic_record = json.loads(
                diagnostics[0].read_text(encoding="utf-8")
            )
            manifest_path = inputs.output / "run_manifest.json"
            manifest_before_resume = manifest_path.read_bytes()
            resume_client = _FakeClientContext(delay_seconds=0.01)
            resumed = asyncio.run(run_smoke(
                inputs,
                client_context=resume_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            manifest_after_resume = manifest_path.read_bytes()

            trace_only_result = results[0]
            trace_only_record = json.loads(trace_only_result.read_text(encoding="utf-8"))
            trace_only_path = Path(trace_only_record["trace"])
            trace_only_result.unlink()
            retry_client = _FakeClientContext(delay_seconds=0.0)
            retried = asyncio.run(run_smoke(
                inputs,
                client_context=retry_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))
            recovered = list(
                (inputs.output / "failed_traces" / trace_only_record["cell_id"]).glob(
                    "recovered-*.json"
                )
            )

            trace_only_result.unlink()
            second_retry_client = _FakeClientContext(delay_seconds=0.0)
            second_retry = asyncio.run(run_smoke(
                inputs,
                client_context=second_retry_client,
                compactor=ContextCompactor(
                    MemoizingSnippetScorer(LexicalOverlapScorer())
                ),
            ))

            corrupt_result_path = results[2]
            corrupt_result_bytes = corrupt_result_path.read_bytes()
            corrupt_record = json.loads(corrupt_result_bytes)
            corrupt_trace_path = Path(corrupt_record["trace"])
            corrupt_trace_bytes = corrupt_trace_path.read_bytes()
            corrupt_trace = json.loads(corrupt_trace_bytes)
            corrupt_trace["trace_sha256"] = "0" * 64
            corrupt_result_path.unlink()
            corrupt_trace_path.write_text(
                json.dumps(corrupt_trace), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "trace hash mismatch"):
                asyncio.run(run_smoke(
                    inputs,
                    client_context=_FakeClientContext(),
                    compactor=ContextCompactor(
                        MemoizingSnippetScorer(LexicalOverlapScorer())
                    ),
                ))
            corrupt_trace_path.write_bytes(corrupt_trace_bytes)
            corrupt_result_path.write_bytes(corrupt_result_bytes)

            wrong_result_path = results[3]
            wrong_result_bytes = wrong_result_path.read_bytes()
            wrong_record = json.loads(wrong_result_bytes)
            wrong_trace_path = Path(wrong_record["trace"])
            wrong_trace_bytes = wrong_trace_path.read_bytes()
            wrong_trace = json.loads(wrong_trace_bytes)
            wrong_trace["method_id"] = "wrong-method"
            wrong_core = dict(wrong_trace)
            wrong_core.pop("trace_sha256")
            wrong_trace["trace_sha256"] = hashlib.sha256(json.dumps(
                wrong_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            wrong_result_path.unlink()
            wrong_trace_path.write_text(json.dumps(wrong_trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace identity mismatch"):
                asyncio.run(run_smoke(
                    inputs,
                    client_context=_FakeClientContext(),
                    compactor=ContextCompactor(
                        MemoizingSnippetScorer(LexicalOverlapScorer())
                    ),
                ))
            wrong_trace_path.write_bytes(wrong_trace_bytes)
            wrong_result_path.write_bytes(wrong_result_bytes)

            result_only_record = json.loads(results[1].read_text(encoding="utf-8"))
            Path(result_only_record["trace"]).unlink()
            with self.assertRaisesRegex(ValueError, "without immutable trace"):
                asyncio.run(run_smoke(
                    inputs,
                    client_context=_FakeClientContext(),
                    compactor=ContextCompactor(
                        MemoizingSnippetScorer(LexicalOverlapScorer())
                    ),
                ))

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_count"], 12)
        self.assertEqual(manifest["remaining_count"], 0)
        self.assertEqual(len(results), 12)
        self.assertEqual(len(traces), 12)
        self.assertEqual(len(diagnostics), 12)
        self.assertEqual(diagnostic_record["status"], "complete")
        self.assertEqual(len(diagnostic_record["llm_calls"]), 2)
        self.assertEqual(diagnostic_record["llm_calls"][0]["usage"], {
            "total_tokens": 1,
        })
        self.assertEqual(manifest["request_concurrency"], 4)
        self.assertEqual(manifest["peak_active_cells_this_invocation"], 4)
        self.assertEqual(
            manifest["execution_policy"]["max_tokens_by_purpose"],
            {
                "parallel_query_expansion": 128,
                "parallel_final": 2048,
                "reactive_action": 2048,
                "reactive_forced_finish": 2048,
            },
        )
        self.assertEqual(
            manifest["execution_policy"]["ranking_reference_mode"],
            "evidence-id-v1",
        )
        self.assertEqual(
            manifest["execution_policy"]["final_answer_max_characters"],
            1200,
        )
        self.assertEqual(
            manifest["execution_policy"]["final_attempt_repair_mode"],
            "evidence-projection-v1",
        )
        self.assertEqual(
            manifest["execution_policy"]["structured_output_schema_mode"],
            "xgrammar-structural-v1",
        )
        self.assertEqual(client.call_count, 24)
        self.assertEqual(client.peak_active_calls, 4)
        self.assertEqual(manifest["compactor_cache_this_invocation"], {
            "entries": 80,
            "hits": 140,
            "misses": 80,
        })
        self.assertEqual(resumed, manifest)
        self.assertEqual(resume_client.call_count, 0)
        self.assertEqual(manifest_after_resume, manifest_before_resume)
        self.assertEqual(retried["status"], "complete")
        self.assertEqual(retry_client.call_count, 2)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(second_retry["status"], "complete")
        self.assertEqual(second_retry_client.call_count, 2)


def _smoke_inputs(root: Path) -> SmokeInputs:
    snapshots = {}
    for engine in ("duckduckgo", "searxng"):
        path = root / f"{engine}.jsonl"
        path.write_text(
            "".join(
                json.dumps({
                    "keyword": "Berlin population",
                    "position": index + 1,
                    "title": f"Berlin source {index}",
                    "url": f"https://{engine}.test/{index}",
                    "snippet": f"Berlin population evidence {index}",
                }) + "\n"
                for index in range(20)
            ),
            encoding="utf-8",
        )
        snapshots[engine] = path
    cross_encoder = root / ("e" * 40)
    cross_encoder.mkdir()
    return SmokeInputs(
        output=root / "output",
        base_url="http://127.0.0.1:8010/v1",
        model_id="test/model",
        model_revision="a" * 40,
        cross_encoder_snapshot=cross_encoder,
        cross_encoder_revision="e" * 40,
        search_snapshots=snapshots,
        seed=11,
        max_tokens=1024,
        query_max_tokens=256,
        final_max_tokens=2048,
        request_concurrency=4,
    )


def _parallel_method(generator: VllmAgentGenerator) -> ParallelExpansionV1:
    snippet = {
        "url": "https://example.test/berlin",
        "title": "Berlin census",
        "text": "Berlin population evidence",
    }
    search = StaticSearchAdapter(
        "duckduckgo",
        {
            "Berlin people": [snippet],
            "Berlin census": [snippet],
            "Berlin residents": [snippet],
        },
    )
    return ParallelExpansionV1(
        llm=generator,
        search=search,
        compactor=ContextCompactor(MemoizingSnippetScorer(LexicalOverlapScorer())),
        condition_hook=IdentityConditionHook(),
    )


class _FakeClientContext:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        failed_final_passes: int = 0,
        truncate_final_at: int | None = None,
    ):
        self.delay_seconds = delay_seconds
        self.failed_final_passes = failed_final_passes
        self.truncate_final_at = truncate_final_at
        self.active_calls = 0
        self.peak_active_calls = 0
        self.call_count = 0
        self.calls: list[tuple[str, int]] = []
        self.failed_final_prompt: str | None = None
        self.failed_final_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def complete(self, *, prompt, schema_name, max_tokens, **kwargs):
        self.call_count += 1
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delay_seconds)
            if "parallel_query_expansion" in schema_name:
                purpose = "parallel_query_expansion"
                value = {"queries": ["Berlin people", "Berlin census", "Berlin residents"]}
            elif "parallel_final" in schema_name:
                purpose = "parallel_final"
                if (
                    self.truncate_final_at is not None
                    and max_tokens <= self.truncate_final_at
                ):
                    self.calls.append((purpose, max_tokens))
                    return '{"ranking": [', {"completion_tokens": max_tokens}
                if self.failed_final_prompt is None and self.failed_final_passes:
                    self.failed_final_prompt = prompt
                if (
                    prompt == self.failed_final_prompt
                    and self.failed_final_calls < self.failed_final_passes * 3
                ):
                    self.failed_final_calls += 1
                    self.calls.append((purpose, max_tokens))
                    return '{"ranking": [', {"completion_tokens": max_tokens}
                value = {
                    "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
                    "answer": "Parallel answer",
                }
            elif "reactive_action" in schema_name and "OBSERVATIONS:\n[]" in prompt:
                purpose = "reactive_action"
                value = {"action": "search", "query": "Berlin population"}
            elif "reactive_action" in schema_name:
                purpose = "reactive_action"
                value = {
                    "action": "finish",
                    "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
                    "answer": "Reactive answer",
                }
            elif "reactive_forced_finish" in schema_name:
                purpose = "reactive_forced_finish"
                value = {}
            else:
                raise AssertionError(schema_name)
            self.calls.append((purpose, max_tokens))
            return json.dumps(value), {"total_tokens": 1}
        finally:
            self.active_calls -= 1


if __name__ == "__main__":
    unittest.main()
