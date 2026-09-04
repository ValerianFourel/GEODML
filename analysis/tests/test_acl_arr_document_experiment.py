"""Contracts for the ACL ARR document-order and ablation experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    ModelConfiguration,
    build_acl_arr_experiment_plan,
    build_blinded_judge_tasks,
    iter_experiment_tasks,
    render_judge_prompt,
    validate_answer_output,
    validate_judge_output,
    validate_rerank_output,
    write_acl_arr_experiment_plan,
    write_blinded_judge_plan,
)
from analysis.scripts.run_acl_arr_vllm import VllmChatClient


def _prompts() -> list[dict[str, object]]:
    information = "What factors distinguish the available alpha software options?"
    action = "Configure the beta platform now and verify the deployment."
    return [
        {
            "candidate_id": "prompt:alpha:information",
            "keyword_id": "keyword:alpha",
            "keyword": "alpha software",
            "target_id": "target:alpha:0",
            "target_index": 0,
            "question": information,
            "target_normalized_axis_1": 0.1,
            "question_sha256": hashlib.sha256(information.encode()).hexdigest(),
        },
        {
            "candidate_id": "prompt:beta:action",
            "keyword_id": "keyword:beta",
            "keyword": "beta platform",
            "target_id": "target:beta:0",
            "target_index": 0,
            "question": action,
            "target_normalized_axis_1": 0.9,
            "question_sha256": hashlib.sha256(action.encode()).hexdigest(),
        },
    ]


def _axis_rows() -> list[dict[str, object]]:
    prompt_hashes = {
        str(row["candidate_id"]): str(row["question_sha256"])
        for row in _prompts()
    }
    return [
        {
            "candidate_id": "prompt:alpha:information",
            "text_sha256": prompt_hashes["prompt:alpha:information"],
            "consensus_axis_1_z": -1.2,
            "axis_1_rank": 0,
            "axis_1_percentile_0_1": 0.0,
            "reference_axis_1_z": -1.22,
            "candidate_aligned_axis_1_z": -1.18,
        },
        {
            "candidate_id": "prompt:beta:action",
            "text_sha256": prompt_hashes["prompt:beta:action"],
            "consensus_axis_1_z": 0.8,
            "axis_1_rank": 1,
            "axis_1_percentile_0_1": 1.0,
            "reference_axis_1_z": 0.81,
            "candidate_aligned_axis_1_z": 0.79,
        },
    ]


def _document_sets() -> list[dict[str, object]]:
    output = []
    for keyword in ("alpha software", "beta platform"):
        prefix = keyword.split()[0]
        output.append(
            {
                "candidate_set_id": f"set:{prefix}",
                "keyword": keyword,
                "search_query": keyword,
                "search_engine": "searxng",
                "search_snapshot_sha256": "c" * 64,
                "candidates": [
                    {
                        "candidate_id": f"C{index:03d}",
                        "source_position": index,
                        "title": f"{keyword} result {index}",
                        "url": f"https://{prefix}{index}.example/page",
                        "snippet": f"Frozen evidence {index} for {keyword}.",
                    }
                    for index in range(1, 4)
                ],
            }
        )
    return output


def _models() -> tuple[ModelConfiguration, ...]:
    return (
        ModelConfiguration(
            model_id="model/dense",
            model_revision="1" * 40,
            architecture="dense",
            total_parameters_b=27.0,
            active_parameters_b=27.0,
            precision="bfloat16",
            rerank_max_tokens=128,
            answer_max_tokens=512,
        ),
        ModelConfiguration(
            model_id="model/moe",
            model_revision="2" * 40,
            architecture="moe",
            total_parameters_b=109.0,
            active_parameters_b=17.0,
            precision="bfloat16",
            rerank_max_tokens=128,
            answer_max_tokens=512,
        ),
    )


class AclArrPlanTests(unittest.TestCase):
    def test_unresolved_model_or_mutable_revision_cannot_enter_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "resolve the model_id"):
            ModelConfiguration(
                model_id="UNRESOLVED_QWEN_DENSE_72B",
                model_revision="1" * 40,
                architecture="dense",
                total_parameters_b=72.0,
                active_parameters_b=72.0,
                precision="bfloat16",
            )
        with self.assertRaisesRegex(ValueError, "40-character SHA"):
            ModelConfiguration(
                model_id="model/dense",
                model_revision="main",
                architecture="dense",
                total_parameters_b=27.0,
                active_parameters_b=27.0,
                precision="bfloat16",
            )

    def test_builds_three_paired_conditions_for_both_pipelines_and_models(self) -> None:
        plan = build_acl_arr_experiment_plan(
            _prompts(),
            _axis_rows(),
            _document_sets(),
            models=_models(),
            top_n=2,
            master_seed=20260904,
            prompt_source_sha256="d" * 64,
            axis_source_sha256="e" * 64,
            document_source_sha256="f" * 64,
            source_git_commit="0" * 40,
        )

        self.assertEqual(plan.summary["prompt_count"], 2)
        self.assertEqual(plan.summary["condition_assignment_count"], 2)
        self.assertEqual(plan.summary["model_count"], 2)
        self.assertEqual(plan.summary["tasks_per_pipeline"], 12)
        self.assertEqual(plan.summary["primary_task_count"], 24)

        for assignment in plan.assignments:
            natural = assignment.natural_document_ids
            ablated = assignment.ablated_document_ids
            shuffled = assignment.shuffled_document_ids
            self.assertEqual(len(natural), 3)
            self.assertEqual(len(ablated), 2)
            self.assertEqual(set(natural) - set(ablated), {assignment.ablation_target_id})
            self.assertEqual(set(shuffled), set(natural))
            self.assertNotEqual(shuffled, natural)
            self.assertTrue(
                all(left != right for left, right in zip(natural, shuffled))
            )

        tasks = list(iter_experiment_tasks(plan))
        self.assertEqual(len(tasks), 24)
        self.assertEqual({task.pipeline for task in tasks}, {"rerank", "answer"})
        self.assertEqual(
            {task.condition for task in tasks}, {"natural", "ablated", "shuffled"}
        )
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))

    def test_plan_is_deterministic_and_changes_with_seed(self) -> None:
        common = dict(
            models=_models(),
            top_n=2,
            prompt_source_sha256="d" * 64,
            axis_source_sha256="e" * 64,
            document_source_sha256="f" * 64,
            source_git_commit="0" * 40,
        )
        first = build_acl_arr_experiment_plan(
            _prompts(), _axis_rows(), _document_sets(), master_seed=11, **common
        )
        repeated = build_acl_arr_experiment_plan(
            _prompts(), _axis_rows(), _document_sets(), master_seed=11, **common
        )
        changed = build_acl_arr_experiment_plan(
            _prompts(), _axis_rows(), _document_sets(), master_seed=12, **common
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.plan_id, changed.plan_id)

    def test_ablation_targets_are_balanced_within_document_count(self) -> None:
        base = _prompts()[0]
        prompts = []
        axes = []
        for index in range(12):
            question = f"What should team {index} know about alpha software?"
            prompt_id = f"prompt:alpha:{index:02d}"
            question_hash = hashlib.sha256(question.encode()).hexdigest()
            prompts.append(
                {
                    **base,
                    "candidate_id": prompt_id,
                    "target_id": f"target:alpha:{index:02d}",
                    "target_index": index,
                    "question": question,
                    "question_sha256": question_hash,
                }
            )
            axes.append(
                {
                    **_axis_rows()[0],
                    "candidate_id": prompt_id,
                    "text_sha256": question_hash,
                    "axis_1_rank": index,
                    "axis_1_percentile_0_1": index / 11,
                }
            )
        plan = build_acl_arr_experiment_plan(
            prompts,
            axes,
            _document_sets()[:1],
            models=(_models()[0],),
            top_n=2,
        )
        target_counts = Counter(
            assignment.ablation_target_id for assignment in plan.assignments
        )
        self.assertEqual(set(target_counts.values()), {4})

    def test_rejects_axis_identity_or_document_coverage_mismatch(self) -> None:
        bad_axis = _axis_rows()
        bad_axis[0] = {**bad_axis[0], "text_sha256": "9" * 64}
        with self.assertRaisesRegex(ValueError, "question hash"):
            build_acl_arr_experiment_plan(
                _prompts(),
                bad_axis,
                _document_sets(),
                models=_models(),
                top_n=2,
            )

        with self.assertRaisesRegex(ValueError, "missing frozen document set"):
            build_acl_arr_experiment_plan(
                _prompts(),
                _axis_rows(),
                _document_sets()[:1],
                models=_models(),
                top_n=2,
            )

    def test_strict_output_validators_fail_closed(self) -> None:
        rerank = validate_rerank_output(
            '{"ranked_document_ids":["C002","C001"]}',
            allowed_document_ids=("C001", "C002", "C003"),
            output_count=2,
        )
        self.assertEqual(rerank["ranked_document_ids"], ["C002", "C001"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rerank_output(
                '{"ranked_document_ids":["C001","C001"]}',
                allowed_document_ids=("C001", "C002", "C003"),
                output_count=2,
            )

        answer = validate_answer_output(
            '{"answer":"Use the first source [C001].",'
            '"cited_document_ids":["C001"]}',
            allowed_document_ids=("C001", "C002"),
        )
        self.assertEqual(answer["cited_document_ids"], ["C001"])
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_answer_output(
                '{"answer":"Use it [C001].","cited_document_ids":["C002"]}',
                allowed_document_ids=("C001", "C002"),
            )

        judgment = validate_judge_output(
            json.dumps(
                {
                    "answer_quality": 4,
                    "evidence_coverage": 3,
                    "citation_correctness": 5,
                    "unsupported_claim_count": 0,
                    "realized_document_ranking": [
                        {"document_id": "C001", "use_score": 4}
                    ],
                }
            ),
            allowed_document_ids=("C001", "C002"),
        )
        self.assertEqual(judgment["answer_quality"], 4)

    def test_writer_and_fake_runner_cover_rerank_answer_and_judge(self) -> None:
        plan = build_acl_arr_experiment_plan(
            _prompts(),
            _axis_rows(),
            _document_sets(),
            models=(_models()[0],),
            top_n=2,
            prompt_source_sha256="d" * 64,
            axis_source_sha256="e" * 64,
            document_source_sha256="f" * 64,
            source_git_commit="0" * 40,
        )
        repository = Path(__file__).resolve().parents[2]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = write_acl_arr_experiment_plan(root / "plan", plan=plan)
            rerank_tasks = artifacts.task_files[(_models()[0].configuration_id, "rerank")]
            answer_tasks = artifacts.task_files[(_models()[0].configuration_id, "answer")]
            self.assertEqual(len(rerank_tasks.read_text().splitlines()), 6)
            self.assertEqual(len(answer_tasks.read_text().splitlines()), 6)

            for pipeline, task_path in (
                ("rerank", rerank_tasks),
                ("answer", answer_tasks),
            ):
                command = [
                    sys.executable,
                    "analysis/scripts/run_acl_arr_vllm.py",
                    "primary",
                    "--tasks",
                    str(task_path),
                    "--plan-manifest",
                    str(artifacts.manifest_path),
                    "--output-dir",
                    str(root / pipeline),
                    "--fake",
                ]
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            answer_rows = [
                json.loads(line)
                for line in (root / "answer" / "outcomes.jsonl").read_text().splitlines()
            ]
            judge_plan = build_blinded_judge_tasks(
                answer_rows,
                plan=plan,
                judge_model_id="judge/model",
                judge_model_revision="3" * 40,
                master_seed=99,
                allow_fake=True,
            )
            self.assertEqual(len(judge_plan.tasks), 6)
            document_set = next(
                item
                for item in plan.document_sets
                if item.candidate_set_id == judge_plan.tasks[0].candidate_set_id
            )
            rendered_judge = render_judge_prompt(judge_plan.tasks[0], document_set)
            self.assertNotIn("generator_model_id", rendered_judge)
            self.assertNotIn("condition", rendered_judge.casefold())
            judge_artifacts = write_blinded_judge_plan(
                root / "judge-plan", judge_plan=judge_plan
            )
            command = [
                sys.executable,
                "analysis/scripts/run_acl_arr_vllm.py",
                "judge",
                "--tasks",
                str(judge_artifacts.tasks_path),
                "--judge-manifest",
                str(judge_artifacts.manifest_path),
                "--output-dir",
                str(root / "judge-results"),
                "--fake",
            ]
            completed = subprocess.run(
                command,
                cwd=repository,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                len((root / "judge-results/outcomes.jsonl").read_text().splitlines()),
                6,
            )


class VllmClientProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_checks_server_model_and_sends_structured_chat_request(self) -> None:
        class Response:
            def __init__(self, status, payload):
                self.status = status
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def text(self):
                return json.dumps(self.payload)

        class Session:
            def __init__(self):
                self.posted = None

            def get(self, url):
                return Response(200, {"data": [{"id": "served/model"}]})

            def post(self, url, json):
                self.posted = (url, json)
                return Response(
                    200,
                    {
                        "choices": [
                            {"message": {"content": '{"value":"ok"}'}}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                    },
                )

        client = VllmChatClient(
            base_url="http://localhost:8000/v1",
            api_key=None,
            server_model_name="served/model",
            timeout_seconds=30,
            maximum_attempts=2,
        )
        session = Session()
        client.session = session
        await client.verify_server_identity()
        content, usage = await client.complete(
            prompt="Return JSON.",
            schema_name="fixture",
            schema={"type": "object"},
            temperature=0.0,
            max_tokens=20,
            seed=7,
        )
        self.assertEqual(content, '{"value":"ok"}')
        self.assertEqual(usage["completion_tokens"], 4)
        self.assertEqual(
            session.posted[0], "http://localhost:8000/v1/chat/completions"
        )
        self.assertEqual(session.posted[1]["model"], "served/model")
        self.assertEqual(
            session.posted[1]["response_format"]["type"], "json_schema"
        )


if __name__ == "__main__":
    unittest.main()
