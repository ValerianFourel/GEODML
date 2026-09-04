"""Paired-analysis contracts for the ACL ARR document experiment."""

from __future__ import annotations

import json
import unittest

from analysis.interpretability.pipeline.acl_arr_document_analysis import (
    analyze_acl_arr_outcomes,
)
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    build_acl_arr_experiment_plan,
    build_blinded_judge_tasks,
    iter_experiment_tasks,
)
from analysis.tests.test_acl_arr_document_experiment import (
    _axis_rows,
    _document_sets,
    _models,
    _prompts,
)


class AclArrAnalysisTests(unittest.TestCase):
    def _fixture(self):
        plan = build_acl_arr_experiment_plan(
            _prompts()[:1],
            _axis_rows()[:1],
            _document_sets()[:1],
            models=(_models()[0],),
            top_n=2,
        )
        assignment = plan.assignments[0]
        rerank_rows = []
        answer_rows = []
        for task in iter_experiment_tasks(plan):
            input_ids = assignment.document_ids(task.condition)
            base = {
                "task_id": task.task_id,
                "pipeline": task.pipeline,
                "condition": task.condition,
                "prompt_id": task.prompt_id,
                "assignment_id": assignment.assignment_id,
                "candidate_set_id": assignment.candidate_set_id,
                "input_document_ids": list(input_ids),
                "ablation_target_id": assignment.ablation_target_id,
                "model_configuration_id": _models()[0].configuration_id,
                "model_id": _models()[0].model_id,
                "fake_backend": True,
            }
            if task.pipeline == "rerank":
                rerank_rows.append(
                    {
                        **base,
                        "parsed_output": {
                            "ranked_document_ids": list(
                                input_ids[: task.output_document_count]
                            )
                        },
                    }
                )
            else:
                cited = input_ids[0]
                answer_rows.append(
                    {
                        **base,
                        "parsed_output": {
                            "answer": f"Fixture answer [{cited}].",
                            "cited_document_ids": [cited],
                        },
                    }
                )
        judge_plan = build_blinded_judge_tasks(
            answer_rows,
            plan=plan,
            judge_model_id="judge/model",
            judge_model_revision="3" * 40,
            master_seed=8,
            allow_fake=True,
        )
        judge_rows = [
            {
                "judge_task_id": task.judge_task_id,
                "blind_case_id": task.blind_case_id,
                "fake_backend": True,
                "parsed_output": {
                    "answer_quality": 3,
                    "evidence_coverage": 3,
                    "citation_correctness": 3,
                    "unsupported_claim_count": 0,
                    "realized_document_ranking": [
                        {"document_id": task.judge_document_ids[0], "use_score": 3}
                    ],
                },
            }
            for task in judge_plan.tasks
        ]
        mappings = [
            {
                field: getattr(item, field)
                for field in item.__dataclass_fields__
            }
            for item in judge_plan.mappings
        ]
        return plan, rerank_rows, answer_rows, judge_rows, mappings

    def test_requires_complete_paired_cells_and_computes_differences(self) -> None:
        plan, rerank, answers, judgments, mappings = self._fixture()
        analysis = analyze_acl_arr_outcomes(
            rerank,
            answers,
            judgments,
            mappings,
            plan=plan,
            allow_fake=True,
        )
        self.assertEqual(analysis.summary["result"], "PASS")
        self.assertEqual(analysis.summary["paired_prompt_model_count"], 1)
        self.assertEqual(len(analysis.paired_rows), 1)
        row = analysis.paired_rows[0]
        self.assertIn("rerank_shuffle_top_k_jaccard", row)
        self.assertIn("answer_ablation_citation_jaccard", row)
        self.assertIn("judge_shuffle_realized_use_jaccard", row)

        with self.assertRaisesRegex(ValueError, "judge coverage"):
            analyze_acl_arr_outcomes(
                rerank,
                answers,
                judgments[:-1],
                mappings,
                plan=plan,
                allow_fake=True,
            )


if __name__ == "__main__":
    unittest.main()
