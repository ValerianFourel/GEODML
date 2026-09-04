"""CPU integration test for the complete ACL ARR command-line pipeline."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from analysis.tests.test_acl_arr_document_experiment import (
    _axis_rows,
    _models,
    _prompts,
)


class AclArrPipelineCliTests(unittest.TestCase):
    def test_complete_fake_pipeline_is_resumable_and_not_scientific(self) -> None:
        repository = Path(__file__).resolve().parents[2]

        def write_jsonl(path: Path, rows) -> None:
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        def run(*arguments: str) -> subprocess.CompletedProcess[str]:
            completed = subprocess.run(
                [sys.executable, *arguments],
                cwd=repository,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return completed

        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            axis = root / "axis.jsonl"
            serp = root / "serp.jsonl"
            pages = root / "pages.jsonl"
            models = root / "models.json"
            write_jsonl(prompts, _prompts())
            write_jsonl(axis, _axis_rows())
            serp_rows = []
            page_rows = []
            for keyword in ("alpha software", "beta platform"):
                prefix = keyword.split()[0]
                for position in range(1, 4):
                    url = f"https://{prefix}{position}.example/page"
                    serp_rows.append(
                        {
                            "keyword": keyword,
                            "position": position,
                            "title": f"{keyword} result {position}",
                            "url": url,
                            "snippet": f"Snippet {position}",
                            "search_engine": "searxng",
                        }
                    )
                    page_rows.append(
                        {"url": url, "extracted_text": f"Frozen page {position}."}
                    )
            write_jsonl(serp, serp_rows)
            write_jsonl(pages, page_rows)
            models.write_text(
                json.dumps({"models": [asdict(_models()[0])]}), encoding="utf-8"
            )

            document_root = root / "documents"
            run(
                "analysis/scripts/prepare_acl_arr_document_sets.py",
                "--serp",
                str(serp),
                "--page-text",
                str(pages),
                "--minimum-documents",
                "3",
                "--maximum-documents",
                "3",
                "--output-dir",
                str(document_root),
            )
            plan_root = root / "plan"
            run(
                "analysis/scripts/prepare_acl_arr_experiment.py",
                "--prompts-jsonl",
                str(prompts),
                "--axis-map-jsonl",
                str(axis),
                "--document-sets-jsonl",
                str(document_root / "frozen_document_sets.jsonl"),
                "--models-json",
                str(models),
                "--top-n",
                "2",
                "--expected-prompt-count",
                "2",
                "--expected-model-count",
                "1",
                "--output-dir",
                str(plan_root),
            )
            configuration_id = _models()[0].configuration_id
            rerank_root = root / "rerank"
            answer_root = root / "answer"
            for pipeline, output in (
                ("rerank", rerank_root),
                ("answer", answer_root),
            ):
                command = (
                    "analysis/scripts/run_acl_arr_vllm.py",
                    "primary",
                    "--tasks",
                    str(plan_root / "tasks" / configuration_id / f"{pipeline}.jsonl"),
                    "--plan-manifest",
                    str(plan_root / "run_manifest.json"),
                    "--output-dir",
                    str(output),
                    "--fake",
                )
                run(*command)
                run(*command, "--resume")
                self.assertEqual(
                    len((output / "outcomes.jsonl").read_text().splitlines()), 6
                )

            judge_plan = root / "judge-plan"
            run(
                "analysis/scripts/prepare_acl_arr_judge_tasks.py",
                "--answer-outcomes",
                str(answer_root / "outcomes.jsonl"),
                "--plan-manifest",
                str(plan_root / "run_manifest.json"),
                "--judge-model-id",
                "judge/model",
                "--judge-model-revision",
                "3" * 40,
                "--allow-fake",
                "--output-dir",
                str(judge_plan),
            )
            judge_results = root / "judge-results"
            run(
                "analysis/scripts/run_acl_arr_vllm.py",
                "judge",
                "--tasks",
                str(judge_plan / "judge_tasks.jsonl"),
                "--judge-manifest",
                str(judge_plan / "judge_manifest.json"),
                "--output-dir",
                str(judge_results),
                "--fake",
            )
            analysis_root = root / "analysis"
            run(
                "analysis/scripts/analyze_acl_arr_experiment.py",
                "--plan-manifest",
                str(plan_root / "run_manifest.json"),
                "--rerank-outcomes",
                str(rerank_root / "outcomes.jsonl"),
                "--answer-outcomes",
                str(answer_root / "outcomes.jsonl"),
                "--judge-outcomes",
                str(judge_results / "outcomes.jsonl"),
                "--private-judge-mapping",
                str(judge_plan / "private_judge_mapping.jsonl"),
                "--allow-fake",
                "--output-dir",
                str(analysis_root),
            )
            summary = json.loads(
                (analysis_root / "analysis_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["result"], "PASS")
            self.assertFalse(summary["scientific_result"])
            self.assertEqual(summary["paired_prompt_model_count"], 2)


if __name__ == "__main__":
    unittest.main()
