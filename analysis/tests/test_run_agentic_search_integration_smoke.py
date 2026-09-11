"""Tests for the agentic-search integration smoke runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from analysis.interpretability.pipeline.agentic_search import (
    ContextCompactor,
    ExperimentalCondition,
    LexicalOverlapScorer,
    Snippet,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    FrozenSnapshotSearchAdapter,
    SmokeInputs,
    SmokeConditionHook,
    run_smoke,
)


class FrozenSnapshotSearchAdapterTests(unittest.TestCase):
    def test_cluster_launcher_forces_offline_model_resolution(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])

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
            )
            manifest = asyncio.run(run_smoke(
                inputs,
                client_context=_FakeClientContext(),
                compactor=ContextCompactor(LexicalOverlapScorer()),
            ))

            results = list((inputs.output / "results").glob("*.json"))
            traces = list((inputs.output / "traces").glob("*.json"))

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_count"], 12)
        self.assertEqual(manifest["remaining_count"], 0)
        self.assertEqual(len(results), 12)
        self.assertEqual(len(traces), 12)


class _FakeClientContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def complete(self, *, prompt, schema_name, **kwargs):
        if "parallel_query_expansion" in schema_name:
            value = {"queries": ["Berlin people", "Berlin census", "Berlin residents"]}
        elif "parallel_final" in schema_name:
            value = {
                "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
                "answer": "Parallel answer",
            }
        elif "reactive_action" in schema_name and "OBSERVATIONS:\n[]" in prompt:
            value = {"action": "search", "query": "Berlin population"}
        elif "reactive_action" in schema_name:
            value = {
                "action": "finish",
                "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
                "answer": "Reactive answer",
            }
        else:
            raise AssertionError(schema_name)
        return json.dumps(value), {"total_tokens": 1}


if __name__ == "__main__":
    unittest.main()
