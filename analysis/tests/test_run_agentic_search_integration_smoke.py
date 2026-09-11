"""Tests for the agentic-search integration smoke runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from analysis.interpretability.pipeline.agentic_search import (
    ContextCompactor,
    ExperimentalCondition,
    LexicalOverlapScorer,
    MemoizingSnippetScorer,
    Snippet,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    FrozenSnapshotSearchAdapter,
    SmokeInputs,
    SmokeConditionHook,
    run_smoke,
)


class FrozenSnapshotSearchAdapterTests(unittest.TestCase):
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

    def test_cluster_launcher_forces_offline_model_resolution(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh"
        ).read_text(encoding="utf-8")
        server_start = launcher.index("python3 analysis/scripts/search_vllm_stage.py run")

        self.assertIn("export HF_HUB_OFFLINE=1", launcher[:server_start])
        self.assertIn("export TRANSFORMERS_OFFLINE=1", launcher[:server_start])

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


class _FakeClientContext:
    def __init__(self, *, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.active_calls = 0
        self.peak_active_calls = 0
        self.call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def complete(self, *, prompt, schema_name, **kwargs):
        self.call_count += 1
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delay_seconds)
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
        finally:
            self.active_calls -= 1


if __name__ == "__main__":
    unittest.main()
