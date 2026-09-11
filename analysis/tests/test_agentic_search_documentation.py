from pathlib import Path
import unittest

from analysis.interpretability.pipeline import agentic_search


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPOSITORY_ROOT / "analysis/docs/agentic_search_retrieval_protocol.md"


class AgenticSearchDocumentationTests(unittest.TestCase):
    def test_protocol_records_implemented_bounds_and_workload(self):
        text = PROTOCOL.read_text(encoding="utf-8")

        expected_contracts = (
            agentic_search.DEFAULT_CROSS_ENCODER,
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "Qwen/Qwen3.8-27B",
            f"At most {agentic_search.SEARCH_RESULT_LIMIT} snippets",
            f"Exactly {agentic_search.PARALLEL_QUERY_COUNT} distinct queries",
            f"Top {agentic_search.PARALLEL_TOP_K}",
            f"At most {agentic_search.REACTIVE_MAX_ITERATIONS}",
            f"Top {agentic_search.REACTIVE_TOP_K}",
            f"At most {agentic_search.SCHEMA_RETRIES} retries",
            "1,872,648 cells",
            "4,369,512",
            "13,108,536",
            "6,242,160",
            "5,617,944 artifacts",
            "3,745,296 search calls",
            "separate, tested membership and presentation hooks",
        )
        for value in expected_contracts:
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_current_docs_link_to_the_protocol(self):
        for relative_path in (
            "README.md",
            "analysis/docs/search_experience_design.md",
            "analysis/docs/search_experience_pilot.md",
            "analysis/docs/search_experience_worklog.md",
            "analysis/docs/search_experience_judge_readiness.md",
        ):
            with self.subTest(path=relative_path):
                text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("agentic_search_retrieval_protocol.md", text)


if __name__ == "__main__":
    unittest.main()
