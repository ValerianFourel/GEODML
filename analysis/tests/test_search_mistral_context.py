from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from analysis.scripts.check_search_mistral_context import (
    check_context_budget,
    resolve_snapshot,
)


class SearchMistralContextTests(unittest.TestCase):
    def test_reports_exact_worst_request_and_rejects_overflow(self):
        items = [
            {"base": {"task_id": "a"}, "prompt": "short", "max_tokens": 8},
            {"base": {"task_id": "b"}, "prompt": "longer", "max_tokens": 5},
        ]
        report = check_context_budget(
            items,
            token_counter=lambda prompt: {"short": 4, "longer": 17}[prompt],
            max_model_len=21,
        )
        self.assertEqual(report["status"], "EXCEEDS_MODEL_CONTEXT")
        self.assertEqual(report["max_prompt_tokens"], 17)
        self.assertEqual(report["max_required_tokens"], 22)
        self.assertEqual(report["worst_task_id"], "b")
        self.assertEqual(report["overflow_tokens"], 1)

    def test_passes_only_when_every_request_fits(self):
        items = [
            {"base": {"task_id": "a"}, "prompt": "x", "max_tokens": 8},
            {"base": {"task_id": "b"}, "prompt": "yy", "max_tokens": 5},
        ]
        report = check_context_budget(
            items,
            token_counter=len,
            max_model_len=10,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["remaining_tokens"], 1)

    def test_resolves_one_exact_pinned_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            locks = root / "model-snapshots.json"
            locks.write_text(json.dumps({"models": [{
                "model_id": "m",
                "revision": "r",
                "snapshot": str(snapshot),
            }]}))
            self.assertEqual(resolve_snapshot(locks, "m", "r"), snapshot.resolve())


if __name__ == "__main__":
    unittest.main()
