import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.merge_readiness_validation_caches import (
    _stable_hash,
    merge_validation_caches,
)


class MergeReadinessValidationCachesTest(unittest.TestCase):
    def _record(
        self,
        root: Path,
        *,
        candidate_id: str,
        accepted: bool,
        terminal: bool = False,
    ) -> Path:
        identity = {
            "version": "readiness-question-population-v2",
            "judge_id": "judge",
            "judge_model": "model/judge",
            "candidate_id": candidate_id,
            "question_sha256": f"question-{candidate_id}",
        }
        payload = {
            "identity": identity,
            "review": {
                "candidate_id": candidate_id,
                "accepted": accepted,
            },
            "failures": [],
        }
        if terminal:
            payload["terminal_parse_failure"] = True
        path = root / f"{_stable_hash(identity)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_unions_sources_and_prefers_nonterminal_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            destination = root / "destination"
            self._record(first, candidate_id="one", accepted=False, terminal=True)
            self._record(second, candidate_id="one", accepted=True)
            self._record(second, candidate_id="two", accepted=True)

            report = merge_validation_caches(
                (first, second),
                destination,
                judge_id="judge",
                judge_model="model/judge",
            )

            payloads = [
                json.loads(path.read_text())
                for path in destination.glob("*.json")
            ]
            reviews = {
                row["identity"]["candidate_id"]: row["review"]["accepted"]
                for row in payloads
            }
            self.assertEqual(reviews, {"one": True, "two": True})
            self.assertEqual(report["destination_file_count"], 2)
            self.assertEqual(report["replaced_terminal_count"], 1)

    def test_rejects_conflicting_nonterminal_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            destination = root / "destination"
            self._record(first, candidate_id="one", accepted=False)
            self._record(second, candidate_id="one", accepted=True)

            with self.assertRaisesRegex(ValueError, "conflicting validator reviews"):
                merge_validation_caches((first, second), destination)


if __name__ == "__main__":
    unittest.main()
