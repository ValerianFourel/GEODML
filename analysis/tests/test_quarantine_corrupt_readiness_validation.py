from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.quarantine_corrupt_readiness_validation import (
    quarantine_corrupt_validation,
)


class CorruptReadinessValidationQuarantineTests(unittest.TestCase):
    def test_quarantines_exact_zero_record_corruption_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "validation-shard-2.jsonl"
            quarantine = root / "validation-shard-2.jsonl.corrupt-job-1481430"
            manifest = root / "quarantine.json"
            source.write_bytes(b"\0" * 4096)

            result = quarantine_corrupt_validation(
                source,
                quarantine,
                manifest,
                source_job_id="1481430",
                recovery_job_id="recovery-one",
            )

            self.assertEqual(result["current_status"], "quarantined")
            self.assertFalse(source.exists())
            self.assertEqual(quarantine.read_bytes(), b"\0" * 4096)
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["newline_count"], 0)
            self.assertEqual(recorded["parsed_row_count"], 0)
            self.assertEqual(recorded["quarantine"]["sha256"], result["quarantine"]["sha256"])

            repeated = quarantine_corrupt_validation(
                source,
                quarantine,
                manifest,
                source_job_id="1481430",
                recovery_job_id="recovery-two",
            )
            self.assertEqual(repeated["current_status"], "already-quarantined")

    def test_existing_quarantine_accepts_a_valid_rebuilt_source(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "validation-shard-2.jsonl"
            quarantine = root / "validation-shard-2.jsonl.corrupt-job-1481430"
            manifest = root / "quarantine.json"
            source.write_bytes(b"\0" * 128)
            quarantine_corrupt_validation(
                source,
                quarantine,
                manifest,
                source_job_id="1481430",
                recovery_job_id="recovery-one",
            )
            source.write_text('{"candidate_id":"candidate:one"}\n', encoding="utf-8")

            repeated = quarantine_corrupt_validation(
                source,
                quarantine,
                manifest,
                source_job_id="1481430",
                recovery_job_id="recovery-two",
            )

            self.assertEqual(repeated["current_status"], "quarantined-source-rebuilt")

    def test_refuses_to_quarantine_valid_or_completed_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "validation-shard-2.jsonl"
            quarantine = root / "quarantine.jsonl"
            manifest = root / "quarantine-manifest.json"
            source.write_text('{"candidate_id":"candidate:one"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                quarantine_corrupt_validation(
                    source,
                    quarantine,
                    manifest,
                    source_job_id="1481430",
                    recovery_job_id="recovery-one",
                )

            source.with_suffix(".jsonl.manifest.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed validation shard"):
                quarantine_corrupt_validation(
                    source,
                    quarantine,
                    manifest,
                    source_job_id="1481430",
                    recovery_job_id="recovery-one",
                )


if __name__ == "__main__":
    unittest.main()
