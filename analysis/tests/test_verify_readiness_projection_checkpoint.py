"""Contracts for immutable, relocatable readiness projection checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.verify_readiness_projection_checkpoint import (
    verify_projection_checkpoint,
)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


class VerifyReadinessProjectionCheckpointTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "attempt" / "merged" / "candidates.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text('{"candidate_id":"candidate-1"}\n', encoding="utf-8")
        manifest = root / "projection_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "candidate_count": 1,
                    "candidate_files": [_identity(source)],
                    "embedding": {"attention_implementation": "eager"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        destination = root / "final" / "merged" / "candidates.jsonl"
        destination.parent.mkdir(parents=True)
        source.replace(destination)
        listing = root / "candidate-files.txt"
        listing.write_text(str(destination) + "\n", encoding="utf-8")
        return manifest, listing, destination

    def test_exact_candidate_content_survives_atomic_checkpoint_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, listing, _ = self._fixture(Path(temporary))
            verify_projection_checkpoint(
                manifest,
                expected_count=1,
                candidate_file_list=listing,
                expected_attention="eager",
            )

    def test_changed_candidate_content_is_rejected_after_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, listing, destination = self._fixture(Path(temporary))
            destination.write_text(
                '{"candidate_id":"different-candidate"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "content identity differs"):
                verify_projection_checkpoint(
                    manifest,
                    expected_count=1,
                    candidate_file_list=listing,
                    expected_attention="eager",
                )

    def test_candidate_count_and_attention_remain_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, listing, _ = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "candidate count differs"):
                verify_projection_checkpoint(
                    manifest,
                    expected_count=2,
                    candidate_file_list=listing,
                    expected_attention="eager",
                )
            with self.assertRaisesRegex(ValueError, "attention implementation differs"):
                verify_projection_checkpoint(
                    manifest,
                    expected_count=1,
                    candidate_file_list=listing,
                    expected_attention="flash_attention_2",
                )


if __name__ == "__main__":
    unittest.main()
