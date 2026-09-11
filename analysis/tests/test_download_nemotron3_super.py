"""Tests for the pinned Nemotron 3 Super download verifier."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.download_nemotron3_super import (
    EXPECTED_ARCHITECTURE,
    MINIMUM_WEIGHT_BYTES,
    MODEL_REVISION,
    snapshot_path,
    verify_snapshot,
)


class NemotronDownloadTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        snapshot = snapshot_path(root)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"architectures": [EXPECTED_ARCHITECTURE]}),
            encoding="utf-8",
        )
        (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        with (snapshot / "model.safetensors").open("wb") as stream:
            stream.truncate(MINIMUM_WEIGHT_BYTES)
        return snapshot

    def test_verifies_exact_revision_architecture_and_weight_size(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = verify_snapshot(self._snapshot(Path(directory)))

        self.assertEqual(manifest["model_revision"], MODEL_REVISION)
        self.assertEqual(manifest["architecture"], EXPECTED_ARCHITECTURE)
        self.assertEqual(manifest["weight_bytes"], MINIMUM_WEIGHT_BYTES)
        self.assertEqual(manifest["status"], "PASS")

    def test_rejects_wrong_architecture(self) -> None:
        with TemporaryDirectory() as directory:
            snapshot = self._snapshot(Path(directory))
            (snapshot / "config.json").write_text(
                json.dumps({"architectures": ["WrongModel"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unexpected architectures"):
                verify_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
