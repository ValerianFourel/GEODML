"""Contracts for the Phase-1 semantic-readiness corpus freeze."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    SEMANTIC_DATASET_VERSION,
    WebTextRecord,
    build_semantic_readiness_corpus,
)
from analysis.interpretability.pipeline.semantic_readiness_phase1 import (
    SurfaceSnapshotInput,
    TransferSnapshotInput,
    build_phase1_freeze_audit,
)
from analysis.interpretability.pipeline.semantic_readiness_transfer import (
    DEFAULT_TRANSFER_SPEC,
)


class SemanticReadinessPhase1Tests(unittest.TestCase):
    def test_freeze_reconstructs_base_and_reports_missing_transfer_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, expected = _fixture(Path(directory))
            audit = build_phase1_freeze_audit(
                **paths,
                expected_counts=expected,
            )

        base = audit["base_corpus"]
        transfer = audit["transfer_registry"]
        gate = audit["phase_gate"]
        self.assertTrue(base["frozen"])
        self.assertTrue(base["exact_rebuild_matches"])
        self.assertEqual(base["cross_split_group_count"], 0)
        self.assertEqual(base["unknown_license_count"], 0)
        self.assertEqual(audit["web_inputs"]["licensed_included_count"], 1)
        self.assertEqual(audit["web_inputs"]["unknown_license_excluded_count"], 1)
        self.assertEqual(transfer["source_count"], 8)
        self.assertFalse(transfer["all_snapshots_present"])
        self.assertTrue(gate["safe_to_begin_base_only_annotation"])
        self.assertFalse(gate["safe_to_begin_full_phase2"])
        self.assertEqual(gate["status"], "phase1-transfer-snapshots-pending")
        self.assertFalse(audit["annotation_performed"])
        self.assertFalse(audit["model_inference_performed"])

    def test_transfer_snapshot_requires_both_local_bytes_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expected = _fixture(root)
            snapshot = root / "oasst1.jsonl"
            snapshot.write_text('{"message_id":"one"}\n', encoding="utf-8")
            snapshot_sha = _sha256(snapshot.read_bytes())
            audit = build_phase1_freeze_audit(
                **paths,
                transfer_inputs=(
                    TransferSnapshotInput(
                        source_id="openassistant-oasst1",
                        path=snapshot,
                        revision="pinned-revision",
                    ),
                ),
                expected_counts=expected,
            )

        sources = {
            row["source_id"]: row for row in audit["transfer_registry"]["sources"]
        }
        self.assertTrue(sources["openassistant-oasst1"]["frozen"])
        self.assertEqual(
            sources["openassistant-oasst1"]["snapshot_sha256"],
            snapshot_sha,
        )
        self.assertFalse(sources["lmsys-chat-1m"]["frozen"])

    def test_surface_snapshot_override_makes_local_provenance_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expected = _fixture(root)
            provenance_path = Path(paths["surface_provenance_path"])
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            original = Path(provenance["sources"][0]["resolved_file"])
            portable = root / "staged" / "dolly.raw"
            portable.parent.mkdir()
            original.replace(portable)
            audit = build_phase1_freeze_audit(
                **paths,
                surface_inputs=(
                    SurfaceSnapshotInput(
                        source_id="databricks-dolly-15k",
                        path=portable,
                    ),
                ),
                expected_counts=expected,
            )

        sources = {
            row["source_id"]: row for row in audit["surface_inputs"]["sources"]
        }
        self.assertTrue(audit["base_corpus"]["frozen"])
        self.assertTrue(sources["databricks-dolly-15k"]["verified"])
        self.assertTrue(sources["databricks-dolly-15k"]["path_override_used"])
        self.assertEqual(
            Path(sources["databricks-dolly-15k"]["local_path"]),
            portable.resolve(),
        )

    def test_unknown_surface_snapshot_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expected = _fixture(root)
            with self.assertRaisesRegex(ValueError, "unknown surface source input"):
                build_phase1_freeze_audit(
                    **paths,
                    surface_inputs=(
                        SurfaceSnapshotInput(
                            source_id="not-in-provenance",
                            path=root / "missing.raw",
                        ),
                    ),
                    expected_counts=expected,
                )

    def test_empty_transfer_snapshot_is_not_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expected = _fixture(root)
            empty_snapshot = root / "empty.jsonl"
            empty_snapshot.touch()
            audit = build_phase1_freeze_audit(
                **paths,
                transfer_inputs=(
                    TransferSnapshotInput(
                        source_id="openassistant-oasst1",
                        path=empty_snapshot,
                        revision="pinned-revision",
                    ),
                ),
                expected_counts=expected,
            )

        sources = {
            row["source_id"]: row for row in audit["transfer_registry"]["sources"]
        }
        self.assertTrue(sources["openassistant-oasst1"]["snapshot_present"])
        self.assertEqual(sources["openassistant-oasst1"]["snapshot_size_bytes"], 0)
        self.assertFalse(sources["openassistant-oasst1"]["frozen"])

    def test_cross_split_group_is_a_hard_layer1_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, expected = _fixture(root)
            corpus_path = Path(paths["base_corpus_path"])
            rows = _read_jsonl(corpus_path)
            development = next(row for row in rows if row["split"] == "development")
            confirmation = next(row for row in rows if row["split"] == "confirmation")
            confirmation["group_id"] = development["group_id"]
            _write_jsonl(corpus_path, rows)
            manifest_path = Path(paths["base_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["corpus_sha256"] = _sha256(corpus_path.read_bytes())
            _write_json(manifest_path, manifest)

            audit = build_phase1_freeze_audit(
                **paths,
                expected_counts=expected,
            )

        self.assertFalse(audit["base_corpus"]["frozen"])
        self.assertEqual(audit["base_corpus"]["cross_split_group_count"], 1)
        self.assertEqual(audit["phase_gate"]["status"], "phase1-blocked")


def _fixture(root: Path) -> tuple[dict[str, Path], dict[tuple[str, str], int]]:
    surface_rows = [
        {
            "source_id": "databricks-dolly-15k",
            "source_record_id": "dolly:1",
            "text": "Explain how residential solar panels generate electricity.",
            "corpus_split": "development",
            "surface_family_id": "surface-family:explanation",
        },
        {
            "source_id": "anthropic-hh-helpful-base",
            "source_record_id": "hh:1",
            "text": "Choose a suitable account and prepare the application.",
            "corpus_split": "confirmation",
            "surface_family_id": "surface-family:action",
        },
    ]
    web_rows = (
        WebTextRecord(
            web_record_id="web:licensed",
            source_platform="stackexchange",
            source_site="diy",
            source_record_id="10",
            text="Which insulation should I select for this renovation?",
            url="https://diy.stackexchange.com/questions/10/example",
            author_name="Example User",
            author_url=None,
            license="CC BY-SA 4.0",
            tags=("insulation",),
            creation_timestamp=1,
            score=2,
            retrieval_probe_ids=("probe-one",),
            retrieval_sampling_regions=("selection",),
            split="confirmation",
        ),
        WebTextRecord(
            web_record_id="web:unlicensed",
            source_platform="stackexchange",
            source_site="diy",
            source_record_id="11",
            text="How should this repair be planned safely?",
            url="https://diy.stackexchange.com/questions/11/example",
            author_name=None,
            author_url=None,
            license="unknown",
            tags=("repair",),
            creation_timestamp=2,
            score=1,
            retrieval_probe_ids=("probe-one",),
            retrieval_sampling_regions=("selection",),
            split="confirmation",
        ),
    )
    corpus = build_semantic_readiness_corpus(surface_rows, web_rows)

    base_corpus = root / "semantic_readiness_corpus.jsonl"
    surface_corpus = root / "surface_coverage_corpus.jsonl"
    web_records = root / "web_text_records.jsonl"
    _write_jsonl(base_corpus, [asdict(item) for item in corpus])
    _write_jsonl(surface_corpus, surface_rows)
    _write_jsonl(web_records, [asdict(item) for item in web_rows])

    base_manifest = root / "base_manifest.json"
    _write_json(
        base_manifest,
        {
            "artifact_version": SEMANTIC_DATASET_VERSION,
            "corpus_count": len(corpus),
            "corpus_sha256": _sha256(base_corpus.read_bytes()),
            "surface_corpus_sha256": _sha256(surface_corpus.read_bytes()),
            "web_records_sha256": _sha256(web_records.read_bytes()),
            "web_included_count": 1,
            "web_excluded_count": 1,
            "semantic_labels_present": False,
            "reranking_outcomes_observed": False,
        },
    )
    surface_manifest = root / "surface_manifest.json"
    _write_json(
        surface_manifest,
        {"corpus_sha256": _sha256(surface_corpus.read_bytes())},
    )

    dolly_raw = root / "dolly.raw"
    hh_raw = root / "hh.raw"
    dolly_raw.write_text("frozen dolly bytes\n", encoding="utf-8")
    hh_raw.write_text("frozen hh bytes\n", encoding="utf-8")
    surface_provenance = root / "surface_provenance.json"
    _write_json(
        surface_provenance,
        {
            "sources": [
                {
                    "source_id": "databricks-dolly-15k",
                    "revision": "dolly-revision",
                    "license": "CC-BY-SA-3.0",
                    "dataset_card": "https://example.test/dolly",
                    "resolved_file": str(dolly_raw),
                    "verified_sha256": _sha256(dolly_raw.read_bytes()),
                },
                {
                    "source_id": "anthropic-hh-helpful-base",
                    "revision": "hh-revision",
                    "license": "MIT",
                    "dataset_card": "https://example.test/hh",
                    "resolved_file": str(hh_raw),
                    "verified_sha256": _sha256(hh_raw.read_bytes()),
                },
            ]
        },
    )
    web_manifest = root / "web_manifest.json"
    _write_json(web_manifest, {"unique_web_record_count": len(web_rows)})
    web_spec = root / "web_spec.json"
    _write_json(
        web_spec,
        {
            "specification_version": SEMANTIC_DATASET_VERSION,
            "probes": [
                {
                    "probe_id": "probe-one",
                    "site": "diy",
                    "query": "choose",
                    "sampling_region": "selection",
                    "split": "confirmation",
                }
            ],
        },
    )
    raw_responses = root / "raw_responses"
    raw_responses.mkdir()
    _write_json(raw_responses / "probe-one.json", {"items": []})

    expected = {
        ("databricks-dolly-15k", "development"): 1,
        ("anthropic-hh-helpful-base", "confirmation"): 1,
        ("stackexchange:diy", "confirmation"): 1,
    }
    return (
        {
            "base_corpus_path": base_corpus,
            "base_manifest_path": base_manifest,
            "surface_corpus_path": surface_corpus,
            "surface_manifest_path": surface_manifest,
            "surface_provenance_path": surface_provenance,
            "web_records_path": web_records,
            "web_manifest_path": web_manifest,
            "web_raw_responses_dir": raw_responses,
            "web_specification_path": web_spec,
            "transfer_specification_path": DEFAULT_TRANSFER_SPEC,
        },
        expected,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
