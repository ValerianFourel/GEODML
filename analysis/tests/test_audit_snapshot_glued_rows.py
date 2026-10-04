"""Glued-title snapshot rows are found by other same-keyword titles inside the snippet."""
import gzip
import json

from analysis.interpretability.pipeline import page_readiness_ordering as ordering
from analysis.scripts import audit_snapshot_glued_rows as audit

ROWS = [
    {"keyword": "tax filing", "position": 1, "url": "https://a.example", "title": "How to File Taxes Online - TurboTax",
     "snippet": "How to File Taxes Online - TurboTax FreeTaxUSA Free Online Tax Filing File your taxes for free - IRS guide"},
    {"keyword": "tax filing", "position": 2, "url": "https://b.example", "title": "FreeTaxUSA Free Online Tax Filing",
     "snippet": "File federal returns for free with FreeTaxUSA."},
    {"keyword": "tax filing", "position": 3, "url": "https://c.example", "title": "File your taxes for free - IRS",
     "snippet": "The IRS offers free filing options for eligible taxpayers."},
    {"keyword": "tax filing", "position": 4, "url": "https://turbotax.example/tips",
     "title": "How to File Taxes Online - TurboTax FreeTaxUSA Free Online Tax Filing File your taxes for free - IRS",
     "snippet": "Step-by-step tips."},  # glued titles in the TITLE field, as in the real snapshot
    {"keyword": "school scheduling", "position": 1, "url": "https://d.example", "title": "Best School Scheduling Software 2026",
     "snippet": "Best School Scheduling Software 2026 compared: features and pricing."},
]


def test_flags_only_rows_that_contain_other_results_titles(tmp_path):
    snapshot = tmp_path / "ddg.jsonl"
    snapshot.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    assert audit.glued_titles(ROWS) == [3, 0, 0, 3, 0]  # glued rows can contain each other; own title repeated (last row) is not glued
    shown = ordering.page_id(ordering.page_text(ROWS[0]["title"], ROWS[0]["snippet"]))
    other = ordering.page_id(ordering.page_text(ROWS[1]["title"], ROWS[1]["snippet"]))
    assert audit.page_id(ROWS[0]["title"], ROWS[0]["snippet"]) == shown  # same identity as the extract
    report = audit.audit(snapshot, min_titles=2, min_title_chars=20, pages={shown: 30, other: 70})
    assert report["flagged_rows"] == 2 and report["keywords_with_a_flagged_row"] == 1
    assert len(report["review_sample"]) == 2 and report["title_chars_flagged"]["1.0"] > report["title_chars_all"]["0.5"]
    assert report["displays_of_flagged_snippets"] == 30 and report["display_share"] == 0.3
    pages = tmp_path / "pages.jsonl.gz"
    with gzip.open(pages, "wt") as stream:
        stream.write(json.dumps({"page_id": shown, "occurrences": 30}) + "\n")
    assert audit.main(["--snapshot", f"duckduckgo={snapshot}", "--pages", str(pages)]) == 0


def test_snapshots_are_located_by_the_hash_recorded_in_search_traces(tmp_path):
    import hashlib
    import pytest
    copy = tmp_path / "workspace/inputs/ddg-results.jsonl"
    copy.parent.mkdir(parents=True)
    copy.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    traces = [{"events": [{"event_type": "llm_call", "payload": {}}]},
              {"events": [{"event_type": "search", "payload": {"engine": "duckduckgo", "raw_payload": {
                  "snapshot": "/e/jupiter/frozen/ddg-results.jsonl", "snapshot_sha256": digest}}}]}]
    recorded = audit.snapshots_from_traces(traces)
    assert recorded == {"duckduckgo": {"path": "/e/jupiter/frozen/ddg-results.jsonl", "sha256": digest}}
    assert audit.resolve_snapshot(recorded["duckduckgo"], tmp_path / "workspace") == copy  # found by name, confirmed by hash
    copy.write_text("changed\n")
    with pytest.raises(FileNotFoundError, match="recorded hash"):
        audit.resolve_snapshot(recorded["duckduckgo"], tmp_path / "workspace")
    conflicting = traces + [{"events": [{"event_type": "search", "payload": {"engine": "duckduckgo", "raw_payload": {
        "snapshot": "/other.jsonl", "snapshot_sha256": "0" * 64}}}]}]
    with pytest.raises(ValueError, match="more than one"):
        audit.snapshots_from_traces(conflicting)
