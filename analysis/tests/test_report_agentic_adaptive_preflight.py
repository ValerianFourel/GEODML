"""The adaptive preflight discovers both direct and attempt-scoped worker configs."""

import json

from analysis.scripts.report_agentic_adaptive_preflight import run_record


def test_run_record_finds_nested_attempt_config(tmp_path):
    root = tmp_path / "run"
    config = root / "models/qwen38/outputs/attempts/job1/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps({"prompt_sources": {}, "search_snapshots": {}}),
        encoding="utf-8",
    )

    record = run_record(root)

    assert [item["path"] for item in record["configs"]] == [str(config.resolve())]
    assert record["issues"] == []
