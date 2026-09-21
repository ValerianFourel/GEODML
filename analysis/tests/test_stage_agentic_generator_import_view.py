"""Legacy generator imports never rewrite their historical output directory."""

import json

from analysis.scripts.stage_agentic_generator_import_view import stage


def test_stage_rewrites_only_copied_trace_path_and_preserves_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "view"
    for name in ("results", "traces", "diagnostics", "provenance"):
        (source / name).mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "run_manifest.json").write_text("{}")
    trace = source / "traces/cell.json"
    trace.write_text('{"trace_sha256":"digest"}')
    result = {"cell_id": "cell", "trace": str(trace.resolve())}
    (source / "results/cell.json").write_text(json.dumps(result))
    (source / "diagnostics/cell.json").write_text("{}")
    (source / "provenance/cell.json").write_text("{}")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert stage(source, output) == 1
    copied = json.loads((output / "results/cell.json").read_text())
    assert copied["trace"] == str((output / "traces/cell.json").resolve())
    assert before == {path: path.read_bytes() for path in before}
