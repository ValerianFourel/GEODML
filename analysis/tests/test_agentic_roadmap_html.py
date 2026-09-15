"""Verify the shipped offline dashboard's JavaScript, without a browser or GPU."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from analysis.scripts.report_agentic_paired_trial import collect_progress
from analysis.scripts.report_agentic_roadmap_snapshot import collect_snapshot

HTML = Path(__file__).resolve().parents[1] / "docs/agentic_experiment_roadmap.html"


def _javascript(assertions: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the offline dashboard checks")
    html = HTML.read_text(encoding="utf-8")
    script = re.search(r'<script id="roadmap-script">(.*?)</script>', html, re.DOTALL)
    assert script is not None
    source = script.group(1)
    result = subprocess.run(
        [node, "-"],
        input="const assert = require('node:assert/strict');\n" + source + "\n" + assertions,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_scopes_do_not_silently_credit_pilot_artifacts_to_full_rollout():
    _javascript("""
const snap = validateSnapshot(DEFAULT_SNAPSHOT);
const current = scopedProgress(snap, 500, false);
assert.equal(current.completed, 6000);
assert.equal(current.expected, 12000);
const full = scopedProgress(snap, 26009, false);
assert.equal(full.completed, 0);
assert.equal(full.expected, 624216);
const reusable = scopedProgress(snap, 26009, true);
assert.equal(reusable.completed, 6000);
assert.equal(reusable.models[0].expected, 312108);
assert.equal(reusable.models[1].completed, 0);
assert.equal(snap.pilot.completed, 0);
assert.throws(() => scopedProgress(snap, 1, false));
""")


def test_estimates_leave_unknown_rates_unknown_and_include_overhead():
    _javascript("""
const rate = rateRange('9', '11', 'Qwen');
assert.deepEqual(trialCapacity(60, 11, 5, rate),
  {usable: 44, low: 396, high: 484, pairGpuHours: 8});
assert.equal(rateRange('', '', 'Llama'), null);
assert.equal(remainingHours(6000, null), null);
assert.deepEqual(remainingHours(0, null), {low: 0, high: 0});
assert.deepEqual(trialCapacity(60, 11, 5, null),
  {usable: 44, low: null, high: null, pairGpuHours: 8});
assert.equal(trialCapacity(10, 11, 5, rate).high, 0);
assert.equal(trialCapacity(120, 11, 5, rate).pairGpuHours, 16);
assert.throws(() => rateRange('0', '11', 'Qwen'));
assert.throws(() => rateRange('12', '11', 'Qwen'));
assert.throws(() => rateRange('9', '', 'Qwen'));
assert.throws(() => trialCapacity(-1, 11, 5, rate));
assert.throws(() => trialCapacity(Infinity, 11, 5, rate));
""")


@pytest.mark.parametrize("mutation", [
    "snap.models[0].completed = 6001",
    "snap.models[0].completed = 0",
    "snap.models[0].shards[0].completed = 0.5",
    "snap.models[1].model = 'qwen38'",
    "snap.models[1].model = '__proto__'",
    "snap.models[0].shards[1].shard = 0",
    "snap.pilot.completed = 25",
    "snap.checked_at = 'yesterday'",
    "snap.models[0].expected = 312108",
])
def test_import_rejects_invalid_or_mismatched_counts(mutation):
    _javascript(
        "const snap = JSON.parse(JSON.stringify(DEFAULT_SNAPSHOT));\n"
        + mutation + ";\nassert.throws(() => validateSnapshot(snap));"
    )


def test_real_collector_schema_imports_without_a_live_scheduler(tmp_path):
    snapshot = collect_snapshot(tmp_path)
    _javascript(
        "const collected = validateSnapshot(" + json.dumps(snapshot) + ");\n"
        "assert.equal(scopedProgress(collected, 500, false).completed, 0);\n"
        "assert.equal(collected.pilot.status, 'not_inspected');"
    )


def test_page_is_offline_and_never_inserts_imported_html():
    source = HTML.read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "insertAdjacentHTML" not in source
    assert "fetch(" not in source
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', source)
    assert 'aria-live="polite"' in source
    assert 'id="import-snapshot"' in source
    assert 'id="count-reuse"' in source
    assert "does not submit jobs" in source


def test_trial_import_stays_separate_from_original_study(tmp_path):
    tasks = [{"cell_id": str(i)} for i in range(1440)]
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks)
    )
    trial = collect_progress(tmp_path, scheduler=False)
    _javascript(
        "const trial = validateTrial(" + json.dumps(trial) + ");\n"
        "const combined = validateSnapshot({...DEFAULT_SNAPSHOT, paired_trial: trial});\n"
        "assert.equal(combined.paired_trial.expected, 2880);\n"
        "assert.equal(scopedProgress(combined, 500, false).completed, 6000);\n"
        "assert.throws(() => validateTrial({...trial, completed: 500}));\n"
        "assert.throws(() => validateTrial({...trial, models: [trial.models[0], trial.models[0]]}));"
    )
