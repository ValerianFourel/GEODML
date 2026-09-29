#!/usr/bin/env bash
# Paste into an already-open HoreKa login shell. Read-only; no inference or submission.
(
  set -eo pipefail
  date -Is
  squeue --me -o '%.18i %.38j %.10T %.12M %.12L %.20R'
  python3 - <<'PY_STATUS'
import json
from pathlib import Path
w = Path('/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen')
r = w / 'reviews/nemotron-si-v3-review20-961ad0b-2026092910'
print('REVIEW_ROOT', r)
m = r / 'selection-manifest.json'
print('FROZEN_SELECTION', m.is_file())
if m.is_file():
    print(m.read_text())
for model in ('qwen38', 'llama4'):
    run = r / 'runs' / model
    print('MODEL', model, 'RUN_EXISTS', run.exists())
    summaries = sorted(run.glob('attempts/job*/trial/summary.json'))
    if not summaries:
        print('NO_SAVED_SUMMARY: execution may be absent, active, or interrupted')
    for p in summaries:
        print('SUMMARY_PATH', p)
        print(p.read_text())
report = r / 'full-review'
print('FULL_REPORT_EXISTS', (report / 'all-judgments.txt').is_file())
print('CELL_REPORT_FILES', len(list(report.glob('cell-*.txt'))))
print('Do not restart existing runs before reconciling summaries, logs and live jobs.')
PY_STATUS
)
