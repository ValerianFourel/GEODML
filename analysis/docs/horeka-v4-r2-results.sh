#!/bin/bash
# Read the frozen queue and its saved latest reports. No allocation or inference.
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  CODE="$W/checkouts/v4-interactive-1b34bebaf0007ed1010b4a4065a880cb4e10d73c"
  RUN="$W/reviews/si-v4-r2-cycle-20261002/development"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
  "$RT/bin/python" - "$RUN" <<'PY'
from collections import Counter
from pathlib import Path
import gzip, json, sys
from analysis.scripts.run_si_v4_cycle import latest_report, read

root = Path(sys.argv[1])
config = read(root / 'config.json')
print('CONFIG', json.dumps({k: config.get(k) for k in
    ('git_commit', 'model_id', 'model_revision', 'evaluation_phase')}))
print('QUEUE', json.dumps(read(root / 'evaluation-results/queue-summary.json')))
budget = read(config['cycle_budget_path'])
print('ALLOCATIONS', json.dumps([{k: row.get(k) for k in ('job_id', 'phase', 'mode')}
    for row in budget['submissions']]))
for item in read(root / 'evaluation-plan.json')['queue']:
    report = latest_report(root / 'evaluation-results' / item['name'])
    if report is None:
        print(item['name'], 'REPORT_MISSING')
        continue
    summary = read(report / 'summary.json')
    print('\nRUN', item['name'], str(report))
    print(json.dumps({k: summary.get(k) for k in ('status', 'counts', 'states',
        'inference_failures', 'quarantined_maps', 'timing', 'semantic_acceptance')}, indent=2))
    errors = Counter()
    with gzip.open(report / 'cells.jsonl.gz', 'rt') as stream:
        for line in stream:
            cell = json.loads(line)
            results = [('map', cell.get('map_result') or {})]
            results += [('source', source) for source in cell.get('sources', [])]
            for kind, result in results:
                if result.get('error'):
                    errors[(kind, result['error'])] += 1
                parsed = result.get('parsed_output') or {}
                if parsed.get('status') == 'map_issue':
                    errors[(kind, 'map_issue: ' + str(parsed.get('note', '')))] += 1
    for (kind, error), count in errors.items():
        print('ERROR', count, kind, error)
PY
)
