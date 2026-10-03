#!/bin/bash
# Print saved inference, assess the frozen standards, and measure cost. No allocation/model calls.
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  CODE="$W/checkouts/v4-interactive-1b34bebaf0007ed1010b4a4065a880cb4e10d73c"
  RUN="$W/reviews/si-v4-r2-cycle-20261002/development"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
  OUT=$(mktemp -d "$RUN/assessment-all-XXXXXXXX")
  printf 'Saving all output under: %s\n' "$OUT"
  "$RT/bin/python" -u - "$RUN" "$OUT" <<'PY' 2>&1 | tee "$OUT/console.txt"
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import fcntl, json, math, os, sqlite3, subprocess, sys, traceback
from analysis.scripts import run_si_v4_cycle as cycle

root, out = map(Path, sys.argv[1:])
config_path = root / 'config.json'
config = cycle.read(config_path)
with cycle.budget_lock(config['cycle_budget_path']) as ledger:
    cycle.checked_config(config_path, ledger)
    submissions = list(ledger['submissions'])
usage = cycle.allocation_usage(submissions)
usage_path = out / 'resource-usage.json'
usage_path.write_text(json.dumps(usage, indent=2) + '\n')
report_path = out / 'assessment.json'
summaries = {}

# Preserve a consistent idle snapshot while reading task records and reports.
with (root / 'evaluation-results/queue.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    with (out / 'all-inference.jsonl').open('x') as dump:
        def emit(value):
            dump.write(json.dumps(value, ensure_ascii=False) + '\n')
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        emit({'type': 'configuration', 'config': config,
              'queue': cycle.read(root / 'evaluation-results/queue-summary.json')})
        for item in cycle.read(root / 'evaluation-plan.json')['queue']:
            folder = root / 'evaluation-results' / item['name']
            report = cycle.latest_report(folder)
            if report is None:
                emit({'run': item['name'], 'status': 'REPORT_MISSING'})
                continue
            summaries[item['name']] = cycle.read(report / 'summary.json')
            emit({'run': item['name'], 'report': str(report), 'summary': summaries[item['name']]})
            database = folder / 'control/index.sqlite'
            with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                db.execute('PRAGMA query_only=ON')
                for task_id, kind, state, record, result in db.execute(
                        'SELECT id,kind,state,record,result FROM tasks ORDER BY kind,id'):
                    emit({'run': item['name'], 'task_id': task_id, 'kind': kind, 'state': state,
                          'input': json.loads(record) if record else None,
                          'result': json.loads(result) if result else None})
    try:
        cycle.assess(SimpleNamespace(config=config_path, output=report_path,
            references=None, reference_packets=None, supplement_references=None,
            supplement_packets=None, resource_usage=usage_path))
    except Exception:
        (out / 'assessment-error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        print('ASSESSMENT_FAILED: raw evidence preserved; no acceptance inferred.')

cycle.export(SimpleNamespace(config=config_path, output=out / 'evaluation-evidence.tar.gz'))

print('\nSTANDARDS SUMMARY')
if report_path.exists():
    report = cycle.read(report_path)
    print('DECISION', report['decision'])
    print('DEVELOPMENT_GATE_PASSED', report['development_gate_passed'])
    print('REFERENCES: independent reviews were not supplied to this command; reference-dependent failures/zeros are not measured semantic accuracy.')
    for name, gate in report['gates'].items():
        value = {key: gate[key] for key in ('status', 'threshold') if key in gate}
        if not isinstance(gate.get('observed'), (dict, list)):
            value['observed'] = gate.get('observed')
        print(name, json.dumps(value))
    print('MATCHED_COST', json.dumps(report['cost']))
    for mode, result in report['repeats'].items():
        print('REPEAT', mode, json.dumps({key: result[key] for key in
              ('counts', 'exact', 'within_one', 'exact_when_both_scored', 'nonempty_top_group')}))
    for name in ('constructed_controls', 'supplementary_controls'):
        print(name, json.dumps(report['gates'][name], ensure_ascii=False, indent=2))

print('\nTIME AND NODE-HOURS')
warm = sum(s.get('timing', {}).get('si', {}).get('node_hours', 0) for s in summaries.values())
print('ALL_DISTINCT_WORKLOADS', json.dumps({'warm_node_hours': warm,
    'warm_minutes': warm * 60, 'warm_gpu_hours': warm * 4}))
for name, summary in summaries.items():
    timing = summary.get('timing', {}).get('si', {})
    print(name, json.dumps({'warm_minutes': timing.get('node_hours', 0) * 60,
        'warm_node_hours': timing.get('node_hours'), 'warm_gpu_hours': timing.get('gpu_hours'),
        'completed_cells': summary.get('counts', {}).get('cells_complete')}))
print('ACTUAL_ALLOCATION_ACCOUNTING', json.dumps(usage, indent=2))
node_hours, nodes_known = 0.0, True
for allocation in usage['allocations']:
    fields = allocation.get('raw', '').strip().split('|')
    tres = dict(part.split('=', 1) for part in fields[3].split(',') if '=' in part) if len(fields) >= 4 else {}
    if len(fields) >= 4 and fields[2].isdigit() and tres.get('node', '').isdigit():
        node_hours += int(fields[2]) * int(tres['node']) / 3600
    else:
        nodes_known = False
print('ACTUAL_NODE_HOURS', json.dumps({'node_hours': node_hours if nodes_known else None,
                                     'accounting_complete': usage['complete']}))
jobs = sorted({row['job_id'] for row in submissions if row.get('job_id')})
if jobs:
    subprocess.run(['sacct', '-X', '-j', ','.join(jobs),
        '--format=JobID,State,Start,End,Elapsed,AllocNodes,AllocTRES%80'], check=True)

# Project full map+source execution only; fixed-map and v3 runs are different work.
rates = [summary['timing']['si']['node_hours'] / summary['counts']['cells_complete']
         for name, summary in summaries.items() if name.startswith('candidate-e2e-')
         and summary.get('counts', {}).get('cells_complete', 0) > 0]
if rates:
    target = int(os.environ.get('V4_REMAINING_CELLS', '1000'))
    if target <= 0:
        raise ValueError('V4_REMAINING_CELLS must be positive')
    low, high = min(rates) * target, max(rates) * target
    # Historical startup 8.9-10.9 min plus five-minute drain/cleanup in a one-hour slot.
    slots = [math.ceil(low / ((60 - 8.9 - 5) / 60)),
             math.ceil(high / ((60 - 10.9 - 5) / 60))]
    print('FULL_PASS_PROJECTION', json.dumps({'target_completed_cells': target,
        'scope': 'requested remaining cells' if 'V4_REMAINING_CELLS' in os.environ else 'per 1000 completed cells; actual remaining corpus count unknown',
        'warm_node_hours_range': [low, high],
        'one_hour_one_node_allocations_range': slots,
        'allocated_node_hours_range': slots,
        'allocated_gpu_hours_range': [value * 4 for value in slots],
        'assumptions': 'Same four-A100 Gemma settings and similar answer/source workload; historical startup margins; no semantic-quality guarantee or additional allocation authorization.'}, indent=2))
print('RETURN_DIRECTORY', out)
print('ALL_INFERENCE', out / 'all-inference.jsonl')
print('FULL_CONSOLE', out / 'console.txt')
PY
)
