#!/bin/bash
# Read saved results and scheduler evidence. No allocation or inference.
(
  set -u
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  V4_RUN="$W/reviews/si-v4-r2-cycle-20261002/development"
  squeue --me -o '%.18i %.40j %.10T %.10M %.10l %R'
  sacct -X -j 5175818 --format=JobID,State,Elapsed,Timelimit,ExitCode,AllocTRES%60
  "$RT/bin/python" - "$V4_RUN" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def show(label, path, keys=None):
    if not path.is_file():
        print(label, 'NOT_FOUND', str(path))
        return
    value = json.loads(path.read_text())
    print(label, json.dumps({k: value[k] for k in keys if k in value} if keys else value, indent=2))
show('RUN', root / 'config.json', ['git_commit', 'model_id', 'model_revision', 'evaluation_phase', 'walltime'])
show('BUDGET', root.parent / 'cycle-budget.json', ['budget', 'submissions'])
show('QUEUE', root / 'evaluation-results/queue-summary.json')
for pointer in sorted((root / 'evaluation-results').glob('*/reports/latest.json')):
    name = json.loads(pointer.read_text())['directory']
    if not isinstance(name, str) or not re.fullmatch(r'si-[a-f0-9]+', name):
        raise ValueError('invalid report pointer: ' + str(pointer))
    report = pointer.parent / name
    print('REPORT_DIRECTORY', str(report))
    show(pointer.parents[1].name, report / 'summary.json',
         ['status', 'counts', 'states', 'inference_failures', 'quarantined_maps', 'timing', 'request_totals', 'semantic_acceptance'])
PY
  if [ -f "$V4_RUN/console-job5175818.log" ]; then
    tail -n 60 "$V4_RUN/console-job5175818.log"
  fi
  if [ -f "$W/reviews/qwen-recovery-5h-20261003/sender.log" ]; then
    tail -n 40 "$W/reviews/qwen-recovery-5h-20261003/sender.log"
  fi
)
