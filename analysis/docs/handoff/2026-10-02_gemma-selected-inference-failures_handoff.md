# Selected Gemma run completed with inference failures

Valerian returned actual output for job 5175121 on hkn0403, selection
`reviews/gemma-selected-xsk1n4bx`, helper pin d47418b. The launcher and model ran;
one v4 pass returned finished_with_failures and GEMMA_EXIT=2. Summary reports
20 cells, four complete, 55 scored sources, 52 sources blocked by failed maps,
six source inference failures and 17 failed inference tasks. Database states
are 101 done and 52 blocked. Done includes successful and failed durable tasks.
No admission stop is reported. Map/source request seconds overlap and must not
be summed into wall time. SI phase is approximately 16.83 minutes on-node,
1.122 GPU-hours; startup is excluded. Current allocation state is unknown.

Read last three handoffs. Traced Coordinator.ready/checkpoint/report and
run_acl_arr_vllm._execute_one. Failed answer maps prevent downstream source
judgments. The summary omits actual errors, failure categories, finish reasons
and validation-attempt details; it cannot establish token truncation, invalid
output or transport failure as the cause. No speculative prompt, token-limit,
model, validator or dataset changes were made. No retry or allocation launched.
Preserve saved results and attempt/ownership records. The next step is the
read-only diagnostic below, then a targeted repair based on returned evidence.
Do not rerun fresh blindly or delete task records to hide these failures.

Verified the diagnostic with a temporary SQLite fixture containing one failed
map, one blocked dependency and one successful source. It reports exactly one
failed request, retains the blocked count and truncation evidence, and leaves
database bytes unchanged. This is command verification, not a scientific result.
No new production code or test suite was needed. No cluster commands executed.

Run in the existing HoreKa login shell:

```bash
python3 - /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/reviews/gemma-selected-xsk1n4bx/run/attempts/job5175121/trial/v4-pass1 <<'PY'
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

path = Path(sys.argv[1]) / 'control/index.sqlite'
with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
    db.execute('PRAGMA query_only=ON')
    rows = db.execute('SELECT id,kind,state,result FROM tasks ORDER BY kind,id').fetchall()
counts = Counter()
failures = []
for task_id, kind, state, raw in rows:
    result = json.loads(raw) if raw else {}
    outcome = 'ok' if result.get('ok') is True else 'failed' if result.get('ok') is False else 'unknown'
    counts[(kind, state, outcome)] += 1
    if state == 'done' and result.get('ok') is False:
        failures.append((task_id, kind, result))
print('TASK_COUNTS')
for (kind, state, outcome), count in sorted(counts.items()):
    print(kind, state, outcome, count)
print('FAILED_REQUESTS', len(failures))
for task_id, kind, result in failures:
    print(json.dumps({
        'task_id': task_id, 'kind': kind,
        'error': result.get('error'),
        'failure_category': result.get('failure_category'),
        'usage': result.get('usage'),
        'duration_seconds': result.get('duration_seconds'),
        'raw_tail': str(result.get('raw_output') or '')[-350:],
        'attempts': [{
            'attempt': a.get('attempt'), 'error': a.get('error'),
            'failure_category': a.get('failure_category'), 'usage': a.get('usage')
        } for a in result.get('validation_attempts', [])]
    }, ensure_ascii=False))
PY
```
