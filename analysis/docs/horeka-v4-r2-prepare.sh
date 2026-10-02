#!/bin/bash
# Render __SI_V4_PIN__ to the published commit before running this handoff.
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  SI_V4_PIN=__SI_V4_PIN__
  CODE="$W/checkouts/si-v4-r2-$SI_V4_PIN"
  CYCLE="$W/reviews/si-v4-r2-cycle-20261002"
  SOURCE_RUN="$W/reviews/gemma-si-v4-development-20261001"
  git -C "$REPO" fetch origin codex/gemma-selected-cells
  if [ ! -d "$CODE" ]; then
    git -C "$REPO" worktree add --detach "$CODE" "$SI_V4_PIN"
  fi
  test "$(git -C "$CODE" rev-parse HEAD)" = "$SI_V4_PIN"
  test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
  mkdir -p "$CYCLE"
  "$RT/bin/python" - "$SOURCE_RUN" "$CYCLE/proctoring-list-spec.json" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
from analysis.scripts.horeka_si_v4 import saved_pool
from analysis.interpretability.pipeline.source_importance import mask_answer_citations
pool, _ = saved_pool(Path(sys.argv[1]))
cell, tasks = pool['2e4098d7ee3d2547ee74']
answer, _ = mask_answer_citations(tasks[cell['j1_task_id']]['answer'], cell['presented'])
if hashlib.sha256(answer.encode()).hexdigest() != '21ace912b4d28bec360646d9e2c510bbb601855a5eec87c964836f49c9aa5978':
    raise ValueError('known proctoring answer differs; preserve it and inspect')
prefix, body = answer.split('1) ', 1)
body, suffix = body.split(' These tools are evaluated', 1)
items = re.split(r' [2-5]\) ', body)
if len(items) != 5:
    raise ValueError('expected exactly five independently reorderable factors')
spec = dict(prefix=prefix, items=items, suffix=' These tools are evaluated' + suffix,
            independent_items_confirmed=True)
path = Path(sys.argv[2])
if path.exists():
    if json.loads(path.read_text()) != spec:
        raise ValueError('existing list specification differs')
else:
    with path.open('x') as stream:
        stream.write(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
PY
  "$RT/bin/python" "$CODE/analysis/scripts/run_si_v4_cycle.py" prepare-development \
    --source-run "$SOURCE_RUN" --output "$CYCLE/preparation" \
    --list-cell-id 2e4098d7ee3d2547ee74 --list-spec "$CYCLE/proctoring-list-spec.json"
  "$RT/bin/python" "$CODE/analysis/scripts/horeka_si_v4.py" prepare \
    --workspace "$W" --output "$CYCLE/development" \
    --evaluation-plan "$CYCLE/preparation/queue/evaluation-plan.json" \
    --cycle-budget "$CYCLE/cycle-budget.json" --account hk-project-p0026831 \
    --walltime 01:00:00 \
    --approval 'Valerian approved one revised-v4 cycle: maximum 32 GPU-hours total, 16 development and 16 fresh, at most four one-hour four-A100 allocations per phase, 32 requested CPUs and node-default memory. This preparation is development only; no production launch or budget expansion.'
)
