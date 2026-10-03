#!/bin/bash
# Run inside tmux on a HoreKa login host after sourcing the existing environment.
# One allocation only: eight saved cells, 30 minutes, 1 node / 4 A100 / 32 CPUs.
set -euo pipefail
set +x
workspace=${1:?workspace required}
source_run=${2:?Gemma evaluation directory required}
output=${3:?new output directory required}
account=${4:?account required}
test -z "${SLURM_JOB_ID:-}" || { echo 'Use a separate login shell; preserve the existing allocation.'; exit 1; }
test ! -e "$output" || { echo "Inspect existing preparation: $output"; exit 1; }
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime="$workspace/environment/qwen-runtime/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$code"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# Restore only the requested pinned model if its verified cache is absent.
if ! "$runtime" - "$workspace" <<'PY'
import sys
from pathlib import Path
from analysis.scripts import horeka_nemotron as stage
try:
    receipt = stage.read(stage.preparation_dir(Path(sys.argv[1])) / 'models-verified.json')
    assert receipt['status'] == 'verified'
    assert [(r['repo_id'], r['revision']) for r in receipt['models']] == list(stage.NEMOTRON)
    assert all(Path(r['snapshot']).is_dir() for r in receipt['models'])
except (OSError, KeyError, ValueError, AssertionError):
    sys.exit(1)
PY
then
  "$runtime" "$code/analysis/scripts/horeka_nemotron.py" download \
    --workspace "$workspace" --account "$account"
fi

"$runtime" "$code/analysis/scripts/horeka_si_v4.py" prepare-nemotron \
  --workspace "$workspace" --source-run "$source_run" --output "$output" \
  --account "$account" --count 8 --seed 20261004 --walltime 00:30:00 \
  --approval 'Valerian requested a Nemotron v4 diagnostic aiming for about 20 minutes; eight cells, one 30-minute allocation, maximum 2 GPU-hours.'

"$runtime" -u - "$output" <<'PY' 2>&1 | tee -a "$output/controller.log"
import json, os, re, subprocess, sys, time
from pathlib import Path
from analysis.scripts import horeka_si_v4 as pilot
root = Path(sys.argv[1]).resolve()
config = pilot.stage.read(root / 'config.json')
pilot.main(['check', '--output', str(root)])
command = ['salloc', '--no-shell', '--immediate=30', '--nodes=1', '--ntasks=1',
           '--gres=gpu:4', '--exclusive', '--cpus-per-task=32', '--mem=0',
           '--partition=accelerated', '--time=' + config['walltime'],
           '--account=' + config['account'], '--job-name=' + config['job_name']]
with (root / 'ALLOCATION_ATTEMPTED').open('x') as f:
    json.dump({'command': command, 'time': time.time(),
               'config_sha256': pilot.judge.file_hash(root / 'config.json')}, f)
result = subprocess.run(command, text=True, capture_output=True, env={**os.environ, 'LC_ALL': 'C'})
(root / 'allocation.json').write_text(json.dumps({
    'command': command, 'returncode': result.returncode,
    'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
print(result.stdout, end='')
print(result.stderr, end='')
jobs = re.findall(r'^salloc: Granted job allocation (\d+)\s*$', result.stderr, re.M)
if result.returncode or len(jobs) != 1:
    raise SystemExit('No unambiguous successful allocation. Inspect allocation.json and squeue before retrying.')
job = jobs[0]
(root / 'job-id.txt').write_text(job + '\n')
print('JOB_ID=' + job, 'OUTPUT=' + str(root), flush=True)
step = subprocess.run(['srun', '--jobid=' + job, '--nodes=1', '--ntasks=1',
                       '--cpus-per-task=32', '--gres=gpu:4', '--unbuffered',
                       'bash', str(root / 'run.sh')])
print(f'STEP_EXIT_CODE={step.returncode} ALLOCATION_PRESERVED={job}', flush=True)
for report in sorted(root.glob('attempts/job*/trial/v4-pass1/reports/*/summary.json')):
    summary = pilot.stage.read(report)
    print(json.dumps({'report': str(report), **{k: summary.get(k) for k in
          ('status', 'counts', 'inference_failures', 'timing')}}, indent=2))
subprocess.run(['sacct', '-X', '-j', job, '--format=JobID,State,Elapsed,AllocTRES'])
raise SystemExit(step.returncode)
PY
