#!/bin/bash
# Explicit operator replacement of the Gemma login-side tmux controller only.
# Usage: bash horeka-gemma-v4-prequeue.sh WORKSPACE EXISTING_RUN
set -euo pipefail
set +x
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ] || { [ "$#" -eq 3 ] && [ "$3" != --prepare-on-login ]; }; then
  echo 'Usage: WORKSPACE EXISTING_RUN [--prepare-on-login]' >&2
  exit 2
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo 'Use a login shell; preserve the existing allocation.' >&2
  exit 2
fi
source "$1/geodml-nemotron-env.sh"
: "${RT:?Existing runtime required}"
GEMMA_ROOT=$2
GEMMA_HELPER=$(cd -- "$(dirname -- "$0")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
GEMMA_SCIENCE=$("$RT/bin/python" - "$GEMMA_ROOT" "$GEMMA_HELPER" <<'PY'
import json, shlex, subprocess, sys
from pathlib import Path
root, helper = map(Path, sys.argv[1:])
spec = json.loads((root / 'preparation.json').read_text())
command = shlex.split((root / 'prepare.sh').read_text().splitlines()[-1])
code = Path(next(p for p in command if p.endswith('/analysis/scripts/horeka_gemma_v4.py'))).resolve().parents[2]
for repo in (code, helper):
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise SystemExit(f'Dirty checkout: {repo}')
if subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != spec['git_commit']:
    raise SystemExit('Existing preparation pin differs')
if (root / 'sender/state.json').exists() and not (root / 'prequeue/state.json').exists():
    raise SystemExit('Inference sender already started; keep the existing sender')
print(code)
PY
)
GEMMA_ARGS=("$RT/bin/python" -u "$GEMMA_HELPER/analysis/scripts/horeka_gemma_v4_prequeue.py"
  --repository "$GEMMA_SCIENCE" --output "$GEMMA_ROOT")
if [ "${3:-}" = --prepare-on-login ]; then
  GEMMA_ARGS+=(--prepare-on-login)
fi
printf -v GEMMA_COMMAND '%q ' "${GEMMA_ARGS[@]}"
printf -v GEMMA_ENV '%q' "$1/geodml-nemotron-env.sh"
printf -v GEMMA_LOG '%q' "$GEMMA_ROOT/prequeue.log"
printf -v GEMMA_LAUNCHER '%q' "$GEMMA_ROOT/prequeue-command.sh"
printf '#!/bin/bash\nset -euo pipefail\nset +x\nsource %s\nexport PYTHONDONTWRITEBYTECODE=1\nunset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE\nexec %s >> %s 2>&1\n' \
  "$GEMMA_ENV" "$GEMMA_COMMAND" "$GEMMA_LOG" > "$GEMMA_ROOT/prequeue-command.sh"
# The user explicitly authorized replacing this named login-side tmux session.
# The preparation sbatch and all Qwen/interactive allocations remain untouched.
if tmux has-session -t '=gemma-v4-bouts' 2>/dev/null; then
  tmux kill-session -t '=gemma-v4-bouts'
  sleep 2
fi
tmux new-session -d -s gemma-v4-bouts "bash $GEMMA_LAUNCHER"
printf 'First-wave controller started. Log: %s/prequeue.log\n' "$GEMMA_ROOT"
printf 'Watch: tmux attach -t gemma-v4-bouts\n'
