# JUPITER live-status commands, 1 October 2026

Valerian requested paste-ready commands to see what is happening on JUPITER.
Provided read-only scheduler, tmux, process and bounded log-tail checks for an
already-open JUPITER shell. No environment setup, HF call, allocation, upload,
termination or SSH invocation is needed. No live cluster output was returned
on this turn; the previous HF snapshot remains the latest verified publication
evidence.

The historical transfer host is `jpbl-s03-01`. Slurm checks cover the user's jobs
cluster-wide; tmux and process listings cover only the current host. Missing logs
or no local tmux server do not establish completion. Process listings use command
names rather than arguments to avoid exposing credentials passed to commands.
Log reads are bounded to the final 16 KiB per file and the final 20 lines.

## Commands supplied

```bash
(
  hostname -s
  date -Is
  squeue --me --format='%.12i %.45j %.12T %.10M %.12l %.24R'
  sacct -X --user="$USER" --starttime=2026-09-29 \
    --format=JobID,JobName%45,State,Elapsed,ExitCode,NodeList%25
  tmux list-sessions
  tmux list-panes -a -F '#S:#I.#P pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}'
  ps -u "$(id -u)" -o pid,ppid,etime,pcpu,pmem,comm --sort=-pcpu
)
```

```bash
python3 - <<'PY'
import os
from datetime import datetime
from pathlib import Path

base = Path('/e/fscratch/scifi') / os.environ['USER'] / 'geodml'
checks = [
    ('General archive', base/'audits/leave-jupiter/hf-archive', 'run-*.log'),
    ('Activations', base/'audits/publication-audit', 'activations-upload-*.log'),
    ('GeoAxis prompts', base/'audits/leave-jupiter', 'geoaxis-26k-upload-*.log'),
    ('Readiness axis', base/'audits/leave-jupiter', 'geoaxis-axis-upload-*.log'),
]
for label, folder, pattern in checks:
    print('\n' + label, flush=True)
    try:
        logs = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not logs:
            print('NO_LOG_FOUND:', folder, flush=True)
            continue
        path = logs[-1]
        info = path.stat()
        print(path, '\nLast modified:', datetime.fromtimestamp(info.st_mtime).isoformat(), flush=True)
        with path.open('rb') as stream:
            stream.seek(max(0, info.st_size - 16384))
            print('\n'.join(stream.read(16384).decode(errors='replace').splitlines()[-20:]), flush=True)
    except OSError as error:
        print(type(error).__name__ + ': ' + str(error), flush=True)
PY
```

Both Bash blocks and embedded Python syntax were checked locally. No production
code changed. Next step is to interpret the returned scheduler and log output;
do not restart transfers based merely on missing output.
