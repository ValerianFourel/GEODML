# HoreKa interactive allocation count

Valerian asked how many interactive shells are in use after a one-hour Gemma
`salloc` appeared not to work. No output or error from that command was supplied.
Read the latest three indexed handoffs; the latest pasted cluster evidence is
still the 03:52 UTC Qwen snapshot and Gemma 5175056 pending. These are historical
observations, not a live check. Do not infer an allocation limit or failed request.

The command below makes one read-only scheduler query and counts this user's
non-batch allocations by state. Slurm documents BatchFlag=0 for jobs submitted
without sbatch, including salloc and direct srun. This counts allocations, not
SSH windows, attached shell processes, or job steps. BatchFlag may exceed one
after a batch requeue. Missing fields and scheduler errors are unavailable,
not zero. Terminal retained records do not inflate active counts. Gemma records
are also shown, including any recent terminal record retained by the controller.

Source checked: https://slurm.schedmd.com/scontrol.html, BatchFlag and --oneliner.
No remote execution, new allocation, cancellation, launcher edit or upload.
The pasted one-hour request differs from the prepared three-hour workflow.
Preserve job 5175056 if it still exists and any allocation-owning shell. The
existing one-new-three-hour approval does not authorize duplicate allocations.

Paste in an existing HoreKa login terminal, using a second terminal if salloc
is holding the first one:

```bash
python3 - <<'PY'
from collections import Counter
from datetime import datetime, timezone
import os
import re
import subprocess

print("SNAPSHOT", datetime.now(timezone.utc).isoformat())
print("THIS_SHELL_SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID", "unset"))
try:
    result = subprocess.run(["scontrol", "--oneliner", "show", "job"],
                            text=True, capture_output=True, check=True, timeout=30)
except (OSError, subprocess.SubprocessError) as error:
    raise SystemExit("STATUS_UNAVAILABLE: " + str(error) + "\n" +
                     str(getattr(error, "stderr", "") or ""))
if result.stderr.strip():
    print("SLURM_WARNING", result.stderr.strip())
if not result.stdout.strip():
    raise SystemExit("STATUS_UNAVAILABLE: empty scheduler response")
jobs = []
for line in result.stdout.splitlines():
    if not line.strip() or line.strip() == "No jobs in the system":
        continue
    job = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", line))
    if "JobId" not in job or "UserId" not in job:
        raise SystemExit("STATUS_UNAVAILABLE: unexpected scheduler record")
    if not job["UserId"].endswith("(" + str(os.getuid()) + ")"):
        continue
    if "JobState" not in job or not job.get("BatchFlag", "").isdigit():
        raise SystemExit("STATUS_UNAVAILABLE: missing state or BatchFlag")
    jobs.append(job)
terminal = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
active = [j for j in jobs if j["JobState"].split("+")[0] not in terminal]
interactive = [j for j in active if j["BatchFlag"] == "0"]
counts = Counter(j["JobState"].split("+")[0] for j in interactive)
print("INTERACTIVE_RUNNING", counts["RUNNING"])
print("INTERACTIVE_PENDING", counts["PENDING"])
print("INTERACTIVE_ALL_STATES", dict(counts))
print("ALL_MY_ACTIVE_JOBS", dict(Counter(j["JobState"] for j in active)))
for title, rows in [("INTERACTIVE_ALLOCATIONS", interactive),
                    ("GEMMA", [j for j in jobs if j.get("JobName") == "geodml-gemma-si-v4"])]:
    print("\n" + title)
    print("ID | NAME | BATCH_FLAG | STATE | ELAPSED | LIMIT | NODES | REASON")
    for j in rows:
        print(" | ".join(j.get(k, "?") for k in
              ("JobId", "JobName", "BatchFlag", "JobState", "RunTime",
               "TimeLimit", "NodeList", "Reason")))
    if not rows:
        print("NONE in this scheduler snapshot")
PY
```

Local verification passed: exercised the exact embedded command with scheduler output
fixtures for running and pending non-batch jobs, a requeued batch job,
a retained completed allocation, another user's allocation, missing BatchFlag,
an empty response, explicit no-jobs response, and a scheduler error. No cluster
counts are inferred from fixtures. Bash syntax and Python parsing also pass.
The next step is to inspect returned states, reasons and the actual salloc
message before proposing any scheduling action.
