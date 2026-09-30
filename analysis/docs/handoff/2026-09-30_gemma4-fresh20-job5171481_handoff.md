# Fresh Gemma diagnostic moved to user-obtained job 5171481

Valerian reports job 5171305 is no longer recognized by scontrol. The page's
step 1 failed on that status read before fetching the code. The explicit old
job guard also stopped step 2 before freezing or preparation.

New pasted evidence: user cancelled pending job 5171480, then obtained job
5171481 on hkn0402 with the same one-hour, exclusive four-GPU, 32-requested-CPU,
whole-node-memory Gemma allocation. The prolog warning resolved and nodes were
ready. This is pasted evidence, not independently observed live state.

Updated the HTML commands to bind freeze, preparation, execution and result
paths to 5171481, including the approval record. Removed the job-status read
from code-fetch setup so code synchronization does not depend on an expired job.
No Python changes needed; code pin remains 779c897cae570949d09e804d1e2f550995af6cc3.
Original input exclusion and fresh sample seed remain unchanged. Existing
conflicting artifacts are still refused, never overwritten or deleted.

Both HTML copies match. Extracted Bash blocks and embedded Python parse. No
allocation, termination, queue mutation or remote command executed by Codex.
Next: step 1 in separate login shell, steps 2–3 in compute job 5171481, then
return summaries and source judgments. The existing 20-minute remaining-time,
whole-node boundary, immutable inputs and attempt guards stay in force.
