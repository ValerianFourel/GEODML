# SI-v4 recovery commands for existing job 5175111

Valerian supplied new HoreKa evidence: CAP=294 was already set and the account
queue was 211 at that snapshot. Checking the old 20261001 preparation correctly
refused another allocation because an attempt exists. An SSH PAM failure was
followed by a successful login. `alloc` was a typo. Pending accelerated job
5175108 was revoked after the user pressed Ctrl-C. A separate one-hour
dev_accelerated allocation 5175111 was granted on hkn0402; an srun shell started.
The old run.sh then executed pin 9aa5d90 and failed at Coordinator._import with
the source dependency hash error, V4_EXIT=1. These are pasted observations;
current remaining time and allocation state have not been queried remotely.
The former pending job 5175056's present state is unknown.

The root cause is already fixed in the published source. The saved run.sh and
configuration still pin old code and inputs; exporting a different V4_PIN does
not rewrite that launcher. Admission check protects new allocation requests and
is not the correct continuation command. Do not delete its attempt marker or
weaken source-hash validation. No production code change is needed.

Added `analysis/docs/horeka-si-v4-existing-job.html`, adapting the previously
verified recovery instructions at 6e45b30. It uses published pin
1bf21cb0e37374634be30a07bca9079ec18d6f87; the relevant helpers are unchanged
between that pin and the current checkout. Copies are in the workspace root
and `/Users/valerianfourel/Downloads/horeka-si-v4-existing-job.html`.

Step 1 uses a separate login shell to verify/create a clean detached checkout.
The recovery checks running job 5175111, owner, account, dev_accelerated partition,
one-hour time limit and at least 1200 seconds remaining. It requires terminal
trial receipts and empty task indexes for the old failed-import attempts;
existing task rows stop recovery for reconciliation. It validates original
manifests, input files, bundles and judge configuration, then prepares
`reviews/gemma-si-v4-development-20261002-hashfix-job5175111` from those bundles.
Task records, cell contents and judge settings must match, apart from deriving
missing source hashes. All three Coordinator imports must pass. The new config
binds execution to 5175111 and a separate serving-cache prefix. A receipt records
verification and the existing allocation snapshot. Original inputs and attempts
remain untouched. Existing destinations refuse overwrite; incomplete repairs
have no verification receipt. No destructive rollback is provided.

Step 2 runs only in the existing compute shell. It checks the receipt, exact
saved launcher and pin, refuses another corrected attempt, refreshes quota and
storage evidence, and rechecks remaining time. The published runner retains
the exclusive-node boundary, actual end-time and cleanup margins. There is no
salloc, allocation-bearing srun, time extension or replacement. The command's
exit stays inside a child shell and preserves the owning shell. Step 3 reads
scheduler state and the new console log. The separate three-hour page is intact.

Estimate remains the recorded 20–50 minutes, with historical startup of roughly
9–11 minutes and unmeasured SI-v4 throughput. Only the existing allocation's
remaining node/GPU time is usable; partial work is possible. User explicitly
requested fixing this within the allocation already obtained; no new allocation
approval was requested or implied.

Verification: 44 existing SI-v4/Gemma tests passed. The exact two HTML Python
programs were exercised by `/tmp/verify-si-v4-existing-job.py` with the actual
legacy freeze function from 9aa5d90, actual frozen-input I/O and Coordinator
imports. The old hash failure reproduced; the corrected preparation preserved
old bytes and task text and passed three imports. Refusals passed for existing
output, prior task rows, insufficient time, wrong allocation, modified receipt
configuration, modified launcher and unsafe storage. Only runtime/model/quota
and scheduler boundaries used fixtures; no inference or scientific validation.
The proof caught a path-alias mismatch in launcher comparison; the page now
resolves the run root before comparing it. Bash, Python and JavaScript syntax,
ARIA references and matching exports passed. Browser verification remains
unavailable, as established in the preceding turn. No remote repair, inference,
allocation, cancellation, upload or push was performed. Commit locally.

Next: Valerian runs the page in order and returns HASH_REPAIR_VERIFIED, then
V4_EXIT and logs if the same allocation has enough time. If it is expired or
the guards refuse, preserve artifacts and inspect that evidence. Do not infer
the preparation or continuation succeeded from these local tests.
