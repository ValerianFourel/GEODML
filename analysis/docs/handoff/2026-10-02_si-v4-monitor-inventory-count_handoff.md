# Gemma blockage and mixed inventory counts

Valerian asked what blocks Gemma after pasting HoreKa status at 2026-10-02
02:15–02:17 UTC. These are timestamped user observations, not live tool checks.
The latest Qwen snapshot shows 893 allocations: 628 completed, 25 failed,
30 running and 210 pending. These are Slurm states, not verified scientific
completion counts. The account queue observation was 241 and then the user's
queue was 240; the 295 account limit did not block admission at that observation.
The three Qwen senders already had CAP=294. No sender was changed or restarted.

No live Gemma job was shown. Job 5174124 ended FAILED after 00:41:17, exit 1:0;
eight other Gemma requests were cancelled at zero elapsed time. Their cancellation
causes are not established. The old 20261001 input hashes passed, but all 214
checked source records still lacked hash metadata; no conflicting hashes were
found. This matches the previously diagnosed legacy metadata issue. The old
v4-pass1 index contained zero tasks versus 294 expected, and v3-bridge had no
index. These observations do not establish any completed Gemma judgment.

The separate monitoring failure was reproduced from the actual HTML command:
`AttributeError: 'int' object has no attribute 'values'`. Saved-input inventories
use task-type mappings, while `prepare_si_v4_diagnostics.freeze` emits an integer
`unique_tasks` total. Fixed only the HTML monitor to accept both existing forms
and reject negative/noninteger counts as unavailable. No input schema, frozen
data, inference behavior, execution pin or admission guard was changed.

Added one owner-boundary regression in `analysis/tests/test_horeka_si_v4.py`.
It executes the decoded status program on a small SQLite fixture, checks both
inventory forms, empty and missing indexes, success/failure/saved/running counts,
continuation to the final pass, and unchanged input/index bytes. It failed before
the repair with the user's exact exception. After the repair, the helper and
sibling Gemma suite passed all 44 tests using the installed Miniconda Python.
The system Python lacked pytest. All ten decoded shell commands pass `bash -n`,
and all eight embedded Python programs parse. `git diff --check` passes.
The workspace-root and Downloads `horeka-si-v4.html` exports match the source.

The approved workflow remains `reviews/gemma-si-v4-development-20261002-3h`,
using published execution pin `1bf21cb0e37374634be30a07bca9079ec18d6f87`.
Valerian's explicit one-new-three-hour-run exception alongside Qwen remains
valid; do not ask for it again. The page retains quota/storage, account capacity,
duplicate/attempt protection and ten-minute observed-start checks. There is no
evidence in this paste that the new preparation or allocation exists. Next step:
use the updated page's preparation, require HASH_REPAIR_VERIFIED, then admission.
Preserve any pre-existing destination or attempt marker and inspect it if the
command refuses to proceed. No remote command, allocation, inference, model
download, cancellation or manuscript change was performed. Commit locally;
the execution pin requires no change for this monitoring-only repair.
