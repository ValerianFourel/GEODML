# New three-hour SI-v4 launch HTML

Replaced the cumulative recovery page with one six-step sequence: prepare corrected inputs, admission, request one approved three-hour allocation, enter compute shell, run finite queue, inspect logs. Keep state/count/forecast sections. Execution pin aab61590c6aa05ad088e4069cfc9f0f84e7eb02a is published to origin/codex/pilot-continuation. Preparation uses reviews/gemma-si-v4-development-20261002-3h and explicitly records the user's three-hour request. Prior one-hour paths appear only as read-only input/progress checks. No job 5174124 binding remains.

Preparation verifies original hashes and refuses observed prior judge work needing reconciliation. It compares unchanged tasks and added metadata, then validates all three imports. Allocation rechecks admission and receipt, refuses nested allocation shells, creates a no-clobber attempt marker and requests 03:00:00. New admission uses default concurrency/no-pending rules. User's answer to the requested new-run queue exception remains pending; no previous one-run exception is silently extended. Duplicate Gemma jobs, fresh quota, account capacity and ten-minute start checks remain enforced. Old allocation end time is historical user evidence only.

Estimate shown: unchanged finite workload 20–50 minutes, unmeasured v4 throughput; approved three hours cap 3 node-hours / 12 GPU-hours / up to 456 allocated CPU-hours. No work expansion, unbounded retry, new allocation submission or inference occurred. Keep all old allocations and results.

Proof: helper preparation/admission and sibling suite 34 tests passed. All decoded HTML shell blocks and embedded Python parse; anchors are unique/resolved; exactly one three-hour salloc command exists and no old job binding or one-hour allocation remains. Root and Downloads exports match. Source fix pinned commit is published. No browser interaction or cluster execution is claimed.

Next: user runs preparation from the updated page. Default admission will block alongside a crowded Qwen queue until either it clears or the user explicitly approves a new scoped exception. Do not infer that silence approves that exception.
