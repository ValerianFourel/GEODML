# Gemma held first wave observed

Valerian ran pinned prequeue helper 51960c9540701d7a8163c92a4280b2edb07f6fc5.
Returned squeue lists all 200 IDs 5177534 through 5177733 as JobHeldUser.
Preparation 5177483 and unrelated Qwen 5177268 remain PENDING Priority.
This establishes accepted queued jobs, not running inference or live controller
health. These are pasted observations; Codex made no cluster query.

Explained that the user hold was deliberately submitted by our adapter. It waits
for preparation completion, verified inputs/ownership/quota and durable shard
assignment before automatic release. Manual release now could start jobs without
assigned.sh or leave the controller unable to verify its release history. No
release, cancellation, rerun, code change or budget expansion is appropriate
based on this output alone. Requested prequeue/status.json, prequeue.log, tmux
presence and preparation status to distinguish normal waiting from controller
failure. The pending preparation remains the immediate bottleneck.

Read the newest three indexed handoffs. Documentation only; whitespace check
passed. No executable tests rerun. Next: inspect returned controller evidence;
repair only if it reports a failure or fails to progress after preparation.
