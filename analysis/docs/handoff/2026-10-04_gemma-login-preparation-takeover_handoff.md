# Gemma preparation moved to login at explicit user request

User explicitly requested "make the preparation now on the shell login now".
This overrides the usual cluster-compute requirement for this preparation only.
The preceding pasted queue shows 200 held jobs 5177534–5177733, preparation
5177483 pending Priority, and unrelated Qwen 5177268 pending. No live cluster
query was made. The previous three handoffs and active code were reviewed.

Extended the prequeue shell with --prepare-on-login and added a takeover path.
It stops only the named login tmux controller, preserves original scientific pin,
locks the run and sender, holds the still-PENDING preparation, verifies first-wave
jobs are held, clears and verifies each afterok dependency BEFORE cancellation,
cancels only the held pending preparation, and waits for terminal accounting.
A running preparation is preserved and stops takeover. This order prevents
kill-on-invalid-dep from destroying the accepted 200 jobs. Mutation receipts and
original controller state are retained under prequeue/.

The helper invokes new prepare-login mode on the login host under nice 10 with
OMP/MKL/OpenBLAS thread counts one, no model load, a one-hour subprocess timeout
and existing storage/quota checks. Actual full-scale preparation duration/RAM are
unmeasured. Existing partial outputs are preserved and cause a stop rather than
a blind restart. A completed plan can resume adoption/release without re-freezing.
The prepare-login command requires a matching pinned takeover and cancelled GPU
preparation receipt. Normal GPU preparation still verifies the Slurm boundary.

Plan/config/run scripts keep original inference repository and commit; the
preparation execution metadata separately records helper revision, login host and
start/end timestamps. Model/prompt/settings, source scope, exclusion lists and
allocation ceilings are unchanged. The cancelled preparation hour remains in the
conservative ceiling; actual accounting is separate. No new GPU preparation or
extra inference jobs are submitted by this change. Existing prequeued jobs are
adopted and released; the original sender then continues ten-minute refill.

Validation: 41 focused Gemma runtime, sender and prequeue tests passed before a
final explicit retirement guard; affected preparation tests rerun after that
change. Real freeze/partition test covers both GPU and login preparation, and
scheduler lifecycle fixtures cover clearing dependencies before cancellation,
200-job adoption, preservation of a running preparation and refusal after an
unverified dependency update. Shell syntax and whitespace checks pass. No cluster
execution occurred locally. Next: user runs the pinned command and returns
prequeue.log; verify actual login preparation and subsequent job release.
