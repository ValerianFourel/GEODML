# Gemma v4 llama-only relaunch through Slurm preparation

Read the Codex thread 01a102c8 and the last three handoffs. The thread ended
after reset 3f32275 cancelled 200 held jobs, kept Qwen 5177268 and started a
reset controller on hkn1990; its reset.log was not yet shown. Pasted, not live.

New request: free the Gemma locks, then judge only the complete llama dataset
in five-hour bouts. No code change: the committed `horeka_gemma_v4.py start`
at 3f32275 accepts one `--source`. Run root is
`$W/reviews/gemma-si-v4-llama-5h-20261004`. Diagnostic exclusions are unchanged.
Up to 200 in flight with ten-minute refills, as authorized earlier in the thread.

The locks are flocks (`$W/control/gemma-v4-sender.lock`, `<run>/start.lock`).
Killing the holder processes on hkn1990 and hkn1991 frees them; no files are
deleted. Only pending Gemma jobs are cancelled by name. Qwen is untouched.

Preparation runs as the usual one-hour GPU sbatch. A login-host preparation path
was drafted, blocked by the permission classifier and reverted, consistent with
AGENTS.md's rule to keep computation on Slurm. So the bouts start only after the
preparation job runs.

Launch refuses if the old run has `reservation.json`. Its HF claims would block
the llama plan, and no code path releases them. Estimate for ~312k llama cells
(half proxy): 318–374 well-filled bouts, about 4.5–5.2 days at 15 running nodes.
The frozen plan decides the actual budget. 30 runner/sender tests pass at 3f32275.
Codex/Claude executed no cluster commands. Next: get the user's output for the
lock check, launch and sender.log.
