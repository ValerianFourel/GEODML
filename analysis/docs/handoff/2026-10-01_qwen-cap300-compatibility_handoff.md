# Recognize the older Qwen CAP default, 1 October 2026

Valerian inspected the actual shared-workspace `send-what-fits.sh` and returned
`CAP=${CAP:-300}`. Added that exact assignment to the queue updater's recognized
inputs in `analysis/docs/horeka-si-v4.html` and its workspace-root copy. The
replacement remains literal `CAP=294`; backups, preflight validation, atomic
replacement and existing job preservation are unchanged.

The local fixture now uses the observed default of 300 for that sender. It
reproduced the same refusal before the edit and passed afterward. All three
sender fixtures stop at 294 despite inherited CAP=295 and refill one slot at
293. Backup, permission, idempotence, unknown-setting and symlink checks pass,
as do HTML command syntax, unchanged other commands and identical page copies.
`git diff --check` passes. Proof: `/private/tmp/verify-qwen-counts-cap294.py`.

No remote mutation or allocation was executed here. The pasted CAP assignment
is the latest evidence; the actual queue and sender state remain unverified.
Next: refresh the page and rerun step 1 in HoreKa, then repeat step 2 admission
when the account queue falls below 295. Preserve all queued/running jobs.
