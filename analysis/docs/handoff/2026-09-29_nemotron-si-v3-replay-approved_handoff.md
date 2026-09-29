# SI-v3 existing-allocation replay approved, 2026-09-29

Valerian explicitly chose "Approve existing-allocation replay" for one SI-v3
replay of the original five cells, only after native schema acceptance and with
at least 25 minutes remaining in an existing one-hour allocation. Resources:
one exclusive node, four A100s, 32 requested CPUs, all node memory. Estimated
use: 10–25 minutes, approximately 0.7–1.7 additional GPU-hours. No new allocation,
extension or additional replay is approved.

Enabled that approval in root `testnemotron.html`, retaining the background-job,
commit, schema, allocation-name/time and 25-minute remaining-time checks. The
page continues to pin published SI-v3 implementation
`961ad0b181c70b9355db5b2366c0602d9abae675`. Run the login check, the node replay
once, then read results in the login shell. Commands use `--no-submit`; no new
allocation is created. Checked all three saved shell blocks with `bash -n`.

This records approval and operator-page preparation, not execution. No live
Slurm status, SI-v3 schema result, or SI-v3 inference result has been received.
Next: inspect the returned results; preserve the owning allocation shell. If
insufficient time remains, stop and report it without extending/replacing the job.
