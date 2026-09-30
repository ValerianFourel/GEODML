# Fresh Gemma results and private Hub export

Valerian returned full fresh20 summaries from job 5171481 and requested commands
to upload all Gemma judgments to Hugging Face, then have Codex read them and
report timing/error statistics. Upload and readback are explicitly authorized.

## Pasted evidence

Fresh run code 135d4685f232776d2e8157e93e41657f3b062ee9, seed 2026093011,
unchanged Gemma/SI-v3 settings. Each pass completed 20 cells, 20 J1 and 108 SI
judgments. All 128 requests per pass successful, all stop, no failure categories
or execution error; both statuses passed. GEMMA_EXIT=0. Pass seconds 67.1 and
61.6, total 128.7 seconds for 256 judgments: 1.989/s, approximately 7,161/hour
while judging. This excludes startup and does not predict production throughput.
Median SI request times 1.8843 and 1.8602 seconds; median J1 .59349 and .58324.
Median SI completion tokens 33, maximum 121; J1 median/maximum 9. Six stored
answers were truncated, with the unchanged full-answer recovery protocol.

Three visible source-grade changes: OKR cell dd7e839f1c21, PeopleManagingPeople
3 to 2; Trello fe8032d957c9, DigitalProjectManager Microsoft alternatives 1 to 2;
online pharmacy 7d70633d5770, Beckers 3 to 4. Thus 105/108 identical SI grades,
17/20 identical grade vectors; all 20 J1 unchanged. Histograms 0:41,1:26,2:30,
3:9,4:2 then 0:41,1:25,2:32,3:7,4:3. These are diagnostic, not validated science.

Printed Monzo example matches a3 about more frequent spending tracking to an
elliptical passage saying Monzo/Revolut/Starling help track. It supports the
tracking topic but does not establish increased frequency or the interface
causal explanation. Need assess the full support rubric and other raw cases.
All-zero training, Confluence pricing and broker-comparison cells still need
the full inputs; J1=5 alone does not prove source support.

Later Slurm message says job 5171481 reached time limit and killed interactive
step at 2026-09-30T11:30:07.007 in the displayed time base. This follows successful
judging. Do not conflate allocation TIMEOUT with inference failure or infer
elapsed job time from mixed timestamp bases. Slurm accounting is needed.
Earlier preparation/FileNotFound/reference errors and user-cancelled pending
5171480 are known operational history, not fresh-pass failed judgments.

## Export preparation

Code pin dcd8588b4a3daee31362275167684218e6e607b0 adds
analysis/scripts/export_gemma_si_review.py. Uses explicit diagnostic file
allowlist and packages both requested run directories, every job attempt and
both passes: config/inputs, full tasks/results/cells/audit/example/summary,
comparison, stage/boundary/runtime/profile/telemetry, console and server logs.
No credentials, environment files, model caches or shared claim mutations.
Missing expected artifacts are listed rather than called complete. Read-only
sacct captures job/step state, exits, timestamps, resources and memory when
available; accounting errors are preserved. JSON report includes pass summaries,
actual result counts, failures, retry-category counts, latency mean/median/max
and judging throughput. Raw logs remain available for independent review.

ZIP includes per-file SHA256/size manifest. Content-addressed upload to
reviews/gemma-si-v3/exports/<sha256>.zip in existing private
ValerianFourel/geodml-experiment-v2-paper-private, using existing HubStore.
One CAS commit; no registry changes. Readback at exact commit must match bytes
before HF_EXPORT receipt is printed. Local ZIP/report/receipt saved under
$W/reviews/gemma-si-v3-exports. Report is evidence, not scientific validation.

14 focused export/Gemma tests passed. Tests cover byte-preserving ZIP and manifest,
latency/rate reporting, missing-pass honesty, credential-file exclusion, symlink
refusal, narrow remote write and failed readback rejection. One HTML copy block
passes bash -n; both page copies match; diff check passed. Page now shows upload
instead of old inference commands. No remote export executed by Codex.

Local Miniconda has huggingface_hub and cached authentication, checked without
printing credentials. Initial repo_info failed sandbox network resolution;
the escalated retry was interrupted by user before completion. Remote read
access is not yet confirmed. User then said continue. Never request HF tokens
in chat. Use local saved auth for eventual readback with needed network approval.

Next: user runs page's login-shell upload block and returns HF_EXPORT. Codex then
downloads that exact revision/path, verifies archive and per-file hashes, reads
raw judgments and logs, reports full timing/error findings. Do not claim upload
or full raw-output review completed before that receipt/data arrives.
