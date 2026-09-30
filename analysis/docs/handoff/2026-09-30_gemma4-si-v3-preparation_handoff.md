# Gemma 4 SI-v3 comparison preparation

Valerian requested removing Nemotron model files, downloading Gemma 4 31B on
HoreKa and testing SI-v3. Removal was explicitly clarified in chat. No GPU
allocation or wall-time is approved. No SSH, download, allocation, inference,
model deletion or environment change was executed on HoreKa.

Prepared `horeka_gemma_si.py` (download/removal and allocation-free preparation),
`replay_source_importance_judge.py` (two passes over exact frozen task records),
and operator guide `analysis/docs/horeka-gemma-si-v3.md`. Reused the existing
quota/checksum downloader, authenticated stage/server lifecycle, SI validation,
result writer and metrics. Extended `horeka_nemotron.stage_commands` to respect
the frozen judge model/revision and route a frozen replay; legacy defaults stay.
Extracted the existing trial's freezing/execution boundary for saved-task replay.

Gemma model pin: google/gemma-4-31B-it at
842da3794eaa0b77d5f08bae87a17459d91ff475. Official HF inventory reports two BF16
shards totaling 62,546,338,248 bytes. Largest shard is 49,784,788,364 bytes;
existing conservative downloader requires ~124 GB initial headroom. Reuse the
current runtime only if static Gemma4 registration/native schema checks pass;
full A100 support is not yet tested. Never upgrade shared Qwen runtime blindly.

Baseline: `$W/reviews/nemotron-si-v3-pilot20-961ad0b-2026093010`. Freeze reads
four existing trial directories, verifies task hashes and that passes share the
same inputs/mappings, preserves results, refuses ambiguous/missing attempts.
Removal targets only Nemotron's HF model-cache directory and its stale verified
receipt, refusing symlink redirection or active/queued Nemotron-named jobs.
Custom-named model consumers must also be ruled out by the operator.

Comparison preserves 0–5 scale, SI-v3 rubric, masks, sources, schemas, task seeds,
temperature 0, thinking off, output caps, TP4/BF16/context/concurrency. New judge
results are separate. Two combined-corpus passes share one loaded Gemma server;
batch composition differs from baseline separate corpus runs. Null failures,
stable task-ID joins, both raw and resolved results. No claim of quality gain.

Validation: 47 focused CPU tests passed via Miniconda Python / pytest:
`test_horeka_gemma_si.py`, `test_horeka_nemotron.py`,
`test_source_importance_pipeline.py`. New tests exercise full frozen replay with
scripted transport, tamper/mapping rejection, comparison nulls, preparation without
allocation, and bounded deletion preserving results/other models. They establish
software behavior only. No live cluster facts have been rechecked; prior status
is historical. Git diff whitespace check passed.

Next: commit/push exact code, operator runs pinned-checkout login download command.
Proposed GPU work 15–45 min; request 60 min for startup/cleanup uncertainty, one
exclusive Green node, 4 A100 40GB, 32 requested CPUs, ~512GiB whole-node memory.
Max 4 GPU-hours / one node-hour (exclusive 152 CPUs reserved). Cheaper 30 min
risks incomplete passes. Ask for explicit wall-time approval before supplying
allocation commands. Recheck scheduling limits/start gap and quota. Preserve live
allocations. Run the generated run.sh in the approved compute allocation, inspect
raw Trello/unstable-cell evidence and repeatability before discussing replacement.
