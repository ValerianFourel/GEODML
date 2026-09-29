# Nemotron 20-cell review, 2026-09-29

## Evidence and scope

User-pasted SI-v3 run in job 5170116 on hkn0403, path
`runs/nemotron-si-v3-replay-job5170116-20260929-112750`, passed structurally:
29/29 SI and 5/5 J1 valid, 5/5 complete cells, 35.3 seconds of judging,
status passed and EXIT=0. Native xgrammar 0.2.7 accepted the schema. This is
pasted evidence, not current allocation status or proof of semantic accuracy.
Example a1 selected the topical title "Inventory control system"; whether that
supports the substantive recommendation remains questionable. All crypto
sources tied at 3, so top-source alignment there is weak evidence of agreement.

Valerian requested a new HTML page with 20 cells and all judgments for review,
and chose 10 fresh Qwen + 10 Llama. No inference code change was necessary.
Published SI-v3 pin remains `961ad0b181c70b9355db5b2366c0602d9abae675`.

## Operator page

Local page: `/Users/valerianfourel/Hamburg/GEODML_Unified/testnemotron20.html`.
Builder: `/private/tmp/build-nemotron20.py`. Neither is part of this checkout.
Login command checks the native schema and freezes ten verified completed
cells per model using seed 2026092910, excluding the original five cells.
It reuses existing Qwen $DS and Llama $W/llama-hf/dataset. Repeated setup checks
and reuses the frozen selection. It prints squeue for current remaining time.

Workspace-relative review root:
`reviews/nemotron-si-v3-review20-961ad0b-2026092910`.
Selection manifests bind identities, datasets and checksums. GPU command runs
existing launcher twice sequentially with --no-submit, reloading Nemotron
between generators. Guards require clean pinned code, no stopped/background
shell jobs, expected allocation name/time, 40 minutes left initially and
20 minutes before each startup. Existing run directories block repetition.

Full saved report includes all 20 slots, original prompts, complete recoverable
answers, every frozen source task, raw and resolved judgments, grades, retries,
errors and missing results. Files: full-review/header.json, cells.jsonl,
all-judgments.txt and cell-01.txt through cell-20.txt. Login step 3 prints two
cells at a time for review without truncation.

## Approval

Valerian explicitly answered "Approve existing-allocation test" for this one
20-cell test in an existing one-hour allocation with at least 40 minutes left.
Estimate: 15–35 minutes including two startups, one exclusive node, four A100s,
32 requested CPUs, all node memory, about 1–2.33 additional GPU-hours.
No new allocation, extension or repeat is approved. Enabled the page gate to
approved-qwen10-llama10. Insufficient time stops admission; do not replace the
allocation without approval. No cluster commands were executed by Codex.

## Verification and next action

Earlier local fixtures verified ten unique identities per model, exclusion,
frozen-selection reuse, twenty report slots, full long source/answer content,
retry/error retention and missing-result visibility. After enabling approval,
reparsed the actual saved HTML: all three blocks pass bash -n, embedded Python
passes ast.parse and copy-button JavaScript passes node --check.

Use login step 1, existing GPU shell step 2 once, then send full report batches
from step 3. No 20-cell results received yet. Review substantive source support,
topical-only matches, qualifications, zero grades, grade calibration and ties.
This is diagnostic review, not independent scientific ground truth. Preserve
production Qwen division and unrelated JUPITER work.
