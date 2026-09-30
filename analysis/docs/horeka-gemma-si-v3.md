# Gemma 4 31B: frozen SI-v3 comparison on HoreKa

The entry point is `analysis/scripts/horeka_gemma_si.py`. It never submits a job
or acquires an allocation. Use the exact commit containing this document in a
separate checkout; keep existing Qwen checkouts and environments intact.

## Download and remove the old model (login shell)

Valerian authorized removing Nemotron model files. Its pilot results and audits
remain in `reviews/`. Download first freezes a verified copy of the pilot inputs,
cell mappings and both passes' results in `preparation/gemma4-<revision>/`.
`--remove-nemotron` deletes only
`models/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` and invalidates that
model's old verification receipt. It refuses while a queued/running job name
contains `nemotron`; also ensure no custom-named job is using those files.
It never cancels jobs or removes other models, serving caches, or results.

From the pinned checkout, in an existing HoreKa login shell:

```bash
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  "$RT/bin/python" analysis/scripts/horeka_gemma_si.py download \
    --workspace "$W" --account hk-project-p0026831 \
    --baseline "$W/reviews/nemotron-si-v3-pilot20-961ad0b-2026093010" \
    --remove-nemotron
)
```

Model: `google/gemma-4-31B-it`, revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Two BF16 shards total
62,546,338,248 bytes. Existing downloader checks fresh quota, free bytes/inodes,
revision and every file checksum. Its conservative initial headroom requirement
is about 124 GB including the largest transient shard and safety margin.
Interrupted downloads resume; quota failure stops the download. Run only on the
login/transfer host, never on the Mac or offline compute nodes.

## Prepare the test after wall-time approval

Valerian approved the proposed one-hour allocation on 2026-09-30 in the reply
requesting an HTML command page. Use [the operator page](horeka-gemma-si-v3.html)
for admission, allocation, execution and results commands. Estimate: 15–45 minutes of
work, a **60-minute allocation** including startup/cleanup margin, one exclusive
Green node, four A100 40 GB GPUs, 32 requested CPUs, whole-node memory (~512 GiB).
Budget: one node-hour, four GPU-hours; the exclusive node reserves all 152 CPUs
(up to 152 CPU-hours), even though the command requests 32. This is an estimate:
Nemotron's cold first run took 11.6 minutes, subsequent starts/runs 6.6 minutes;
Gemma throughput is unmeasured. Gemma stays loaded across both passes. A
30-minute allocation is cheaper (two GPU-hours) but may end before both passes.
CPU-only preparation/download requires no GPU allocation. TP2 is not validated
by this repository's four-GPU boundary, so it is not offered as a tested profile.

After an explicit wall-time choice, from the pinned checkout:

```bash
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  GEMMA_WALLTIME='<APPROVED_WALLTIME>'
  GEMMA_APPROVAL='<EXACT_APPROVAL_AND_DATE>'
  GEMMA_RUN="$W/reviews/gemma4-si-v3-$(git rev-parse --short HEAD)"
  "$RT/bin/python" analysis/scripts/horeka_gemma_si.py prepare \
    --workspace "$W" --output "$GEMMA_RUN" \
    --walltime "$GEMMA_WALLTIME" --approval "$GEMMA_APPROVAL"
)
```

Preparation checks installed vLLM's Gemma4 registration without initializing CUDA,
records package versions, checks native SI schema support, and creates immutable
`config.json`, `inputs.json`, and `run.sh`. It does not upgrade the shared runtime.
If compatibility fails, retain the error and prepare an isolated compatible
environment before requesting compute. Static registration is not proof that
Gemma loads on A100; that remains part of the GPU test.

The HTML page checks fresh scheduling and quota evidence before allocation.
The allocation must match job name
`geodml-gemma-si-replay`, the approved time, whole-node exclusivity and the
resources above. Observe the current concurrency/start-gap rules. Start `run.sh`
on the compute host within that allocation; if the interactive shell stays on
the login host, use an explicit step with its existing job ID. Do not create or
release another allocation implicitly.

## What the test measures

Replays the saved 20 cells (10 Qwen, 10 Llama) twice using exactly the stored
task records, schemas, masked answers, full sources, seeds, SI-v3 0–5 scale and
J1. It verifies prompt/passage/schema hashes before requesting inference. It
does not repair the long Trello record, masking, rubric or source assignment.
The request setting remains `enable_thinking=false`, temperature 0, concurrency
4, 640 SI tokens and 64 J1 tokens. Serving remains BF16, TP4, context 73,728,
memory utilization 0.85 and eager mode; Gemma is text-only.

The existing authenticated loopback/offline stage runner owns server lifetime,
whole-node verification and cleanup before Slurm's end time. It does not release
the allocation. Every completed judgment and audit event is flushed. A deadline
leaves partial evidence; there is no automatic retry or resubmission. Inspect
and approve any continuation separately. This is the fixed historical pilot
exception, not a production backlog worker.

Outputs under `attempts/job<id>/trial/`:

- `pass1/` and `pass2/`: exact tasks, raw outputs, parsed/resolved passages,
  original response audits, cell grades/ranks/alignment, example and summary.
- `comparison.json`: Nemotron and Gemma scores joined by task ID; failed/missing
  Gemma judgments are null. Shared baseline tasks retain each observed score.
- `summary.json`: both pass statuses. Structural success does not prove accuracy.

Batch composition differs from the original separate Qwen/Llama server runs;
record this when interpreting nondeterminism. A better grade spread or higher
scores alone is not an improvement. Review passage matches and repeat stability,
especially cells `2dd070ba00e1` and `9407a920b4e1`, against the preserved evidence.

References checked 2026-09-30:
[model](https://huggingface.co/google/gemma-4-31B-it),
[vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html),
[HoreKa hardware](https://www.nhr.kit.edu/userdocs/horeka/hardware/),
[batch guide](https://www.nhr.kit.edu/userdocs/horeka/batch/).
