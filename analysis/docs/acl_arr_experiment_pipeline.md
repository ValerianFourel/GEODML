# Run the ACL ARR document experiment pipeline

This guide prepares and runs the Natural, Ablated, and Shuffled document
experiment. The pipeline handles 26,009 audited prompts, four model arms, two
primary pipelines, and one blinded judge phase.

The code does not start Slurm allocations or vLLM servers. Measure each frozen
model configuration in a GPU pilot before you request production wall time.

## Pipeline files

The implementation uses these files:

- `analysis/interpretability/pipeline/acl_arr_document_freeze.py` joins a frozen
  search snapshot to extracted page text.
- `analysis/interpretability/pipeline/acl_arr_document_experiment.py` creates
  condition assignments, task files, schemas, and blinded judge tasks.
- `analysis/interpretability/pipeline/acl_arr_document_analysis.py` checks
  paired coverage and computes first-pass intervention outcomes.
- `analysis/scripts/prepare_acl_arr_document_sets.py` writes immutable document
  sets.
- `analysis/scripts/prepare_acl_arr_experiment.py` writes the complete task
  plan.
- `analysis/scripts/run_acl_arr_vllm.py` runs primary or judge tasks through a
  vLLM OpenAI-compatible server.
- `analysis/scripts/prepare_acl_arr_judge_tasks.py` separates public judge inputs
  from the private condition and generator mapping.
- `analysis/scripts/analyze_acl_arr_experiment.py` joins complete paired outputs.

## Required inputs

Prepare these inputs before model inference:

1. `compliant-candidates.jsonl` from the passing 26,009-prompt audit.
2. `final-axis-map.jsonl` from the same audit.
3. A frozen SERP in Parquet or JSONL format.
4. Extracted page text keyed by exact URL.
5. A model configuration with four exact model IDs and 40-character revision
   SHAs.

The model template is
`analysis/config/acl_arr_model_panel.template.json`. Copy it to a run directory
and replace every revision placeholder. The approved dense 72B arm is
`Qwen/Qwen2.5-72B-Instruct`. This model replaces the unavailable Qwen3.8-72B
model. Record the substitution in the run manifest and paper.

## Freeze the document sets

Run `prepare_acl_arr_document_sets.py` after the search snapshot and extracted
page text are complete:

```bash
python3 analysis/scripts/prepare_acl_arr_document_sets.py \
  --serp "$SEARCH_SNAPSHOT" \
  --page-text "$EXTRACTED_PAGE_TEXT" \
  --minimum-documents 11 \
  --maximum-documents 20 \
  --max-document-characters 12000 \
  --source-git-commit "$GIT_COMMIT" \
  --output-dir "$RUN_ROOT/document-freeze"
```

The default path requires extracted page text. It drops URLs without extracted
text and fails if any keyword retains fewer than 11 documents. Do not add
`--allow-snippet-fallback` to the production run unless the protocol explicitly
defines a mixed full-text and snippet experiment.

The command writes:

```text
document-freeze/
  document_freeze_manifest.json
  frozen_document_sets.jsonl
```

Each document record contains the original search position, the URL, the
extracted text, the full-text hash, the used-text hash, and the truncation
status. The preparation step does not fetch a live page.

## Prepare the complete task plan

Create the paired task plan after you freeze all model revisions:

```bash
python3 analysis/scripts/prepare_acl_arr_experiment.py \
  --prompts-jsonl "$AUDIT_ROOT/compliant-candidates.jsonl" \
  --axis-map-jsonl "$AUDIT_ROOT/final-axis-map.jsonl" \
  --document-sets-jsonl "$RUN_ROOT/document-freeze/frozen_document_sets.jsonl" \
  --models-json "$RUN_ROOT/models.json" \
  --top-n 10 \
  --master-seed 20260904 \
  --expected-prompt-count 26009 \
  --expected-model-count 4 \
  --source-git-commit "$GIT_COMMIT" \
  --output-dir "$RUN_ROOT/plan"
```

The plan contains one condition assignment per prompt. The assignment is shared
by every model and both primary pipelines.

- Natural preserves the frozen search order.
- Ablated removes one target document.
- Shuffled applies a cyclic derangement with no document left in its original
  position.

The ablation target positions and shuffle offsets differ by at most one count
within each document-count group. Stable hashes make both assignments
reproducible.

The exact 26,009-prompt workload is:

| Unit | Count |
| --- | ---: |
| Tasks per model and primary pipeline | 78,027 |
| Tasks per primary pipeline | 312,108 |
| Reranking and answer tasks | 624,216 |
| Planned judge tasks | 312,108 |
| Total inference requests before retries | 936,324 |

The plan writes eight primary task files. Each model has one reranking file and
one answer file.

## Run a CPU plumbing check

Use `--fake` on a small plan before a GPU pilot:

```bash
python3 analysis/scripts/run_acl_arr_vllm.py primary \
  --tasks "$TASK_FILE" \
  --plan-manifest "$RUN_ROOT/plan/run_manifest.json" \
  --output-dir "$RUN_ROOT/smoke/rerank" \
  --max-tasks 12 \
  --fake
```

Fake outputs test joins, schemas, checkpointing, and file creation. Every fake
manifest has `scientific_result=false` and `eligible_for_analysis=false`.

## Run one primary vLLM task file

Start one vLLM server for the exact model and revision recorded in the task
file. Then run one task file:

```bash
python3 analysis/scripts/run_acl_arr_vllm.py primary \
  --tasks "$TASK_FILE" \
  --plan-manifest "$RUN_ROOT/plan/run_manifest.json" \
  --base-url http://127.0.0.1:8000/v1 \
  --server-model-name "$SERVED_MODEL_NAME" \
  --server-model-revision "$MODEL_REVISION" \
  --max-concurrency 32 \
  --request-timeout 600 \
  --max-attempts 3 \
  --resume \
  --output-dir "$MODEL_PIPELINE_OUTPUT"
```

The runner checks `/v1/models` before it sends a task. vLLM performs continuous
batching. The client holds at most four concurrency windows in memory.

The runner writes:

```text
MODEL_PIPELINE_OUTPUT/
  outcomes.jsonl
  failures.jsonl
  run_manifest.json
```

The parser rejects unknown IDs, duplicate IDs, missing citations, citation-list
mismatches, invalid JSON, and wrong output sizes. It does not replace a failed
ranking with the source order. Use `--resume` to skip valid outcomes and retry
the remaining tasks.

## Prepare blinded judge tasks

After all four answer files pass their count checks, create one judge task per
answer:

```bash
python3 analysis/scripts/prepare_acl_arr_judge_tasks.py \
  --answer-outcomes "$MODEL_1_ANSWERS" \
  --answer-outcomes "$MODEL_2_ANSWERS" \
  --answer-outcomes "$MODEL_3_ANSWERS" \
  --answer-outcomes "$MODEL_4_ANSWERS" \
  --plan-manifest "$RUN_ROOT/plan/run_manifest.json" \
  --judge-model-id "$JUDGE_MODEL_ID" \
  --judge-model-revision "$JUDGE_MODEL_REVISION" \
  --master-seed 20260905 \
  --output-dir "$RUN_ROOT/judge-plan"
```

The public judge task file excludes generator and condition labels. The private
mapping retains those labels for analysis. Do not expose
`private_judge_mapping.jsonl` to the judge server.

The judge receives the documents in an independently deranged order. This step
prevents the generator's input order from becoming the judge's document order.

## Run the judge

Run the frozen judge with the same resumable client:

```bash
python3 analysis/scripts/run_acl_arr_vllm.py judge \
  --tasks "$RUN_ROOT/judge-plan/judge_tasks.jsonl" \
  --judge-manifest "$RUN_ROOT/judge-plan/judge_manifest.json" \
  --base-url http://127.0.0.1:8000/v1 \
  --server-model-name "$JUDGE_SERVER_MODEL_NAME" \
  --server-model-revision "$JUDGE_MODEL_REVISION" \
  --max-concurrency 32 \
  --request-timeout 600 \
  --max-attempts 3 \
  --resume \
  --output-dir "$RUN_ROOT/judge-results"
```

Calibrate the judge against a human-labeled sample before you treat judge scores
as paper outcomes. Do not use a model's self-judgment as the only primary score
for that model.

## Build the paired analysis table

Run the analysis only after every expected task has a valid outcome:

```bash
python3 analysis/scripts/analyze_acl_arr_experiment.py \
  --plan-manifest "$RUN_ROOT/plan/run_manifest.json" \
  --rerank-outcomes "$MODEL_1_RERANK" \
  --rerank-outcomes "$MODEL_2_RERANK" \
  --rerank-outcomes "$MODEL_3_RERANK" \
  --rerank-outcomes "$MODEL_4_RERANK" \
  --answer-outcomes "$MODEL_1_ANSWERS" \
  --answer-outcomes "$MODEL_2_ANSWERS" \
  --answer-outcomes "$MODEL_3_ANSWERS" \
  --answer-outcomes "$MODEL_4_ANSWERS" \
  --judge-outcomes "$RUN_ROOT/judge-results/outcomes.jsonl" \
  --private-judge-mapping "$RUN_ROOT/judge-plan/private_judge_mapping.jsonl" \
  --output-dir "$RUN_ROOT/analysis"
```

The analysis fails unless all prompt, model, condition, pipeline, and judge
cells are complete. It writes one paired row per prompt and model. The first
version includes ranking overlap, common-item Kendall agreement, citation
overlap, target-citation indicators, realized-use overlap, and judge-score
deltas.

## Interpret the output

Use the assigned readiness coordinate as the prompt variable. Use the consensus
axis coordinate as a measurement field or stratification field. Do not call the
consensus coordinate a confounder.

Randomized document order identifies position effects under the assigned
shuffle policy. Randomized target removal identifies the effect of document
availability under the assigned ablation policy. Page-feature coefficients
remain observational unless the experiment changes the page content or feature.
