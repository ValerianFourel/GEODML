# Run the captured-evidence pilot

Read [the design and scientific boundaries](search_experience_design.md) before
using this separately versioned pilot. It does not run live search or allocate
GPUs. Use one active model server per four-GH200 node. Repeat primary execution
for each exact model configuration in the frozen plan.

## Check the implementation without inference

From the repository root, run:

```bash
python3 -m unittest analysis.tests.test_search_experience analysis.tests.test_run_search_experience analysis.tests.test_acl_arr_document_experiment -v
python3 analysis/scripts/run_search_experience.py --help
```

The tests use synthetic fixtures and an injected client. Their outputs are not
real search captures, generated answers or judge measurements.

## Supply real captures

Use the existing frozen plan and select three existing prompt IDs: an ordinary
request, a long-evidence request and a request whose evidence is insufficient.
Keep all three conditions for each selected prompt. Do not rebuild the original
assignments on this subset.

Supply the collector's JSONL checkpoint, not just the flattened SERP parquet.
Checkpoint objects contain `keyword`, `query` and `raw_results`. Each result
contains `position`, `title`, `url` and `snippet`. Existing optional query
parameters, timestamps, page text, hashes and fetch errors remain in the trace.
When page text or its hash is supplied, it must match the frozen model evidence.
Missing acquisition metadata is not reconstructed.

Preparation defaults to `--query-contract metadata-keyword-v1`.
`--query-contract full-request-v1` accepts a previously captured full-request
search only when it also matches the frozen plan's search query and evidence.
It does not execute or rewrite a query. A different evidence pool requires a
separately prepared experiment; relabeling an old keyword capture is rejected.

Do not invent records to satisfy this input. The adapter can check consistency
with the frozen plan, but cannot authenticate a search provider from metadata.
Keep the original capture files and source plan artifacts available unchanged.

After setting paths and selecting the three real IDs, run:

```bash
python3 analysis/scripts/run_search_experience.py prepare \
  --plan-manifest "$ACL_ARR_RUN_ROOT/plan/run_manifest.json" \
  --captures-jsonl "$SEARCH_CAPTURE_JSONL" \
  --prompt-id "$ORDINARY_PROMPT_ID" \
  --prompt-id "$LONG_PROMPT_ID" \
  --prompt-id "$INSUFFICIENT_PROMPT_ID" \
  --output-dir "$SEARCH_PILOT_ROOT/bundle"
```

## Run independent rankings and answers

Run the following only on the allocated compute node with its pinned model
server already running. Verify the server's tokenizer, chat template and context
capacity for these new requests, including their output allowance. Do not reuse
the old 32,768-token limit or shorten evidence to fit it.

New four-model plans use a common 2,048-token answer allowance. When running an
older frozen bundle, pass `--answer-max-tokens 2048` and use a fresh output
directory. The override is part of request identity, so a 768-token output
directory cannot be resumed under the new allowance. Historical artifacts stay
unchanged.

The pinned Qwen2.5-72B configuration advertises 32,768 tokens by default. For a
request above that limit, set the vLLM rope-scaling override explicitly to
`{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}`.
The worker passes it through vLLM's `--hf-overrides` interface. The serving
profile records and hashes the exact override. Do not apply it to the other
panel models or use it to bypass the measured request budget. This follows the
[Qwen2.5-72B model card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct#processing-long-texts),
which also warns that static YaRN can affect shorter inputs.

```bash
python3 analysis/scripts/run_search_experience.py run-primary \
  --bundle-dir "$SEARCH_PILOT_ROOT/bundle" \
  --model-configuration-id "$MODEL_CONFIGURATION_ID" \
  --server-model-revision "$MODEL_REVISION" \
  --base-url http://127.0.0.1:8000/v1 \
  --answer-max-tokens 2048 \
  --max-concurrency 8 --resume \
  --output-dir "$SEARCH_PILOT_ROOT/primary-answer2048-v1/$MODEL_CONFIGURATION_ID"
```

The output directory is specific to one frozen model configuration. Ranking
does not filter answer evidence. Concurrency eight is a starting setting, not a
measured optimum. Three prompt IDs yield 18 primary requests per model before
retries. Invalid outputs stay failures; resume never treats them as successes.

## Judge and export the human sample

After primary execution, use a different pinned judge model. Run the judge
server as a separate stage on the same allocation, not alongside generator
weights. The judge contract has its own 2,048-token output allowance.

```bash
python3 analysis/scripts/run_search_experience.py run-judge \
  --bundle-dir "$SEARCH_PILOT_ROOT/bundle" \
  --primary-output "$SEARCH_PILOT_ROOT/primary/$MODEL_CONFIGURATION_ID" \
  --judge-model-id "$JUDGE_MODEL_ID" \
  --judge-model-revision "$JUDGE_MODEL_REVISION" \
  --base-url http://127.0.0.1:8000/v1 \
  --max-concurrency 8 --resume \
  --output-dir "$SEARCH_PILOT_ROOT/judge/$MODEL_CONFIGURATION_ID"
python3 analysis/scripts/run_search_experience.py export-human \
  --bundle-dir "$SEARCH_PILOT_ROOT/bundle" \
  --primary-output "$SEARCH_PILOT_ROOT/primary/$MODEL_CONFIGURATION_ID" \
  --sample-size 9 --seed 20260907 \
  --output-dir "$SEARCH_PILOT_ROOT/human/$MODEL_CONFIGURATION_ID"
python3 analysis/scripts/run_search_experience.py inspect \
  --bundle-dir "$SEARCH_PILOT_ROOT/bundle" \
  --primary-output "$SEARCH_PILOT_ROOT/primary/$MODEL_CONFIGURATION_ID" \
  --judge-output "$SEARCH_PILOT_ROOT/judge/$MODEL_CONFIGURATION_ID" \
  --output-dir "$SEARCH_PILOT_ROOT/report/$MODEL_CONFIGURATION_ID"
```

Give human reviewers `human_packets.jsonl`, not `private_mapping.jsonl`.
Review claim completeness as well as support: text in the uncertainty field
could itself contain an unsupported assertion. No automatic human-agreement or
quality claim follows from exporting packets.

Inspect `report.json` for the original request, capture trace, frozen evidence,
ranking, cited answer and judgment. A missing or invalid stage keeps its cell
incomplete. Preserve old reports and choose a new report directory after resume.

## Current execution boundary

These commands require real captures, selected prompt IDs, a committed checkout
on the cluster and verified model servers. They are not a six-hour allocation
command. No new allocation or full production run is authorized by this guide.
Do not close or relinquish an existing allocation-owning shell.
