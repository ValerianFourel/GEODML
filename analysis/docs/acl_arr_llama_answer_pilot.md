# Resumable Llama answer pilot

Run `analysis/scripts/slurm/jupiter/run_acl_arr_llama_answer_4gpu.sh` only
inside an explicitly approved existing Slurm job step. Keep the allocation-owning
shell open. The worker never requests, cancels, extends, or releases an allocation.

This milestone covers the 384 Llama answer tasks, not the other models, judges,
or the 26,009-prompt experiment. It verifies the completed rerank recovery and
leaves both the original pilot and recovery artifacts unchanged.

## Budget and protocol

The approved step is one hour on one node with four GPUs and 32 requested CPUs.
The submission deadline is 50 minutes after the worker starts, including module
and model loading. Each request has a five-minute timeout. The final ten minutes
are reserved for draining requests and shutting down this worker's server.

Use context length 41,984: the measured answer maximum requires 40,974 tokens,
including its 768-token output budget. Concurrency is eight. The original answer
prompt, schema, seed, temperature, output budget, and validator are unchanged.
There is no automatic repair decoding and no automatic HTTP retry.

The runtime estimate is 44–111 minutes for all 384 answers, including about ten
minutes of loading, assuming 256–768 generated tokens per answer and the observed
rerank aggregate rate of 48.86 tokens/second. Answer throughput is not yet measured;
the one-hour step may therefore produce only a checkpoint. Its cap is four GPU-hours.

## Outputs and continuation

Choose a new `ACL_ARR_ANSWER_ROOT` for every invocation. The worker creates its
`results` directory exclusively and refuses an existing one. Every returned
response is saved in `attempts.jsonl`; valid answers are also saved in
`outcomes.jsonl`, and invalid answers or request errors in `failures.jsonl`.
Raw invalid answers are retained without being repaired or counted as successes.
Rows are flushed and synced before the manifest is updated.

`results/answer_manifest.json` reports global completed and remaining counts.
All outputs are explicitly pilot-only and ineligible for scientific analysis.

- Exit 0: all answers completed.
- Exit 3: submission deadline reached; valid partial results are checkpointed.
- Exit 2: all tasks attempted but some answers failed validation or requests failed.
- Other errors: inspect the controller log and saved artifacts before retrying.

A separately approved continuation may set `ACL_ARR_ANSWER_RESUME_FROM` to the
previous results directory and use a new output root. Saved answers are validated
against the frozen tasks and copied before only the remaining tasks are submitted.
Do not delete the original failures, modify frozen inputs, or restart the old
whole-pilot controller to continue this isolated step.
