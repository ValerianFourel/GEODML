# Recover the first Llama pilot shard

Keep the allocation-owning shell open. Run the recovery worker with `bash` in
a step of the existing approved allocation. Never source the worker. Do not
rerun the full pilot controller for this recovery.

The approved scope is the 65 unresolved rerank tasks from the first Llama
pilot shard at plan commit `f9c35e499b708a4c812b49a3f550e097306ce25d`.
The source must contain 384 tasks and 319 validated successes. The recovery
reads the original artifacts and creates a separate output directory.

## Set the recovery limits

The observed maximum request is 40,456 tokens, including the unchanged
256-token output budget. Use context 40,960, tensor parallelism 4, GPU memory
utilization 0.90, and request concurrency 8. This context was measured for the
Llama rerank shard only. It does not validate other models or the answer phase.

The approved recovery step limit is 30 minutes in the existing allocation.
The estimated runtime is 15 to 30 minutes, based on the observed roughly
10-minute model load and 3m36s first shard. Four GPUs consume at most
2 GPU-hours during this step. The already-reserved allocation continues to
consume its allocation budget afterward unless Valerian explicitly releases
it or its existing deadline expires.

The launch must specify the existing `srun --jobid` and `--time=00:30:00`.
Do not request a new allocation or extend the original deadline. Require at
least 35 minutes remaining before launch, including a five-minute margin.

## Run the recovery worker

Use `analysis/scripts/slurm/jupiter/recover_acl_arr_llama_rerank_4gpu.sh` from
the exact committed recovery checkout. Set these environment variables:

- `ACL_ARR_RUN_ROOT`: original pilot directory.
- `ACL_ARR_VENV`: existing vLLM environment.
- `ACL_ARR_RECOVERY_ROOT`: fresh directory for controller and server logs.
- `ACL_ARR_RECOVERY_JOB_ID`: existing approved allocation ID.
- `ACL_ARR_RECOVERY_APPROVED_WALLTIME`: `00:30:00`.
- `ACL_ARR_RECOVERY_ESTIMATE`: the approved estimate and resource assumptions.
- `GEODML_RECOVERY_COMMIT`: exact recovery checkout commit.

The worker validates the source, installed vLLM arguments, and installed
XGrammar enum and prefix schemas before loading weights. It starts only the
pinned Llama model on loopback port 8001. Its cleanup stops only its private
server process group. The allocation remains open.

## Inspect the separate recovery artifacts

Read `results/recovery_manifest.json` under `ACL_ARR_RECOVERY_ROOT`.
Completion requires `status=complete`, `completed_count=384`, and
`remaining_count=0`. A zero-failure subset is not sufficient.

`results/outcomes.jsonl` starts with a byte-preserved copy of the 319 original
outcomes and appends recovered outcomes. Original results, failures, runtime
records, and logs remain unchanged. `results/attempts.jsonl` records every
recovery request, response, constraint, token usage, seed, and prompt hash.
Successful recovery outputs are flushed as they arrive. Failed and interrupted
attempts remain in their directory. Do not delete or blindly rerun that
directory; inspect it before authorizing another recovery.

## Keep the pilot out of confirmatory analysis

The recovery constrains IDs to those present in each task. When the model emits
duplicates, the next request fixes its unique prefix and excludes those IDs
from the remaining positions. It never invents or fills document IDs. Every
request preserves the original prompt, seed, temperature, and output budget.
The number of decoding attempts is bounded by the requested ranking length.

This is a decoding-protocol change. The recovery records both original-plan
and recovery-code commits, remains `scientific_result=false`, and remains
`eligible_for_analysis=false`, even when all 384 tasks have valid outputs.
Do not automatically merge it into the original pilot or the full experiment.

The backend does not support JSON Schema `uniqueItems`, so the recovery uses
explicit prefix constraints instead. See the
[vLLM 0.28 XGrammar schema checks](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/structured_output/backend_xgrammar/#has_xgrammar_unsupported_json_features).
