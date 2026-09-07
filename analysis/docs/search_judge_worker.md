# Resumable quote-judge worker

`analysis/scripts/slurm/jupiter/run_search_quote_judge.sh` runs inside an existing
four-GPU Slurm step. It never creates, extends, or cancels an allocation.

The runner's new `--preflight-only` option validates frozen inputs and saved raw
outputs through the same resume validator used during execution. It returns 0
for complete, 3 for pending, and fails on invalid inputs. It neither contacts a
server nor writes results. Use `--resume` to check an existing output directory.

The worker checks completion before loading a tokenizer or model. Complete runs
return success. Partial runs require `SEARCH_ALLOW_RESUME=1`, after reviewing
retry authority. Original journals are retained. Changed judge protocols must
use separate result directories; this worker only uses quote-only v2.

Required exported variables:

- `GEODML_EXECUTION_COMMIT`, `GEODML_EXPECTED_JOB_ID`
- `ACL_ARR_VENV`, `ACL_ARR_RUN_ROOT`, `GEODML_CACHE_ROOT`
- `SEARCH_BUNDLE_DIR`, `SEARCH_PRIMARY_OUTPUT`, `SEARCH_JUDGE_OUTPUT`
- `SEARCH_JUDGE_MODEL`, `SEARCH_JUDGE_REVISION`

The model/revision pair must match a snapshot lock. The worker checks the actual
chat-template token counts with a 49,152-token serving context, including output
allowances. It uses TP=4, BF16, memory utilization 0.90, prefix caching, xgrammar,
port 8010, concurrency 8, and fscratch compiler caches. These settings reuse the
pilot launch; they are not measured optima for every arm. Unsupported context
sizes fail rather than changing RoPE, evidence, or decoding policy.

Unique sibling server logs record the checkout commit and allocation. Server
startup is bounded to roughly 15 minutes. Cleanup targets only the server process
group started by the worker. The enclosing Slurm step supplies the total time
limit. A killed step retains the runner's existing durable journals; inspect and
explicitly authorize a partial resume before continuing.

## Evidence and limitations

The user-reported Qwen v2 run at commit e2944e8 completed 9/9 judgments with no
validation failures. That establishes output-contract success, not factual
correctness. The new worker has CPU branch tests and shell syntax validation;
its complete GPU path still needs a cluster run. No additional inference was
performed during this change.

Use the existing `inspect` command with `--bundle-dir`, `--primary-output`,
`--judge-output`, and a new `--output-dir` to export `report.json`. Review actual
claim support and citation coverage against the frozen evidence. A resolvable
quote can still fail to support a claim. Preserve the independent failed Llama
ranking; neither judging nor human inspection makes that ranking valid.
