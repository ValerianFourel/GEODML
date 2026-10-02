# Gemma whitespace truncation and semantic-feedback repair

Returned evidence from job 5175121 identifies 17 failed tasks: 10 answer maps
and 4 source judgments end at 4096 tokens with repetitive whitespace tails;
one final map failure leaves answer words uncovered; two final source failures
have positive grades inconsistent with support findings. One truncated map also
had a semantic error on its first attempt. All 20 fulfilment tasks, 9 maps and
55 source judgments succeeded. The 52 blocked source dependencies follow failed
maps. Preserve these 84 successes and 69 unresolved logical tasks. No transport
failure is shown. Current allocation state/remaining time remains unknown.

Read the last three indexed handoffs. Applied surgical-patch and existing
HoreKa/test-audit guidance. Traced the exact request/schema, server profile and
validation-feedback code. Confirmed vLLM v0.28.0 supports
`structured_outputs_config={"backend":"xgrammar","disable_any_whitespace":true}`
and forwards it to xgrammar compile_json_schema(any_whitespace=False). Sources:
https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/structured_outputs.py
https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/structured_output/backend_xgrammar.py
The existing stage/profile infrastructure already supports this setting; no new
client flags, dependencies or serving framework were introduced. This constrains
JSON formatting whitespace, not whitespace inside string values. Raw tails alone
do not prove all 14 failures arose outside strings. Model-level effectiveness
still requires cluster validation; do not claim the run has been recovered.

Published helper patch on codex/gemma-selected-cells:
`b3cab6e52cce89dbefaa347faaaf246f67fe9d6f`.
A subsequent docs commit carries these updated instructions without changing code.
Release checkout `.worktrees/gemma-selected-cells-release` was checked against
active HEAD before transferring the six changed files, avoiding unrelated history.

Changes:
- horeka_si_v4 selected-cell preparation records compact structured decoding in
  run and judge configurations; the latter is included in execution identity.
- horeka_nemotron forwards an explicitly recorded SI-v4 setting through the
  existing --structured-outputs-config stage/profile interface. Historical saved
  configurations retain their defaults.
- run_source_importance_judge accepts/validates the optional compact-xgrammar
  configuration. It does not change task completion or truncation acceptance.
- source_importance_v4 validation errors now include up to 24 missing word IDs
  and total missing count, or the actual grade/support conflict and correction
  rule. This improves the existing second-attempt feedback. It neither creates
  missing claims nor changes grades. Acceptance rules, rubric, prompt templates,
  max_tokens=4096, temperatures, retry count and model revision remain unchanged.
  Changed feedback is code-versioned; do not merge new results as unchanged runs.

Five targeted assertions failed on the pre-fix code for missing server/config
settings and uninformative corrective feedback. After repair, 124 tests and
3 subtests pass in both active and release checkouts:
`PYTHONDONTWRITEBYTECODE=1 /Users/valerianfourel/miniconda3/bin/python -m pytest -q analysis/tests/test_horeka_si_v4.py analysis/tests/test_horeka_gemma_si.py analysis/tests/test_source_importance_v4.py analysis/tests/test_search_vllm_stage.py`
Proof covers native CLI forwarding, recorded/loaded configuration, actual
_execute_one corrective feedback, unchanged rejection rules and preserved legacy
workflow. Existing truncation and terminal-failure non-retry tests remain intact.
These are CPU/fixture tests; no vLLM server or GPU inference ran locally.

Updated horeka-gemma-failures.html with confirmed diagnosis, published repair,
limitations and copyable read-only allocation status. Updated selection page pin
and prominently direct this partial run to recovery diagnosis instead of fresh.
Matching diagnostic/selection exports are in Downloads and workspace root.
Bash/Python/JavaScript syntax checked. No browser rendering/copy proof claimed.

No results, ledger, lock files or saved cluster preparation were modified. No
cluster command, inference, allocation, retry, cancellation or extension ran.
An existing saved run.sh/config still pins its original revision. Fetching the
new helper does not rewrite it. Do not rerun fresh blindly over these same cells.

Next: Valerian returns squeue/sacct state for existing Gemma jobs. A separate
failed-only recovery milestone must reconcile and retain successful artifacts,
record the decoding/feedback revision and select only unresolved work before
execution. The current helper does not implement terminal-failure re-admission;
never reset terminal_failed/done rows or weaken the attempt guard as a shortcut.
Any new allocation requires measured remaining-work estimate and fresh wall-time
approval. The current task delivered a tested code correction, not a recovered
scientific result.
