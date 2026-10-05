# Qwen bouts: identical repeated failures no longer kill the worker

Qwen recovery bouts 2 and 6 (sweep of 2026-10-03/04, pin 8f60b25) died with
`duplicate record ID in active shard: failed-...`. Cause: the dataset-mode failure path
in `run_agentic_search_integration_smoke.py` built the failed-attempt record ID from
(transaction, LLM-call count, trace hash); a cell that fails identically on its bounded
retry produces the same ID, and `JsonlShardWriter.append` refuses duplicates, which
raised out of the worker. The file-mode path already kept only the first identical trace.

Fix (branch `codex/qwen-failed-attempt-idempotent-20261006`, from 8f60b25):
`FinalDatasetWriter.has_active_record(table, record_id)`; the failure path skips the
append when that exact record is already in the open shard. Schema, record-ID format and
the duplicate guard itself are unchanged; the cell still becomes a terminal failure after
its bounded retry. Regression test reproduces the production error on the old code
(`test_identical_repeated_failure_in_dataset_mode_keeps_worker_alive`) and passes with the
fix; 96 tests in the dataset, backlog, smoke, Qwen bout and recovery suites pass.

Pasted state 2026-10-05 15:00 UTC: HF repo has 310,623 Qwen cells of 312,108 (1,485
missing); HoreKa CURRENT division 287,878 cells with 327 unattempted, 75 checkpointed,
54 running. A second recovery sweep covers only that division (~456 cells); the ~1,000
cells outside it need a separate route. Running bouts keep their old pin.
Next: push, then launch a second recovery sweep pinned at this commit (5 h GPU walltime,
explicitly requested by Valerian 2026-10-06), `--previous-recovery` = the 2026-10-03 sweep.
