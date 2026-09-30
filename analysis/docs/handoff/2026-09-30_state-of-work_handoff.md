# State of work — 2026-09-30, 04:32 CEST

Replaces the Shell A resume note and the 29 Sep SI-v1/Qwen note. Cluster facts
are pasted evidence from the times given; recheck before acting.

## 1. JUPITER (leaving; purge of `$FSCRATCH` starts 1 Oct)

| Item | State |
|---|---|
| Slurm jobs | none (`JOBS 0`); dispatcher lock free; no cron/scrontab |
| Llama | **done**: 310,374 / 310,380 cells published, 6 terminal failures, 0 unpublished attempts (`LLAMA_FULLY_DONE=yes`) |
| Qwen from JUPITER | 24,218 cells published (`coordination/qwen-results.json`) |
| Shell A archive → private `geodml-jupiter-archive-private` | **running**, tmux `hf-archive` on **jpbl-s02-01** (PID 3337146), started 04:26 after YES to the 74 LMSYS/WildChat files. 1/12 units verified; unit 2 (`fscratch-scifi`, 10 GB, 333k files) packing. `ARCHIVE_COMPLETE=no`. Rate ~0.3–5 MB/s. |
| Shell B → **public** `geodml-emnlp-2026-probing-activations` | **done**: 8,020 files, 551.7 GB, 0 missing (`ACTIVATIONS_UPLOAD_COMPLETE`, `UPLOAD_EXIT=0`) |
| `geoaxis-prompts-generation-26k` (private) | done 29 Sep: 272 files, 9.66 GB, contract hashes match |
| `geoaxis-readiness-axis` (private) | done 29 Sep: 953 files, 1.28 GB |
| `geodml-experiment-v2-paper-private` | 22,318 files, 73.33 GB (results, registry) |

Leftover tmux: `geodml-agentic-resume` on jpbl-s02-01 (since 13 Sep, idle); it
blocks `LEAVE_READY`. Other login nodes are not reachable by ssh; check each on
its own node. `$HOME` is over its inode quota: write nothing there.

**To leave, in order**
1. Watch Shell A until `ARCHIVE_COMPLETE=yes` (`jupiter-leave-status.html` block 1).
   If the window shows `RUN_EXIT` / Finished, rerun `finish-shell-a.html` block 2
   (resumes; verified units are skipped).
2. Close the idle `geodml-agentic-resume` session (check it runs nothing first).
3. `leave-jupiter.html` step 9 (remove stored tokens), then the checklist until
   `LEAVE_READY=yes`. Revoke the JUPITER HF token; make a separate HoreKa token first if shared.
4. Optional: GeoAxis restore checks; find the 1.5 TB `jutil` reports on `/e/scratch`
   (only 26 GB found); project1 cleanup only after Hub copies are verified.

## 2. HoreKa Qwen (workspace `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen`)

- 893 five-hour bouts (1 node × 4 A100) cover the remaining 287,878 cells. **All
  approved** (631–893 on 29 Sep). As of 29 Sep 04:30: 137 completed, 4 failed,
  3 running, 292 pending; ledger 47,909 cells completed.
- Sender: `horeka-qwen-send-all.html` → tmux `qwen-sender-all`
  (`qwen-bouts/send-all-bouts.sh`, every 10 min, ≤7 days, lowest sendable bout
  first, reads per-bout `SUBMISSION_ATTEMPTED`/`submission.json`; never resends).
  Confirm it is running and old `qwen-sender(-2)` loops are stopped.
- Limits: 295 queued+running per account, **50 running per user** (reached
  28 Sep 21:42). Priority = waiting age (max after 7 days) + QOS; fair-share
  weight is 0, so past use does not penalize. Pace is set by cluster load.
- Estimate for the rest: ~2,000–2,700 node-hours; 2–3 days at 50 nodes, weeks at 3–5.

## 3. Nemotron judge (HoreKa)

- Protocol **SI** (source importance): one request = request + full answer + one
  source; 0–5 importance with provenance; rankings/metrics in code; J1 separate;
  old J3 and claims-v3 out of the primary path. Plan: `~/.claude/plans/zesty-orbiting-magpie.md`.
- Judged answer = the **complete** generator answer: 35 % of Qwen answers were
  stored cut at 1,200 chars, 99.7 % recoverable from the trace (median 1,492 chars).
- Current pin **SI-v3 `961ad0b`** (passage IDs instead of copied quotes): 5/5
  cells, 29/29 judgments valid (job 5170116). v1/v2 failed on exact quotes.
  Quality not established: topical overlap may pass as support; ties (all 3s)
  weaken alignment. 20-cell review (10 Qwen + 10 Llama) prepared in
  `testnemotron20.html`; no result received yet. See
  `2026-09-29_nemotron-tomorrow_handoff.md`.
- Evidence: Shuffled is erased by reranking in ~95 % of pairs; ablation target
  shown in ~40 % (Parallel) / ~20 % (Reactive); 41 % of answers carry `(S1)` markers.
- Llama data on HoreKa: `$W/llama-hf/dataset` (pull started 28 Sep; 56.5 GB).
- Not built: production runner, results aggregation, reference judges
  (`deepseek-v4-flash-0731`, `glm-4.5-air`), validation selection, regime
  recovery for axis ≥ 0.70.
- Compute (estimate): Nemotron on everything ~150–450 node-hours once concurrency
  is measured; total with Qwen ~2,100–4,100 node-hours.

## 4. Code

Branch `threehour-relaunch-fix` → `origin/codex/pilot-continuation`. SI code:
`source_importance.py`, `agentic_cells.py`, `prepare_source_importance_tasks.py`,
`condition_manipulation.py`, `cluster_bootstrap.py`, `try_source_importance_judge.py`,
`horeka_nemotron.py --trial`. Archivers: `archive_jupiter_to_hub.py` (pin `062f06a`),
`archive_geoaxis_*`. Operator pages are on the Mac in `~/Hamburg/GEODML_Unified/`.
