# Nemotron SI-v1 judge and the Qwen send-all sender — handoff 2026-09-29

Continues [2026-09-28_horeka-qwen-sender_handoff.md](2026-09-28_horeka-qwen-sender_handoff.md).
Cluster facts below are pasted evidence from 29 Sep; recheck before acting.

## Code state (branch `threehour-relaunch-fix`, pushed to `origin/codex/pilot-continuation`)

| Commit | What |
|---|---|
| `ddf8371` | SI-v1 core: `source_importance.py` (units, citation masking, one-answer x one-source prompt, 0-5 schema with exact quotes, validator, rank groups, tau-b, cell metrics); runner accepts `si-identical-retry-v1` |
| `b1d3634` | `agentic_cells.py`, `prepare_source_importance_tasks.py`, `condition_manipulation.py` + `audit_condition_manipulation.py`, `cluster_bootstrap.py` |
| `ed3ddaf` | Judge the complete answer: recovered from the trace when stored truncated; `truncated`/`answer_source`/lengths per cell; 10 % stored-answer sensitivity subset |
| `55b7a4a` | `try_source_importance_judge.py` smoke driver; `horeka_nemotron.py --trial` (default `source-importance`, 640 tokens) |

Plan: `/Users/valerianfourel/.claude/plans/zesty-orbiting-magpie.md` (SI-v1, approved).
Decisions: no human annotators (references `deepseek-v4-flash-0731`, `glm-4.5-air`);
all judging on HoreKa; earliest ARR cycle; judge the full generator answer.
Claims-v3 stays as exploratory legacy (trial job 5169065 area: J2 order-unstable).

## Evidence gathered (HoreKa, read-only probes)

- 200 Qwen cells: Parallel 7 sources, Reactive 2-6; ~5.4 sources/cell.
- Shuffled erased by reranking in ~95 % of groups (generator input identical).
  Ablation target shown in ~40 % (Parallel) / ~20 % (Reactive) of groups.
- 41 % of answers carry spontaneous `(S1)` markers (masked to `[ref]`).
- 3,000 Qwen cells: 35 % stored answers cut at 1,200 chars; 99.7 % recoverable in
  full from the trace (median 1,492, p90 2,033, max 3,309 chars).
- HoreKa scheduling: fair-share weight 0 (age 5000 over 7 days, QOS 15000);
  `GrpJobs=50` per user (reached 28 Sep 21:42); submit cap 295; `dev` QOS 1 running.

## Operator pages (Mac, `/Users/valerianfourel/Hamburg/GEODML_Unified/`)

`nemotron-si-smoke.html` (SI smoke test in salloc), `horeka-qwen-send-all.html`
(sender for all bouts; approval 631-893 recorded 29 Sep), `llama-from-hub.html`
(Llama pull into `$W/llama-hf`, started 28 Sep in tmux `llama-pull` on hkn1993),
`full-answers-check.html`, `job-status-counts.html`, `peak-node-usage.html`,
`si-probe-followup.html`, `llama-answer-cutoffs.html`.

## Open at session end

1. SI smoke test: salloc job 5169925 on hkn0403 was granted; results not returned.
2. Qwen: bouts 631-893 approved; confirm `qwen-sender-all` was started and old
   senders stopped. 137 completed / 4 failed bouts as of 29 Sep 04:30.
3. Llama pull: confirm `PULL_EXIT=0`, then run block 3 and the full-answer check.
4. Not yet built: SI runner/CLI for production, results aggregation reader,
   reference-judge client, validation selection, regime recovery for axis >= 0.70.
