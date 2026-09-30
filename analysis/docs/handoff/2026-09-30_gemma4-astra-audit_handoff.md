# Gemma SI-v3 Astra audit, 30 September 2026

Valerian requested download of the verified private Hub export, GPT-6 Astra subagent review against ten criteria, and a report including timing and errors. The task is complete. No new inference, allocation, corpus mutation or judge configuration change occurred.

## Evidence and code state

Active checkout `.worktrees/threehour-relaunch-fix`, branch `threehour-relaunch-fix`, prior HEAD `6b3e74a`, upstream `origin/codex/pilot-continuation`. This handoff and report are documentation/evidence only. No production code changes or new tests.

Private dataset `ValerianFourel/geodml-experiment-v2-paper-private`, revision `098d3f467b4a0f3fec6da2789043421428390770`, archive path `reviews/gemma-si-v3/exports/31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f.zip`. Downloaded 1,253,914 bytes; archive SHA256 and all 57 member sizes/hashes/inventory verified. Local extraction `/private/tmp/gemma-v3-audit/data`; archive under that directory's `download/` subtree. This is downloaded evidence, not a live cluster inspection.

Durable report: [Gemma SI-v3 audit](../reviews/gemma-si-v3-20260930/report.md). Adjacent JSON/Markdown files contain the two semantic reviews, unchanged blind ranges, integrity checks, timing and aggregate counts, plus a SHA256 evidence manifest. Raw archive/corpus and credentials are not included in Git.

Three explicitly configured GPT-6 Astra/high agents completed: `original_astra`, `fresh_astra`, `integrity_astra`. Original/fresh semantic reviewers read all cells/sources, saved acceptable ranges before opening grades, and reviewed both passes. The integrity agent checked inputs/results/metrics. An initial incorrectly forked `audit_original` agent was interrupted immediately and does not count as a reviewer. Reviewer assessments are reference-model evidence, not human gold. Primary nonblind checks include a challenge to the loyalty Popupsmart 4–5 range; it remains frozen with the disagreement documented.

## Findings

- Forty distinct cells/prompts: twenty original development, twenty fresh, zero prompt overlap. Both splits balance Qwen/Llama and Parallel/Reactive.
- 214 unique SI source tasks, each repeated twice, plus forty J1 tasks twice: 508 successful requests. All structural checks pass. No judge retry/failure/token truncation. Integrity report includes 24,286 assertions, including embedded Nemotron history; that count is not the sample size.
- Two-pass SI grade agreement 209/214, all within one. Cell vectors 35/40 exact. Top-group MEMBERSHIP changes in three cells, not four; pharmacy changes top grade3→4 without membership change. All forty J1 repeats equal. Third repetitions absent.
- Semantic audit: 181 distinct selected pairs, 157 supported, eight unsupported, sixteen ambiguous; 361 occurrences, 314/15/32. Range agreement223/428, mostly under-credit. Pass1-only subgroup rates avoid counting repeats as independent evidence.
- Clear error: Alibaba evidence assigned grade3 for AliExpress claim twice. Strong centrality concerns: EMR demo and MetaMask recommendation sources receive2. Capability-vs-procedure and some centrality differences are uncertain and explicitly distinguished.
- Original traces/sealed generator records absent, preventing independent provenance/recovery confirmation. Five cells retain prose S IDs after masking. Actual measured semantic-axis coordinates and controlled interventions absent.
- Decision: do not approve production SI-v3 yet. Human-adjudicate key cases, close provenance/eligibility/masking gaps, then test a narrowly revised protocol on new cases. Shared answer outline is an untested centrality hypothesis; it cannot by itself fix entity errors. All reviewed cases become development evidence for subsequent tuning.

## Timing and errors

Original pass seconds47.4/44.9; fresh67.1/61.6; judging total221.0s. Wrapper start to server finish533.2s/654.5s, distinct from stage elapsed464.1s/566.1s. Both server/controller exits0. Median SI latencies original0.716/0.704s, fresh1.884/1.860s.

Saved sacct: jobs5171305/5171481 parent TIMEOUT after3607/3601s, four GPUs each, approximately8.009 GPU-hours combined. The successful judge outputs and shutdown preceded timeouts. No pre-shutdown server ERROR or CUDA OOM. Original teardown has EngineDeadError traceback; both have cleanup warnings. Intentional HTTP401 auth probes are not judge failures. Earlier shell preparation failures and the fixed write-token403 are reported separately.

## Verification and next actions

Report counts were recomputed from per-source audit records, archive result/task coverage cross-checked, copied evidence hashes verified, Markdown links checked and `git diff --check` run. No scientific computation or production tests were needed for this documentation audit.

The requested audit is complete. Later preparation or inference is a separate milestone: no allocation authorized. If continuing, first read the report's limits and adjudication notes; preserve both original frozen inputs and blind review ranges. Do not infer population accuracy, semantic-axis effects, or a working v4 from this diagnostic batch.
