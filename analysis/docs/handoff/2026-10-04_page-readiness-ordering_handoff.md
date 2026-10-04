# Page readiness ordering: relocate the subspace, embed pages, relate rankings

Valerian asked for the full analysis of the prompt latent space against output
permutations. That means: a way to locate the information-to-action subspace
again, LLM2Vec embeddings of the page text, and the ordering along the axis.
Confirmed choices: all four parts; embed the snippets the generator saw
(title + snippet); llama4 first, qwen38 once published.

Reused, unchanged: frozen readiness maps, `project_text_embeddings`,
`LLM2VecPromptEmbedder`, the archived Qwen/Mistral alignment
(`_aligned_projection_rows`), and the SI reader (`completed_generator_refs`,
`iter_cells`, `_trace_evidence`). The prompt-only axis-to-ranking reports already
exist (`report_axis_ranking_change.py`, `report_latent_ranking_relationship.py`).

New: `analysis/interpretability/pipeline/page_readiness_ordering.py`,
`analysis/scripts/page_readiness_ordering.py` (extract, sample-prompts, embed,
merge, relocate, analyze with HTML), `analysis/docs/horeka-page-readiness-embed.sbatch`,
`analysis/docs/page_readiness_ordering.md`, and tests. Prompt position =
`axis_1_percentile_0_1` (consensus of Qwen and aligned Mistral axis-1 z,
percentile among 26,009). Pages use the same chain and are interpolated onto the
prompt percentile scale; out-of-range pages are flagged. Ranking model: top-k
Plackett–Luce with page z, page z × (axis − 0.5) and presented-position effects;
keyword-cluster bootstrap and within-keyword permutation null (200 each).

116 tests pass (9 new, synthetic only; no scientific result). Nothing was run on
a cluster. Unknown on HoreKa: the final-audit artifacts (`merged/qwen`,
`merged/mistral`, battery, map roots, final-axis-map, compliant-candidates), the
LLM2Vec snapshots and a Python with llm2vec. Both views may need different
Pythons, as in the final audit. Next: run the read-only probe, then extract,
embed (1-hour 4-A100 jobs, resumable), relocate, analyze.

## Update 2026-10-04 11:15 UTC (pasted evidence, not live)

Axis relocation passed exactly from archived artifacts (26,009 prompts, max z and
percentile difference 0.0; map replay 1.39e-16); fresh re-embedding of 512 prompts on
HoreKa: Spearman 0.99997 (Qwen view), 0.99998 (Mistral). Interactive runner
(`horeka-page-readiness-interactive.sh`, shared `horeka-page-readiness-lib.sh`) produced
`analysis-llama/`: llama, 282,000 ranked answers, consensus b_page +0.173,
b_int +1.252; intervals and per-view/method fits in results.json not yet read.

Search finding, documented for the paper: the frozen snapshot adapter ranks every row
of all 1,011 keywords by word overlap (keyword not passed to search). Glued-title
DuckDuckGo rows act as hubs (TurboTax row: 530 keywords, 3,763 displays). Artifact:
https://claude.ai/artifact/1N5JkPMM2493hsbeYYUbhN. `audit_snapshot_glued_rows.py`
(runner `horeka-audit-snapshot-glued-rows.sh`, locates snapshots by trace-recorded
SHA-256) not yet run. Fable prompt for the ACL rewrite:
`analysis/docs/fable-acl-rewrite-prompt-20261004.md`.

Snippets: llama + Qwen answers 600,034; distinct shown snippet texts 20,297; distinct
URLs 16,253 (2,774 URLs with 2 texts, 483 with 3+; DuckDuckGo variants mostly differ by
glued titles). Axis on snippets: 0.1% outside the prompt range; percentile median 0.41
(p95 0.66); Qwen/Mistral agreement r = 0.766; extremes face-valid.
Published `snippet-embeddings-v1` (texts, axis values, both float32 vector files) to
the private dataset under derived/snippet-embeddings/, Hub commit e6b253db, 5 files verified.
Next: run the corpus stage with commit 224fc75 (every servable snapshot row, glued flag),
publish `snippet-embeddings-corpus-v1`, evidence audit, per-engine/model fits.

Other launches today: Qwen recovery sender restarted with `--gpu-all-at-once`
(codex/qwen-recovery-20261003 at 8f60b25; 7/7 bouts submitted). Gemma v4 llama run
`gemma-si-v4-llama-cpu-5h-20261004`: CPU preparation 5179554 at 94% at 09:23 UTC; the
sender then submits up to 200 five-hour bouts. Note: a test-failing commit (14f954f)
was pushed by a `;`-chained command and fixed in 224fc75; chain commits with `&&`.
