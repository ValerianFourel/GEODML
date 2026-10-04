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
