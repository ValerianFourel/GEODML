# Markdown report of what the saved generator datasets show

Valerian asked for code that writes .md reports on the ~300k-cell Llama and Qwen
generator datasets. New: `analysis/scripts/report_generator_outputs.py` and
`analysis/tests/test_report_generator_outputs.py`. CPU-only and read-only. It
counts every registered generator task by its latest ledger state and reads only
the verified generation rows of completed cells, never traces. Each prompt is
joined to `axis_1_percentile_0_1` from `final-audit/final-axis-map.jsonl`
(archived in `ValerianFourel/geoaxis-prompts-generation-26k` @ 74e64ff, sha256
43189f68…). Output: report.md, summary.json, descriptives/decile/association/
matched-pair CSVs. Sections: inventory against 26,009×12 design cells; outputs
by method/engine/condition; natural-vs-shuffled and llama4-vs-qwen38 ranking
differences (ranking_agreement); axis deciles; Spearman associations of prompt
means with keyword-blocked permutation p (reusing `_association`). All results
are descriptive; the axis is a prompt property, not a treatment. No judge data.

Tests: 4 new synthetic tests; 64 pass together with SI and Gemma v4 suites. No
scientific result yet. Mid-turn Valerian said an interactive GPU allocation was
open and asked for a GPU script for the latent-space/permutation study. This
report needs no GPU and can run inside that allocation. The GPU step of the page
readiness pipeline (LLM2Vec page embeddings) belongs to the concurrent session's
`page_readiness_ordering.py`, whose page-extract 5179545 / page-relocate 5179548
were live, so no duplicate GPU script was written. Next: download the axis map
on a login node, run the report in the allocation, return report.md.
