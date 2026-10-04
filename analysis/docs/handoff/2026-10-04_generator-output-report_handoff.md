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

## Update: first run returned (pasted evidence, descriptive only)

Run in interactive allocation 5179583 on hkn0403, output
`$W/reviews/generator-output-report-20261004-0918` (commit 6fc1390, 200
permutations, axis map sha 43189f68). Qwen `$DS` = `$W/shared-hours/dataset`:
287,919 verified cells, 24,153 prompts (23,753 never claimed, 250 checkpointed,
11 running, 163 completed with unverified generation). Llama: 312,090 cells,
26,008 prompts, 6 terminal_failed. Natural vs shuffled: top-1 differs 1.6%
(Qwen 0.08%, Llama 3.2%). Llama vs Qwen on the same cell: top-1 differs 50.2%.
Toward action readiness, both rank fewer URLs (Llama 2.97→2.17, ρ −0.30; Qwen
5.15→4.79, ρ −0.22), write shorter stored answers (Qwen capped at 1,200) and
search more (Llama ρ +0.37, Qwen +0.19). Cross-model top-1 disagreement rises
42%→55%. Target selection falls slightly (ρ −0.07/−0.04), possibly through
ranking length. All p are at the 200-permutation floor 0.00498. Flags: Llama
Parallel-Expansion on searxng has 25% empty rankings; headline bullets omit the
design-cell labels for per-cell associations (report cosmetic bug).

## Update: answer text along the axis (`answer_readiness.py`)

Valerian: ignore shuffling; focus on how the prompt's latent position changes
the answer text. New `analysis/scripts/answer_readiness.py` (+ tests):
`export` writes the natural-condition answers (deduplicated, page format usable
as `page_readiness_ordering.py embed --pages`) with prompt axis positions;
`text` reports heuristic text measures (length, sentences, lists, digits,
currency, URLs, second person, action verbs, immediacy, hedges, explanatory
words per 100 words), axis deciles, keyword-blocked associations, within-keyword
high-vs-low-half contrasts and informative-Dirichlet distinctive words;
`analyze` places answers on the prompt scale after both LLM2Vec views are
embedded/merged, reusing `aligned`, `consensus` and `PromptScale` unchanged.
Answer coordinates are out-of-domain descriptions, not treatments/confounders.
`report_generator_outputs.load_cells` gained `keep_answer` (default unchanged).
68 tests pass. GPU embedding paths (LLM2Vec snapshots, llm2vec Python, maps,
battery on HoreKa) are unverified; a probe is needed before the embed step.

## Update: Fable planning prompt

Valerian asked for a prompt asking Fable to determine which tests can run on the
existing Qwen/Llama generator corpus before Gemma inference. Saved as
`analysis/docs/fable-pre-gemma-tests-prompt.md` (planning only; no code, cluster
or inference). It lists files to read, the corpus facts above with their caveats
(shuffle undone by compaction and out of scope; pool differences; Qwen cap; Llama
searxng empty rankings), the failed SI-v4 r2/Nemotron reviews, and the
auto-submitting Gemma sender after preparation job 5179554. It asks for a
measurement inventory, test catalogue, Gemma gating decision, ordered plan and
an implementation prompt for milestones 1–2. The answer-text report
(`$W/reviews/answer-readiness-20261004-text-0941`) is still unreviewed.

## Update: answer embeddings complete (pasted evidence, 2026-10-04)

Interactive allocation 5179583 (hkn0403) expired at its one-hour limit during
the first Qwen-view pass; 28 of 40 shards survived (atomic writes). Allocation
5179895 (hkn0402, commit 45099a6): Qwen view 28→40/40 between 09:22:48Z and
09:34:08Z; Mistral view 0→40/40 between 09:34:08Z and 10:09:44Z, both status 0.
Shard size 5,000 over 199,329 unique answers. Next: merge both views and run
`answer_readiness.py analyze` (CPU, minutes) against `$W/geoaxis-archive/battery`
and the final axis map. The fresh re-embedding (`relocate`) check is still
unreported, so answer positions stay provisional until it passes.
