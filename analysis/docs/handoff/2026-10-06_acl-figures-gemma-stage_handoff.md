# ACL figures and the Gemma answer-support stage

Valerian asked for the math on all Qwen, llama and Gemma runs (how prompt intent moves the
search: queries, pool before reranking, shortlist, ranking) and four ACL figures: the prompt
axis with two keywords and three picks each, the mini internet, the full pipeline, and prompt
position (x) against the position of the cited sources (y).

Branch `codex/acl-figures-20261006` (from a7fc602):
- `analysis/scripts/acl_figures.py`: `axis-examples` (seeded pick per band: start 0–0.15,
  middle 0.425–0.575, end 0.85–1; robinhood vs etrade plus one random keyword) and `render`
  (fig1–fig4 as vector PDF + PNG). Fig2/fig3 use only established counts; fig4 needs results.json.
- `intent_stages.stage_curves` and `results["curves"]` (20 bins, keyword-clustered SE, natural and
  all conditions, per model x engine and per model).
- Stage G (Gemma SI-v4 answer support, development): `intent_stages_study.py gemma-extract`
  (read-only over shards/*/results: latest report cells.jsonl.gz + frozen task records, pages
  matched by exact title+text) and `analyze --gemma`; excluded from replication. Runner: set
  GEMMA_RUN for `horeka-intent-stages.sbatch`. Protocol addendum 13–14, before any result.
- Tests: test_acl_figures (2), test_intent_stages (+3, incl. a fake Gemma run end to end);
  38 pass with geo-drivers and page-readiness suites.

Not covered: ~21k Qwen cells that exist only on Hugging Face (stated limit); answer stage A
needs the answer-export paths. Nothing ran on a cluster. Figure previews (fig1 one keyword,
fig2, fig3) in ARR_ACL_CycleOct2026/figures-wip. Next: HoreKa steps in the session reply.
