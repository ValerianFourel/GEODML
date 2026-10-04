# Page readiness and generator ordering

Question: do generators rank action-ready pages higher, and does that preference
grow as the prompt moves from information seeking to action readiness?

Everything here is observational. The prompt axis is a measured property of the
prompt text, not a randomized treatment. Page coordinates use a subspace fitted
on prompts, so they are out-of-domain descriptions of page text. They describe
pages; they are not treatments or confounders.

## The subspace and how to locate it again

The 26,009-prompt axis comes from the relaxed final audit:

1. Each prompt is embedded with two frozen LLM2Vec views (Qwen3-8B and Mistral)
   and projected through that view's frozen ridge map
   (`readiness_embedding_map.json`): normalize, center, project on the two
   supervised axes.
2. Raw axis values are z-scaled with the development means and scales recorded
   in the robustness battery. The Mistral view is rotated onto the Qwen view
   with the battery's development-only orthogonal Procrustes rotation
   (`_aligned_projection_rows`).
3. `consensus_axis_1_z` is the mean of the two axis-1 z values. The prompt's
   0–1 position is its percentile rank among the 26,009 prompts, ties broken by
   prompt ID (`final-axis-map.jsonl`, `axis_1_percentile_0_1`).

`relocate` proves this chain is recovered from archived artifacts alone. It
recomputes every prompt's coordinate from the archived per-view projections and
the battery, and requires exact agreement (1e-9) with `final-axis-map.jsonl`.
With `--qwen-map/--qwen-embeddings` (and the Mistral pair) it also pushes the
archived 26,009 prompt embeddings back through each frozen map and requires the
archived raw axes (max difference 1e-4, float32 storage); this needs no GPU.
With `--fresh-qwen/--fresh-mistral`, it also compares a fresh GPU re-embedding of
512 archived prompts with their archived raw axis-1 values (Spearman ≥ 0.999).
Together these show that the frozen maps and the embedding models still place
text where they placed it before.

## Pages

A page is one evidence item exactly as the generator saw it: title, newline,
snippet text. Pages are deduplicated by the SHA-256 of that text. Each page is
embedded with both views, through the same maps, z-scaling and rotation as the
prompts. Each page then gets a consensus z and a percentile on the prompt scale;
values outside the prompt range are flagged.

## Analysis

For every answer, the presented evidence order and the final ranking (URLs) come
from the sealed generator dataset, read the same way as SI judging reads them.

- Ranking model: a top-k Plackett–Luce model over the ranked prefix. Each
  ranked page is chosen among presented pages not yet ranked; unranked pages
  remain available. Utility = `b_page·z + b_interaction·z·(prompt_axis − 0.5)`,
  plus a fixed effect per presented position, capped at 20. The evidence-order
  conditions randomize presented order, so the position effects separate order
  from page content.
- Inference: keyword-cluster bootstrap (200 replicates) for 95% intervals, and a
  within-keyword permutation null for the interaction (200 replicates): prompt
  axis values are shuffled among a keyword's prompts.
- Robustness: the same fit using only the Qwen or only the aligned Mistral page
  z, and separate fits per search method.
- Descriptive: per answer, the mean z of ranked pages minus the mean z of all
  presented pages, and the top-ranked page minus the presented mean, by prompt
  decile with keyword-cluster intervals.

The prompt-only question, how the rankings themselves change along the prompt
axis, is already covered by `report_axis_ranking_change.py` and
`report_latent_ranking_relationship.py`. Run them on the same dataset.

## Stages

| Stage | Where | Writes |
| --- | --- | --- |
| `extract` | login or CPU job | `observations.jsonl.gz`, `pages.jsonl.gz`, counts of skipped answers |
| `sample-prompts` | login | 512 archived prompts for the re-embedding check |
| `embed` (`horeka-page-readiness-embed.sbatch`) | one 4-A100 node per view | resumable 20,000-page shards |
| `merge` | login | archived projection format plus manifest |
| `relocate` | login | `relocation.json`; exit 2 if it fails |
| `analyze` | CPU job | `results.json`, `page_coordinates.jsonl.gz`, `report.html` |

Each stage writes a new directory and refuses to overwrite. An interrupted
stage leaves `<output>.partial` for inspection. Embedding resumes per shard and
refuses a rerun with changed settings. Use llama4 now; add `--source
<qwen dataset>:qwen38` once Qwen publication is complete. No code change is
needed for that.

Tests: `analysis/tests/test_page_readiness_ordering.py`. They cover exact
relocation and drift detection, prompt-scale placement, recovery of a planted
page effect and interaction, a null with no interaction, extraction from a
sealed fixture dataset, sharded and resumable embedding with merge, and an
end-to-end analysis with the HTML report. These synthetic tests establish no
scientific result.

## One interactive allocation

`horeka-page-readiness-interactive.sh JOBID` runs every stage inside an existing
four-A100 allocation, from the login shell that holds it, with `srun
--jobid`. CPU stages use 32 cores. Each embedding view is one 4-GPU step with
one pinned worker per GPU. The fresh 512-prompt re-embedding must reproduce
the archived axis before any page is embedded. Finished stages are skipped and
embedding resumes from saved shards, so the same command continues in a later
allocation. The script never cancels, extends or releases the allocation.

## Snippet embedding dataset

`horeka-snippet-embeddings-interactive.sh JOBID` extracts every distinct snippet
shown to llama (`llama-hf/dataset`) and Qwen (`shared-hours/dataset`, sealed
finished cells at extraction time) across both engines. It embeds them with both
views using `--save-embeddings`, and `package` writes `snippet-embeddings-v1/`:

- `snippets.parquet`: snippet id, title, snippet, text, URLs, engines, models,
  times shown, raw and aligned axis values, consensus z, prompt-scale percentile;
- `embeddings/qwen3-8b-llm2vec.npy`, `embeddings/mistral-7b-instruct-v0.2-llm2vec.npy`:
  float32 vectors, row `i` = parquet `row` `i`;
- `README.md`, and `manifest.json` with model revisions, code commit, the relocation
  check and SHA-256 of every file.

`publish --package DIR` (login node, write token prompted) uploads it to the private
dataset under `derived/snippet-embeddings/<name>/`. It refuses a public dataset or an
existing path, verifies every file's size and LFS SHA-256, and writes
`<name>.published.json`. Later Qwen completions need a new extract and package name.
