# Audit C — Provenance facts

Repos inspected: `~/Hamburg/GEODML` (original pipeline + `paperSizeExperiment/` scaled run) and `~/Hamburg/GEODML_Analysis` (ported analysis repo). Git-dating caveat: the GEODML repo history was bulk-imported in one squash commit `9de74cc 2026-04-22`, so `git log --follow` collapses most pipeline files to that date; `git log -S` on an un-squashed sibling recovers true first-appearance dates, cross-checked against file mtimes and run manifests.

## 1. Re-ranking prompts

**Biased prompt (paper Listing 2)** — invoked by the scaled run (`paperSizeExperiment/run_experiment.py:75` calls `pipeline/gather_data.py`, confirmed in `paperSizeExperiment/experiment.log`). Template at `GEODML/pipeline/gather_data.py:409-425` (`_build_rerank_prompt`), verbatim:

```
Search keyword: {keyword}

Below are search engine results for the above keyword. Re-rank the results and return the top {top_n} software product domains, ordered by relevance to the keyword.

Exclude non-product sites: review aggregators, directories, Wikipedia, news, blogs, forums, YouTube.

Return only root domains, one per line. No explanations.

Search results:
{results_text}

Re-ranked product domains:
```

Byte-identical copy at `GEODML/src/llm_ranker.py:39-50`. Git: `git log -S 'software product domains' -- src/llm_ranker.py` → first appearance `ef08b10 2026-02-11 "fixed algorithm"`. The biased prompt text has existed since **2026-02-11**.

**Neutral prompt (paper Listing 1)** — only in the analysis repo: `GEODML_Analysis/interpretability/pipeline/prompts.py:72-77` (`_NEUTRAL_HEADER`) + footer lines 90-92, assembled by `build_rerank_prompt_with_spans` (lines 141-216):

```
Below are search engine results for the above keyword. Re-rank the results and return the top {top_n} URLs ordered by relevance to the keyword.

Return only root domains, one per line. No explanations.
```

Git history of `prompts.py`: first appearance `f00a180 2026-04-28` (pipeline port), passage variants `b1b1c51 2026-04-30`, last modified `ac3e993 2026-05-18` (bf16 LocalRanker + precision tracking).

**First data-collection run**: `GEODML/paperSizeExperiment/consolidated_results/tracker.json` — `"created_utc": "2026-03-26T16:01:13.295616+00:00"`; earliest run `duckduckgo_Llama-3.3-70B-Instruct_serp20_top10` started `2026-03-26T16:01:13Z`, ended `2026-03-27T22:35:58Z`. (`experiment_manifest.json`'s `2026-04-16T09:10:30Z` start is a no-op re-verification: `runs_skipped: 4`, `total_elapsed_min: 0.1`.)

**Conclusion**: the biased prompt (Listing 2) that generated the paper's main outcome data predates all outcome data (text in git since 2026-02-11; first inference run 2026-03-26). The neutral prompt (Listing 1) was introduced 2026-04-28 for the de-biased re-run — consistent with its stated purpose; the de-biased outcome data was collected after its prompt was written, and the neutral prompt was never edited after collection of the paper's neutral-arm outcomes began (last prompt change 2026-05-18 concerns the bf16 execution backend module, not the neutral header text — verify wording against the diff before citing this last clause).

## 2. User-Agent (Phase-2 page scraper)

Canonical fixed UA, verbatim:

```
Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0
```

- `GEODML/run_page_scraper.py:37` (`USER_AGENT = ...`; applied at line 83)
- `GEODML/pipeline/gather_data.py:56` (used at lines 155, 266, 891)
- `GEODML/paperSizeExperiment/config.py:67`; `run_phase0_serp.py:52`; `50_larger/run_page_scraper.py:52`; `src/engine_scraper.py:47,136`
- Documented at `GEODML/EXPERIMENT_REGISTRY.md:460`.

Note: the post-hoc gap-fill re-scraper in the analysis repo uses a different Chrome UA (`GEODML_Analysis/scripts/scrape_missing_html.py:40-43`, added 2026-05-07, commit `c4d4724`) — it only fills html-cache gaps and postdates data collection. Non-page-fetch UAs: `GEODML-DomainClassifier/1.0 (valerian.fourel@gmail.com)` (`paperSizeExperiment/classify_domains.py:333`), `Mozilla/5.0 (compatible; llms-txt-checker/1.0)` (`paperSizeExperiment/check_llms_txt.py:30`).

## 3. Keyword generation (1,011 queries)

Code: `GEODML/paperSizeExperiment/generate_keywords.py` → `keywords.txt`.

1. Seed source: hardcoded two-level `TOPIC_TREE`, 20 categories × 8-10 subcategories (159 total), `generate_keywords.py:33-236`.
2. Per subcategory an LLM call — default `meta-llama/Llama-3.3-70B-Instruct` via HF `InferenceClient.chat_completion` (`generate_keywords.py:291,262-265`; temperature=0.8, max_tokens=1000) — with the template at lines 245-259 requesting "exactly 25 unique search keywords that a real person would type into Google", 2-6 words, and "Do NOT repeat" a sampled set of existing keywords.
3. QA/filter: strips <think> tags, numbering, bullets, quotes; lowercases; keeps a line only if `2 <= len(line.split()) <= 8 and len(line) < 80` (lines 268-280).
4. Dedup = set membership on the normalized lowercased string: added only `if kw not in all_keywords` (lines 316, 346-349; `keywords.append(line.lower())` line 280) — exact-string, case-insensitive, no fuzzy/semantic dedup.
5. Stops at `len(all_keywords) >= args.target` (default 1000; lines 289, 334-336), second pass over 10 "Miscellaneous" topics if short (lines 358-385); output `sorted(all_keywords)` (lines 388-393).
6. Count reconciliation: `keywords.txt:1` header says "1011 unique keywords across 20 categories"; the file has exactly 1,011 non-comment lines; `tracker.json:20` and `experiment_manifest.json:3` both record `"keywords_count": 1011`; overshoot of the 1,000 target by 11 arises because each LLM batch (≤25) is added before the target check re-fires. **1,011 is the exact produced count — no discrepancy.**

## Reviewer-pack note (HF `ValerianFourel/geodml-emnlp-2026-reviewer`)

`scripts/build_reviewer_pack.py` ships only condensed tables (`tables/table2_dml_headline.csv`, `dml_all_specs_all_slices.csv`, probing/saliency/ablation summaries), figures, `verify.py`, and a README (lines 48-162, 199-309). It does NOT contain the prompt templates, `keywords.txt`, `generate_keywords.py`, or scraper code — the facts above must be cited to the source repos (raw runs are in HF `ValerianFourel/geodml-papersize`), not to reviewer-pack paths.
