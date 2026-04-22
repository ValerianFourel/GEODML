# Experiment Roadmap

## Pipeline Phases

**Phase 1 — SERP + LLM Re-Ranking** (`gather_data.py`)
- For each keyword: query the search engine (DDG or SearXNG) for N results (serp20 or serp50)
- Send those results to the LLM (Llama/Qwen via HF API) to re-rank and pick top 10
- Output: `rankings.csv`, `keywords.jsonl` (pre_rank vs post_rank per domain)

**Phase 2 — HTML Fetch + Code-Based Features** (`gather_data.py`)
- Fetch the actual HTML of every unique URL from Phase 1
- Extract code-based features via BeautifulSoup: T1 statistical density, T2 question headings, T3 structured data (JSON-LD), T4 citation count, word count, readability, links, images
- Cache HTML to disk (`html_cache/`)
- Output: `features.csv` with code-measured treatments + confounders

**Phase 3 — LLM Feature Extraction** (`gather_data.py` with `--llm-features`)
- Re-read cached HTML, build a page digest, send to the LLM
- LLM scores the same T1-T4 treatments but from its own "perception" (T1_llm, T2_llm, T3_llm, T4_llm)
- Resumable, checkpoints every 5 URLs
- Output: updates `features.csv` with LLM-judged treatment columns

**Phase 4 — Enriched Features / Moz API** (`extract_features.py`)
- Adds confounders from external APIs: domain authority, backlinks, referring domains via Moz Links API
- Adds new treatments: T5 topical competence (embedding similarity), T6 freshness (date extraction), T7 source type (earned media classification)
- Adds confounders: BM25 score, title/snippet similarity, brand recognition
- Output: `features_new.csv`

## Design

- **1011 keywords** (B2B SaaS queries)
- **Search engines**: DuckDuckGo, SearXNG
- **LLM models** (HuggingFace Inference API):
  - `meta-llama/Llama-3.3-70B-Instruct`
  - `Qwen/Qwen2.5-72B-Instruct`
- **SERP pool sizes**: serp20/top10, serp50/top10
- **Full 2x2x2 factorial**: 2 engines x 2 models x 2 pool sizes = **8 runs**
- **Pipeline per run**: gather_data (P1+P2+P3) -> extract_features (P4) -> clean_data -> analyze -> halo analysis

## Run Status (as of 2026-04-09)

| # | Engine | Model | SERP/Top | P1 (SERP+LLM) | P2 (HTML) | P3 (LLM feat) | P4 (extract) | clean_data | analysis_full | halo |
|---|--------|-------|----------|----------------|-----------|----------------|--------------|------------|---------------|------|
| 1 | DDG | Llama-3.3-70B | 20/10 | DONE (1011 kw) | DONE (6413 URLs) | DONE (5636/5637) | DONE | DONE | DONE (204 exp, 32 sig) | DONE |
| 2 | DDG | Llama-3.3-70B | 50/10 | DONE (1011 kw) | DONE (6817 URLs) | DONE (5647/5647) | DONE | DONE | DONE (204 exp, 39 sig) | DONE |
| 3 | DDG | Qwen2.5-72B | 20/10 | DONE (1011 kw) | DONE (6947 URLs) | 91.8% (5422/5906) | TODO | DONE* | TODO | TODO |
| 4 | DDG | Qwen2.5-72B | 50/10 | DONE (1011 kw) | DONE (8591 URLs) | 35.6% (2612/7347) | TODO | TODO | TODO | TODO |
| 5 | SearXNG | Llama-3.3-70B | 20/10 | DONE (960 kw) | DONE (6968 URLs) | DONE (5941/5941) | DONE | DONE | DONE (216 exp, 44 sig) | DONE |
| 6 | SearXNG | Llama-3.3-70B | 50/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 7 | SearXNG | Qwen2.5-72B | 20/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8 | SearXNG | Qwen2.5-72B | 50/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

\* Run 3 has geodml_dataset.csv with outcomes (77%) but no features_new.csv, so enriched treatments/confounders are missing. Analysis needs re-run after Phase 3+4 complete.

**Note**: Runs 6-8 require a running SearXNG container for Phase 1.

## Pipeline Steps (per run)

1. **gather_data.py** — Phase 1: SERP queries + LLM re-ranking -> keywords.jsonl, rankings.csv
2. **gather_data.py** — Phase 2: Fetch HTML for all URLs -> html_cache/, features.csv
3. **gather_data.py** — Phase 3: LLM-based feature extraction (T1-T4 via LLM) -> features.csv (updated)
4. **extract_features.py** — P4: T5-T7, BM25, Moz API -> features_new.csv
5. **clean_data.py** — Merge rankings + features + features_new -> geodml_dataset.csv
6. **analyze.py** — Full DML: 17 treatments x 3 outcomes x 2 methods x 2 learners -> analysis_full/
7. **earned_media_halo.py** — Halo effect analysis -> analysis_halo/

## Key Findings (Runs 1, 2, 5 completed)

### Consistent across all completed runs:
- **T7 (Earned media) demotion**: rank_delta coef = -1.3 to -2.4 (***) — earned media pages consistently demoted by LLM
- **Halo effect**: brands mentioned in earned media get better post_rank (***) — LLM boosts mentioned brands

### Run-specific findings:
- **Run 1** (DDG/Llama/s20): 32/204 significant results. T1_llm small negative effect (-0.01, p=0.015)
- **Run 2** (DDG/Llama/s50): 39/204 significant results. Larger SERP pool amplifies T3 structured data and T6 freshness effects
- **Run 5** (SXG/Llama/s20): 44/216 significant results. T4_llm citation authority significant (-0.09, p=0.004)

### Cross-model analysis:
- 492 experiments, 165 significant (p<0.05)
- T7 earned media is the dominant treatment across all subsets

## Known Issues

1. **`experiment.json` resume bug** (FIXED): when gather_data.py resumes, `per_keyword_results` in experiment.json is empty. Fixed `clean_data.py` to fall back to `keywords.jsonl`.

2. **Moz API coverage ~6-10%**: domain authority/backlinks unusable as confounders. OpenPageRank already integrated as alternative.

3. **T5 topical competence 0% for DDG runs**: embedding similarity depends on SearXNG-specific data. Not available for DuckDuckGo runs.

4. **conf_title_kw_sim / conf_snippet_kw_sim 0% for DDG runs**: these confounders rely on SearXNG snippet format.

## SearXNG Availability

SearXNG requires a running container. Start with:
```bash
docker run -d --rm -p 8888:8080 -v $(pwd)/searxng-config:/etc/searxng searxng/searxng:latest
```
SearXNG runs must be done when the container is active. DDG runs use the `ddgs` library directly (no container needed).
