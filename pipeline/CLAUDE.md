# CLAUDE.md

## Project

GEODML — GEO (Generative Engine Optimization) vs SEO causal inference experiment using Double Machine Learning. Proves that LLM-powered search engines cite domains based on different causal factors than traditional search engines.

## Architecture

The experiment instruments a **local Perplexica clone** (Next.js + SearXNG + LLM) to capture pre-LLM and post-LLM data, then compares against traditional SERP rankings.

### 5-Phase Pipeline

```
Phase 1: Instrument Perplexica → capture pre-LLM (raw SearXNG) + post-LLM (citations) data
Phase 2: Batch runner → process 50 keywords through Perplexica's internal pipeline
Phase 3: SERP collector → query Google/DDG via SearXNG for the same domains
Phase 4: Feature extractor → scrape pages, extract treatment + confounder variables
Phase 5: Exporter → join all tables → CSV ready for DML in Python
```

### Running the Experiment

```bash
cd Perplexica
npm install

# Phase 2: Run keywords through Perplexica (requires configured LLM + SearXNG)
npx tsx src/lib/experiment/runner.ts --keywords data/keywords.txt --mode speed

# Phase 3: Collect Google/DDG rankings for same domains
npx tsx src/lib/experiment/serpCollector.ts

# Phase 4: Extract page features (treatments + confounders)
npx tsx src/lib/experiment/featureExtractor.ts

# Phase 5: Export joined CSV for DML analysis
npx tsx src/lib/experiment/exporter.ts --output data/experiment_results.csv
```

Or via API (while Perplexica is running):
```bash
curl -X POST http://localhost:3000/api/experiment \
  -H "Content-Type: application/json" \
  -d '{"keywordsFile": "data/keywords.txt", "optimizationMode": "speed"}'

curl http://localhost:3000/api/experiment/status
```

### Key Files

```
Perplexica/
├── src/lib/experiment/
│   ├── types.ts             # TypeScript interfaces
│   ├── db.ts                # Drizzle ORM schema (5 experiment tables)
│   ├── citationParser.ts    # Parse [1][2][3] from LLM output
│   ├── runner.ts            # Phase 2: batch keyword processor
│   ├── serpCollector.ts     # Phase 3: Google/DDG SERP collection
│   ├── featureExtractor.ts  # Phase 4: page feature extraction
│   └── exporter.ts          # Phase 5: CSV export for DML
├── src/app/api/experiment/
│   ├── route.ts             # POST trigger experiment
│   └── status/route.ts      # GET check progress
├── drizzle/0001_experiment_tables.sql  # Migration
├── data/keywords.txt        # 50 B2B SaaS keywords
└── data/raw_html/           # Saved HTML for reproducibility
```

### DB Tables

- `experiment_queries` — one row per keyword run
- `pre_llm_results` — raw SearXNG results BEFORE LLM touches them
- `post_llm_citations` — which sources the LLM cited (and which it didn't)
- `serp_rankings` — Google/DDG rank for each domain
- `page_features` — treatments (T1-T5) and confounders (X1-X6)

### Design Decisions

- **GEO-first**: start from LLM citations, then check traditional engines (not the reverse)
- **Internal function calls**: runner calls Perplexica internals directly (not HTTP)
- **Checkpoint/resume**: runner skips already-processed keywords via DB check
- **Bare keyword queries**: keywords passed as-is, no sentence wrapping
- **All existing Perplexica functionality preserved** — experiment code is additive only
