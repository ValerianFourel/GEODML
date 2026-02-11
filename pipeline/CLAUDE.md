# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GEODML — GEO (Generative Engine Optimization) causal inference experiment. Compares how an LLM re-ranks search results vs. traditional search engine rankings for 50 B2B SaaS keywords.

## Architecture

```
keywords.txt → SearXNG (Apptainer container, port 8888) → LLM re-ranking (HF API) → results/
```

- **SearXNG**: Apptainer/Docker container aggregating Google, Bing, DuckDuckGo, Brave, Startpage
- **LLM**: Qwen2.5-72B-Instruct via HuggingFace Inference API (remote, no local GPU)
- **DuckDuckGo fallback**: `ddgs` library used directly if SearXNG unavailable

## Setup & Run

```bash
bash setup.sh                          # pulls SearXNG container + creates Python venv
bash start_searxng.sh &                # start SearXNG in background
source venv/bin/activate
python run_ai_search.py                # run full experiment (50 keywords)
python run_ai_search.py --keywords 5   # test with 5 keywords
python summarize_results.py results/ai_search_rankings.json
```

## Key Files

- `setup.sh` — Apptainer/Docker/native detection, SearXNG pull, venv creation
- `run_ai_search.py` — main experiment: SearXNG → LLM re-ranking → JSON/CSV
- `run_ranking_comparison.py` — live experiment + side-by-side terminal comparison
- `summarize_results.py` — reads existing results JSON, prints ranking report
- `run_engine_search.py` — traditional DDG + Google scraper for baseline
- `src/searxng_client.py` — SearXNG client with DDG fallback
- `src/llm_ranker.py` — Qwen2.5-72B re-ranking via HF chat_completion API
- `src/experiment_context.py` — IP, geolocation, machine info for provenance

## Environment

- `.env.local`: `HF_TOKEN`, `SEARXNG_URL` (default `http://127.0.0.1:8888`)
- Python venv in `venv/`, SearXNG image in `searxng.sif`
- HPC (JUWELS): Apptainer for SearXNG, `--no-cache-dir` for pip (home quota)
