# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GEODML — B2B SaaS Keyword Search Ranking Experiment. Compares how an AI-powered search (SearXNG + LLM re-ranking) ranks B2B SaaS domains vs. traditional search engines (DuckDuckGo, Google) for 50 B2B SaaS keywords.

## Setup

```bash
pip install -r requirements.txt
```

### Required Environment Variables (`.env.local`)

- `HF_TOKEN` — Hugging Face API token (https://huggingface.co/settings/tokens)
- `SEARXNG_URL` — SearXNG instance URL (default: `https://searx.be`)

## Running

```bash
# AI-powered search: SearXNG → Mistral-7B LLM re-ranking
python run_ai_search.py

# Traditional engines: DuckDuckGo + Google scraping
python run_engine_search.py
```

Results are saved to `results/` as JSON and CSV files.

## Project Structure

- `keywords.txt` — 50 B2B SaaS category keywords
- `src/config.py` — env loading, constants (TOP_N=10)
- `src/keywords.py` — keyword file parser
- `src/searxng_client.py` — SearXNG JSON API client
- `src/llm_ranker.py` — HF Inference API (Mistral-7B-Instruct) domain ranker
- `src/engine_scraper.py` — DuckDuckGo + Google scrapers
- `src/results_io.py` — JSON/CSV result persistence
- `run_ai_search.py` — main script for AI search pipeline
- `run_engine_search.py` — main script for traditional engine scraping

## Key Dependencies

- `huggingface_hub` — HF Inference API client
- `duckduckgo_search` — DDG search library
- `googlesearch-python` — Google search library
- `tldextract` — domain normalization
- `python-dotenv` — env file loading
