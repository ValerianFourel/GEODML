# DataForSEO integration — paper-size experiment

Thin DataForSEO client + Bundle A' runner for refilling the audit-documented
broken confounders (Moz-failed domain metrics + missing keyword difficulty).

## Files

- `client.py` — `DataForSEOClient` with bulk-ranks/backlinks/referring-domains/spam-score and bulk KD methods. HTTP Basic auth via `DATAFORSEO_LOGIN` / `DATAFORSEO_PASSWORD` in `paperSizeExperiment/.env`.
- `run_bundle_a.py` — orchestrator. Reads domains from `consolidated_results/regression_dataset.csv` and keywords from `keywords.txt`. Batches at 1,000 targets per request, saves raw JSON per chunk and flat CSVs.
- `output/` — gitignored. Contains per-endpoint CSVs, `raw/<endpoint>_chunk_<i>.json`, `run_manifest.json`, `run.log`.

## Usage

```bash
# Dry run — estimates cost, no API calls
source venv312/bin/activate
python -m paperSizeExperiment.dataforseo.run_bundle_a --dry-run

# Live run — hard cost ceiling
python -m paperSizeExperiment.dataforseo.run_bundle_a --max-cost 5

# Skip either block
python -m paperSizeExperiment.dataforseo.run_bundle_a --skip-backlinks
python -m paperSizeExperiment.dataforseo.run_bundle_a --skip-kd
```

## KD geography note

Labs keyword difficulty for Germany (`location_code=2276`) only supports German
(`language_code=de`). The keyword list is English B2B SaaS, so the default is
USA + English (`--location-code 2840 --language-code en`). Override if you need
a different locale.

## Backlinks subscription

Bulk Ranks / Backlinks / Referring Domains / Spam Score all require a separate
Backlinks API subscription on the DataForSEO account. Until that is activated
(https://app.dataforseo.com/backlinks-subscription) every call returns status
40204. The runner aborts that endpoint after the first chunk and moves on.

## Cost model (at 13,435 domains × 1,011 keywords)

| Endpoint | Scope | Unit price | Estimated |
|---|---|---|---|
| Bulk Ranks | 13,435 domains | $0.00006 | $0.81 |
| Bulk Backlinks | 13,435 domains | $0.00006 | $0.81 |
| Bulk Referring Domains | 13,435 domains | $0.00006 | $0.81 |
| Bulk Spam Score | 13,435 domains | $0.00006 | $0.81 |
| Labs Bulk Keyword Difficulty | 1,011 keywords | $0.0005 | $0.51 |

Backlinks endpoints only bill if the subscription is active.
