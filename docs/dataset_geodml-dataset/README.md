# GEODML consolidated dataset

Single tree pulling together the three sources of truth for the GEODML
prompt-induced-bias study:

1. **Mac upstream** (`~/Hamburg/GEODML/paperSizeExperiment/output/`) — extracted
   html_caches for all 8 cells, plus per-cell `phase3/features_new.parquet`
   with full treatments + Moz/DataForSEO confounders.
2. **Cluster snapshot** — rerank checkpoints, order-probe jsonls, DataForSEO
   API outputs, Stage F interpretability outputs (ablation, saliency,
   weights, partial probing).
3. **Local Mac runs** (2026-05-07) — Stage B/C/D outputs, unified parquet,
   scraped HTML gap-fills.

## Layout

```
data/
├── serp/                                       4 phase0 SERP parquets (input)
├── dataforseo/                                 keyword-level confounders + raw API
├── runs/<cells>/phase2/                        per-cell rerank outputs
│   ├── html_cache/                             extracted HTML, sha256(url)[:16].html
│   ├── keywords.jsonl                          rerank output (per variant)
│   ├── rankings.csv
│   └── .rerank_ckpt.json
├── order_probe/                                jsonls + summary parquet
├── features/
│   ├── features_<engine>_top<pool>.parquet     Stage B (4 files, ~91% coverage)
│   └── dfs_keyword_confounders.parquet         keyword-level DataForSEO
├── main/
│   ├── full_experiment_data_<variant>.parquet  Stage C — variants:
│   │                                              biased, neutral, biased_rag, neutral_rag
│   └── full_experiment_unified.parquet         single multi-cell DML-ready table
├── dml_results/
│   └── dml_results_long_<variant>.parquet      Stage D (4 variants — same set as above)
└── rag_index/<engine>_top<pool>/               RAG retrieval index (NEW 2026-05-08)
    ├── full_passages.parquet                   url → trafilatura body (no 800-char cap)
    ├── chunks.parquet                          (url, chunk_idx, text), char-recursive splitter
    ├── chunk_embeddings.npy                    float32 (n_chunks, 1536) — text-embedding-3-small
    ├── keywords.parquet                        per-cell unique keyword list
    ├── keyword_embeddings.npy                  float32 (n_keywords, 1536)
    ├── retrieved_top3.parquet                  precomputed top-3 chunks per (keyword, url)
    └── meta.json                               build config + provenance

interpretability/output/                         Stage F (ablation, saliency, weights, probing)
archives/                                        zip snapshots ready for HF push

PROVENANCE.md                                    where every file came from
CHANGELOG.md                                     dated entries of changes
AUDIT.txt                                        latest audit_pipeline.py output
README.md                                        this file
refresh.sh                                       rebuild the symlinked tree from sources
push_to_hf.sh                                    upload to Hugging Face dataset
```

## Per-row LLM execution metadata (NEW 2026-05-17)

Every row of `data/main/full_experiment_data_<variant>.parquet` and every
record in `data/runs/*/phase2/keywords.jsonl` + `data/order_probe/*.jsonl`
now carries:

| Column | Values | Meaning |
|---|---|---|
| `llm_backend`   | `local`, `api`, `openai` | Which Python class served the inference (LocalRanker / InferenceRanker / OpenAIRanker). |
| `llm_precision` | `bf16-full`, `4bit-nf4`, `api-hf`, `api-openai`, `unknown` | Normalized execution-regime label. |

Use these to stratify when comparing snippet vs RAG arms — the two were
originally produced under different inference stacks (snippet on cluster
4-bit, RAG via HF Inference API at full precision). Current breakdown:

```python
biased         {'4bit-nf4': 45,967}
neutral        {'4bit-nf4': 52,256}
biased_rag     {'api-hf':   33,384}
neutral_rag    {'api-hf':   31,525}
```

A full reconciliation re-running snippet in `bf16-full` on JUWELS is in
progress; the next dataset revision will show `bf16-full` for snippet too.

See `PROVENANCE.md` for the backfill methodology
(`scripts/backfill_precision.py` in the analysis repo).

## Quick use

```bash
export GEODML_DATA_ROOT=/Users/valerianfourel/Hamburg/geodml-dataset

# inspect current state
cat $GEODML_DATA_ROOT/AUDIT.txt | tail -10

# load the unified DML-ready parquet
python -c "
import pandas as pd
df = pd.read_parquet('$GEODML_DATA_ROOT/data/main/full_experiment_unified.parquet')
print(df.shape)
print(df.groupby(['axis_prompt','axis_passage_mode']).size())
"

# read the headline DML result
python -c "
import pandas as pd
b = pd.read_parquet('$GEODML_DATA_ROOT/data/dml_results/dml_results_long_biased.parquet')
n = pd.read_parquet('$GEODML_DATA_ROOT/data/dml_results/dml_results_long_neutral.parquet')
m = lambda d: d[(d['subset']=='POOLED') & (d['method']=='plr') & (d['outcome']=='rank_delta') & (d['learner']=='lgbm')].set_index('treatment')['coef']
out = pd.DataFrame({'biased': m(b), 'neutral': m(n)})
out['delta'] = out['neutral'] - out['biased']
print(out.sort_values('delta', key=abs, ascending=False).round(3))
"
```

## Axes of variation (32 cells)

The unified parquet exposes every dimension as an explicit column:

| Axis | Values | Cells |
|---|---|---|
| `axis_engine` | searxng / ddg | 2 |
| `axis_model` | Llama-3.3-70B-Instruct / Qwen2.5-72B-Instruct | 2 |
| `axis_pool` | 20 / 50 | 2 |
| `axis_prompt` | biased / neutral | 2 |
| `axis_passage_mode` | snippet / passage | 2 |
| **total** | | **32** |

Plus the constant `axis_top_n=10`.

## Status (see AUDIT.txt for live numbers)

| Stage | Coverage | Notes |
|---|---|---|
| A — rerank | 32 snippet cells (full) + 16 RAG cells (400+ each) | passage variants archived → `archives/passage_runs_20260508_143229/` |
| A' — order probe | 64/64 snippet cells + 7/32 RAG cells | RAG order_probe partial — see `docs/RESUME-RAG.md` for the resume runbook (HF credit top-up needed) |
| B — features | 4/4 | built from upstream phase3 (91% treatment coverage) |
| C — merge | 4/4 | DataForSEO confounders joined |
| D — DML | 2/4 reliable (biased + neutral, 280 fits each); 2/4 partial (passage, smoke n too small) | full grid PLR × {LGBM, RF} × 7 subsets × 10 treatments × 2 outcomes |
| F — interpretability | 74/80 | probing 2/8 — needs JUWELS GPU |

## Headline result

POOLED · plr · lgbm · rank_delta:

```
T7_source_earned: biased -1.61***  →  neutral -0.42***   Δ = +1.19
```

SEO-biased prompts demote earned-media domains 4× harder than neutral
prompts. Reproduces the GEODML paper's main finding via DML on full
treatment + confounder controls.

## How to maintain

```bash
# rebuild the unified parquet from per-variant main tables
GEODML_DATA_ROOT=$PWD python ../GEODML_Analysis/scripts/build_unified_main.py

# refresh the audit
GEODML_DATA_ROOT=$PWD python ../GEODML_Analysis/scripts/audit_pipeline.py > AUDIT.txt

# rebuild from upstream sources (see refresh.sh)
bash refresh.sh
```

See `PROVENANCE.md` for what each file came from and `CHANGELOG.md` for
when it changed.
