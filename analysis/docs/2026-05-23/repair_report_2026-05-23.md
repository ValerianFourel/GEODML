# Repair audit — what's missing for full 1011-keyword coverage

_Generated: 2026-05-23 15:12_

_Source manifest: `manifests/repair_manifest.parquet` (rows=104)_

_DATA_ROOT used by last audit: `~/geodml_data`_


> **Note:** numbers reflect whatever mirror the audit was pointed at. Run `repair_audit.py` on JUPITER for the authoritative cluster-side view.

## 1. Rollup

| stage | cells | done | partial | empty | total_kw_gap |
|---|---|---|---|---|---|
| order_probe | 64 | 16 | 16 | 32 | 42832 |
| probing | 8 | 2 | 0 | 6 | 6 |
| rerank | 32 | 8 | 8 | 16 | 21416 |

## 2. Status (last dispatch state)

| stage | DONE | TODO |
|---|---|---|
| order_probe | 16 | 48 |
| probing | 2 | 6 |
| rerank | 8 | 24 |

## 3. Stage `order_probe` — 48 cells with gap > 0

| variant | engine | pool | model | seed | actual_kw | target_kw | gap | status | last_jobid |
|---|---|---|---|---|---|---|---|---|---|
| neutral_rag | ddg | 50 | Llama-3.3-70B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 50 | Llama-3.3-70B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Llama-3.3-70B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Llama-3.3-70B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Qwen2.5-72B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Qwen2.5-72B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 50 | Qwen2.5-72B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 50 | Qwen2.5-72B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Qwen2.5-72B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Qwen2.5-72B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Qwen2.5-72B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Qwen2.5-72B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Llama-3.3-70B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Llama-3.3-70B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Llama-3.3-70B-Instruct | 42 | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Llama-3.3-70B-Instruct | 123 | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | searxng | 20 | Qwen2.5-72B-Instruct | 123 | 0 | 1009 | 1009 | TODO |  |
| neutral_rag | searxng | 20 | Qwen2.5-72B-Instruct | 42 | 0 | 1009 | 1009 | TODO |  |
| neutral_rag | searxng | 20 | Llama-3.3-70B-Instruct | 42 | 0 | 1009 | 1009 | TODO |  |
| neutral_rag | searxng | 20 | Llama-3.3-70B-Instruct | 123 | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Llama-3.3-70B-Instruct | 123 | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Llama-3.3-70B-Instruct | 42 | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Qwen2.5-72B-Instruct | 42 | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Qwen2.5-72B-Instruct | 123 | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 50 | Qwen2.5-72B-Instruct | 42 | 0 | 980 | 980 | TODO |  |
| biased_rag | searxng | 50 | Qwen2.5-72B-Instruct | 123 | 0 | 980 | 980 | TODO |  |
| biased_rag | searxng | 50 | Llama-3.3-70B-Instruct | 123 | 0 | 980 | 980 | TODO |  |
| biased_rag | searxng | 50 | Llama-3.3-70B-Instruct | 42 | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Llama-3.3-70B-Instruct | 42 | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Llama-3.3-70B-Instruct | 123 | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Qwen2.5-72B-Instruct | 123 | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Qwen2.5-72B-Instruct | 42 | 0 | 980 | 980 | TODO |  |
| neutral | ddg | 20 | Llama-3.3-70B-Instruct | 42 | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 20 | Llama-3.3-70B-Instruct | 123 | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 20 | Qwen2.5-72B-Instruct | 123 | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 20 | Qwen2.5-72B-Instruct | 42 | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Qwen2.5-72B-Instruct | 42 | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Qwen2.5-72B-Instruct | 123 | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Llama-3.3-70B-Instruct | 123 | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Llama-3.3-70B-Instruct | 42 | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 50 | Qwen2.5-72B-Instruct | 123 | 600 | 1011 | 411 | TODO |  |
| neutral | ddg | 50 | Qwen2.5-72B-Instruct | 42 | 600 | 1011 | 411 | TODO |  |
| neutral | ddg | 50 | Llama-3.3-70B-Instruct | 42 | 600 | 1011 | 411 | TODO |  |
| neutral | ddg | 50 | Llama-3.3-70B-Instruct | 123 | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Llama-3.3-70B-Instruct | 123 | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Llama-3.3-70B-Instruct | 42 | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Qwen2.5-72B-Instruct | 42 | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Qwen2.5-72B-Instruct | 123 | 600 | 1011 | 411 | TODO |  |

## 3. Stage `probing` — 6 cells with gap > 0

| variant | engine | pool | model | seed | actual_kw | target_kw | gap | status | last_jobid |
|---|---|---|---|---|---|---|---|---|---|
| neutral | nan |  | Llama-3.3-70B-Instruct |  | 0 | 1 | 1 | TODO |  |
| neutral | nan |  | Qwen2.5-72B-Instruct |  | 0 | 1 | 1 | TODO |  |
| biased_rag | nan |  | Llama-3.3-70B-Instruct |  | 0 | 1 | 1 | TODO |  |
| biased_rag | nan |  | Qwen2.5-72B-Instruct |  | 0 | 1 | 1 | TODO |  |
| neutral_rag | nan |  | Llama-3.3-70B-Instruct |  | 0 | 1 | 1 | TODO |  |
| neutral_rag | nan |  | Qwen2.5-72B-Instruct |  | 0 | 1 | 1 | TODO |  |

## 3. Stage `rerank` — 24 cells with gap > 0

| variant | engine | pool | model | seed | actual_kw | target_kw | gap | status | last_jobid |
|---|---|---|---|---|---|---|---|---|---|
| neutral_rag | ddg | 50 | Llama-3.3-70B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Llama-3.3-70B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 20 | Qwen2.5-72B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | ddg | 50 | Qwen2.5-72B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Qwen2.5-72B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Qwen2.5-72B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 20 | Llama-3.3-70B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| biased_rag | ddg | 50 | Llama-3.3-70B-Instruct |  | 0 | 1011 | 1011 | TODO |  |
| neutral_rag | searxng | 20 | Qwen2.5-72B-Instruct |  | 0 | 1009 | 1009 | TODO |  |
| neutral_rag | searxng | 20 | Llama-3.3-70B-Instruct |  | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Llama-3.3-70B-Instruct |  | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 20 | Qwen2.5-72B-Instruct |  | 0 | 1009 | 1009 | TODO |  |
| biased_rag | searxng | 50 | Qwen2.5-72B-Instruct |  | 0 | 980 | 980 | TODO |  |
| biased_rag | searxng | 50 | Llama-3.3-70B-Instruct |  | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Llama-3.3-70B-Instruct |  | 0 | 980 | 980 | TODO |  |
| neutral_rag | searxng | 50 | Qwen2.5-72B-Instruct |  | 0 | 980 | 980 | TODO |  |
| neutral | ddg | 20 | Llama-3.3-70B-Instruct |  | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 20 | Qwen2.5-72B-Instruct |  | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Qwen2.5-72B-Instruct |  | 79 | 1011 | 932 | TODO |  |
| biased | ddg | 20 | Llama-3.3-70B-Instruct |  | 79 | 1011 | 932 | TODO |  |
| neutral | ddg | 50 | Qwen2.5-72B-Instruct |  | 600 | 1011 | 411 | TODO |  |
| neutral | ddg | 50 | Llama-3.3-70B-Instruct |  | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Llama-3.3-70B-Instruct |  | 600 | 1011 | 411 | TODO |  |
| biased | ddg | 50 | Qwen2.5-72B-Instruct |  | 600 | 1011 | 411 | TODO |  |

## 4. RAG-variant gap (biased_rag + neutral_rag)

**52 cells** with missing RAG output. Total keyword-gap = **48,136**.

| variant | cells_with_gap | total_kw_gap |
|---|---|---|
| biased_rag | 26 | 24068 |
| neutral_rag | 26 | 24068 |

## 5. Next actions

Run on JUPITER:

```bash
set -a; source .env; set +a
# Foreground loop (Ctrl-C safe, resumable):
./scripts/repair_loop.sh
# Or one-shot (re-invoke as needed):
./scripts/repair_run.sh
```

After Stage A/A'/probing cells all reach `gap = 0`, re-derive downstream:

```bash
./scripts/slurm/dispatch_bcd.sh --with-stage-f
```


**Note on RAG cells at 0 keywords:** the underlying rag_index must be built first (or the rerank job will produce empty passages). On a machine with internet + an OpenAI key:

```bash
OPENAI_API_KEY=sk-... bash scripts/run_rag_embeddings.sh
```
