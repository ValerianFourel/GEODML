# BM25 generation rerun (SIGIR baseline)

The agentic search experiment rerun with only the retriever changed: BM25 (Lucene's BM25Similarity, k1 = 0.9,
b = 0.4, the ARR rebuttal's `bm25l`) in place of the frozen word-overlap search. Natural condition only, both methods,
both engine snapshots, every prompt. Qwen3.8-27B runs on HoreKa with the original Qwen bouts' vLLM setup; Llama-4-Scout
runs through Cheaper Inference's OpenAI-compatible API, with an `orig` arm (original search, same API) as its
comparison. Exploratory until analysed; no result here is a scientific finding by itself.

What stays the original run's: `ParallelExpansionV1` / `ReactiveSnippetLoopV1`, their prompts and schemas, the
generator client and seeds (`20260911 + int(cell_id[:8], 16) + call index`), max tokens (Qwen 256/2048 with thinking
off; Llama 256/4096), bge-reranker-v2-m3 on CPU, cell ids. Parallel Expansion replays its three logged queries
(written before any search, so the retriever cannot change them); the Reactive Loop runs live.

## Files

| File | Role |
| --- | --- |
| `porter.py` | NLTK 3.9.2 Porter stemmer, vendored (no NLTK on HoreKa) |
| `bm25.py` | `Bm25Index`, `Bm25SnapshotSearchAdapter` (same snapshot rows and payload as the frozen search) |
| `score_cache.py` | precomputed reranker scores in front of the live scorer |
| `build_bundle.py` | Mac: `cells`, `check-bm25`, `pairs`, `cache` |
| `generate.py` | resumable runner (`run`, `status`); JSONL part files, one line per finished cell |
| `horeka_qwen.py`, `run_qwen_bm25.sbatch` | Qwen on HoreKa under the original serving setup |
| `run_scores.sbatch` | GPU scores for the uncached Parallel pairs (`www_robustness/score_pairs.py score`) |
| `run_llama_api.sbatch` | Llama through the API on a HoreKa GPU node (reranker on the 4 GPUs, no vLLM) |
| `run_llama_api.sh` | the same on a Mac with a usable torch (not the Intel Mac: about one reranker pair per second) |
| `submit_qwen_chains.sh` | Qwen production: parallel slices (`SHARDS`), optional follow-ups (`CONTINUE=1`), `RESERVATION` |

## Steps

1. Mac: `build_bundle.py cells`, then `check-bm25` (must report every list identical).
2. Mac: `build_bundle.py pairs --scores <rebuttal HoreKa run>`; copy the pairs bundle and the cells bundle to HoreKa.
3. HoreKa: `run_scores.sbatch` (dev_accelerated, chain until done); copy `results/scores` back.
4. Mac: `build_bundle.py cache --scores <rebuttal run> --scores <new scores run>`; copy the cache to HoreKa.
5. HoreKa: one Qwen bout's `config.json` gives the serving setup and snapshot paths; copy the two snapshot files to the
   Mac for the Llama arms.
6. Smoke tests: Qwen 1 h on dev_accelerated (`PHASES="4:12:900 32:64"`, plus a small `RETRIEVER=orig` run);
   Llama `run_llama_api.sh bm25 200` and `orig 200`. Measure cells per hour and cost.
7. Production only after the budget is approved: Qwen chain on `accelerated`, Llama arms in full.

Outputs: `<run-dir>/parts/part-*.jsonl` (lines with status `complete` or `failed`, the original controller's result
fields, model, retriever, answer_id, the full trace and diagnostics) and `config.json` (settings a resume must
match). Importing the lines into the sealed dataset for the analysis is a later step.
