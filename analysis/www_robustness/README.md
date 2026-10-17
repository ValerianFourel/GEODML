# WWW 2027 brief, T1: retriever robustness (GPU part)

Exploratory check requested by the WWW 2027 brief (work order T1): does the paper's stage split (query rewriting and the
cross-encoder shortlist carry the shift in cited-source intent) survive retrievers other than the frozen word-overlap
search? Parallel Expansion answers on the 733 held-out keywords, natural condition, the logged queries held fixed. No
generator is called; only retrieval, embeddings and cross-encoder scores are recomputed.

Division of work:

- **Mac, paper repository** (`paper_v2-www/www/analysis/t1_build.py`, CPU): the frozen search replicated (gate G1: it
  reproduces every logged retrieval and the prompt-text replay exactly), the no-keyword-field ablation and BM25
  (rank_bm25 0.2.2 BM25Okapi formula, k1 1.5, b 0.75, epsilon 0.25, English stopwords), pooled per answer like the
  pipeline; writes the bundle this step reads.
- **HoreKa, this folder** (`score_pairs.py`, four A100s): e5-large-v2 dense retrieval (revision f169b11, "query: " /
  "passage: " prefixes, cosine top 20, the frozen search's tie-break) and bge-reranker-v2-m3 scores (revision 953dc6f,
  the pipeline's own `SentenceTransformersCrossEncoderScorer`, batch 32, fp32) for every pair not scored in the logged
  run, plus the logged pairs of 1,000 answers as validation (gate G2 on the Mac).
- **Mac again** (`paper_v2-www/www/analysis/t1_stages.py`): the shortlist (top 7, ties in input order), stage slopes
  R0, R, C, P with the chain's own estimator and 200 keyword-bootstrap draws, P-normalised shares, cross-keyword shares
  and the overlap with the logged shortlist.

Resources (estimate, labelled [H] until the first shard measures it): embed 2–4 min, retrieve 3–6 min, scoring of
about 3–6 million pairs at 700–1,500 pairs/s per A100 in fp32, so 15–35 min on four GPUs; one allocation of one node,
four A100, 32 CPUs, 60 minutes wall time (4 GPU-hours), at most one more 60-minute allocation if shards remain
(resumable). Cheaper alternative held back: fp16 scoring (about half the time) would change the logged scores and
weaken gate G2.

Commands (HoreKa login shell; `W` is the GEODML workspace):

```bash
# 1. setup and status (read-only, plus the e5 download on the login node)
W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen
ls -d $W/environment/qwen-runtime/bin/python $W/models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
$W/environment/qwen-runtime/bin/python -c "import torch, transformers, sentence_transformers as s; print(torch.__version__, transformers.__version__, s.__version__)"
HF_HUB_CACHE=$W/models $W/environment/qwen-runtime/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('intfloat/e5-large-v2', revision='f169b11e22de13617baa190a028a32f3493550b6', allow_patterns=['*.json', '*.safetensors', '*.txt', '1_Pooling/*']))"
git -C "$CODE" fetch origin www-robustness && git -C "$CODE" worktree add "$W/code-www-robustness" FETCH_HEAD
df -h $W | tail -1

# 2. run inside one allocation (rerun the same line in a new allocation to continue)
bash $W/code-www-robustness/analysis/www_robustness/run_horeka.sh "$SLURM_JOB_ID" $W/runs/www-robustness/bundle-20261018 t1-20261018
```

Outputs: `$W/runs/www-robustness/<RUN_ID>/results/` (dense_searches, dense_items, shards and scores parquet files; the
embeddings in `results/embed/` need not be copied back), `diagnostics/` (versions, GPU, timings) and `logs/`.
