# JUPITER end-of-experiment runbook (2026-05-18)

Single source-of-truth for finishing the GEO-DML bf16 reconciliation on
JUPITER Booster under the **scifi** project. Captures everything learned
from the May 17–18 setup attempts: the `/p/` ↔ `/e/` filesystem
distinction, the cell-clear pattern needed for a real smoke test, and the
push-to-HF flow at the end.

> Where you are right now: code, .venv, and HF model cache (459 GB) are
> on JUPITER under `/e/project1/scifi/fourel1/GEODML_Analysis/`. Dataset
> still needs to be copied from `/p/scratch/` to `/e/scratch/`. SLURM
> association `scifi/booster/normal` is live. Smoke test on JUPITER
> compute node hasn't actually run inference yet — the skip-guard
> triggered on existing 1009-record cells.

---

## Key infrastructure facts (don't forget these)

| Thing | Value |
|---|---|
| JUPITER login | `ssh fourel1@login.jupiter.fz-juelich.de` → `jpbl-s02-*` |
| JUPITER compute | `salloc` node like `jpbo-***-**` |
| SLURM account | `scifi` |
| SLURM partition | `booster` (no devel exists; pass `--partition booster` if using `--smoke`) |
| Filesystem on login | `/p/project1/scifi/...` AND `/e/project1/scifi/...` both mounted |
| Filesystem on compute | **`/e/` only** — `/p/` is invisible from compute |
| Repo (target path) | `/e/project1/scifi/fourel1/GEODML_Analysis/` |
| Dataset (target path) | `/e/scratch/scifi/fourel1/data/...` |
| HF model cache | `/e/project1/scifi/fourel1/GEODML_Analysis/hf_cache/` |
| Python venv | `/e/project1/scifi/fourel1/GEODML_Analysis/.venv` (ARM-built, torch 2.8) |
| HF dataset repo | `ValerianFourel/geodml-papersize` |

`jutil env activate -p scifi` sets `$PROJECT=/p/project1/scifi` on login
and `$PROJECT=/e/project1/scifi` on compute. Use `${SCRATCH}` and
`${PROJECT}` in `.env` so paths resolve correctly on both.

---

## ① Cancel the futile copy job, then verify what's already on /e/

```bash
# JUPITER login node
scancel 462458   # was trying to read /p/ from a compute node — won't work

ls /e/project1/scifi/fourel1/GEODML_Analysis/hf_cache/
du -sh /e/project1/scifi/fourel1/GEODML_Analysis/hf_cache/
# Expect: ~459 GB, 3 model dirs (Llama-3.3-70B, Qwen-72B, Llama-3.1-8B)

ls /e/scratch/scifi/fourel1/ 2>/dev/null || echo "scratch MISSING — proceed to ②"
```

## ② Create scratch dir + copy dataset (~15 GB, ~3 min on login)

```bash
mkdir -p /e/scratch/scifi/fourel1

# Run on login node (sees both /p/ and /e/). Background + log.
nohup rsync -a \
  /p/scratch/scifi/fourel1/data \
  /p/scratch/scifi/fourel1/*.md \
  /e/scratch/scifi/fourel1/ \
  > /tmp/dataset_copy.log 2>&1 &
echo "PID=$!"

# Watch
tail -f /tmp/dataset_copy.log
watch -n 5 'du -sh /e/scratch/scifi/fourel1/'

# Sanity once done
ls /e/scratch/scifi/fourel1/data/main/
ls /e/scratch/scifi/fourel1/data/rag_index/
ls /e/scratch/scifi/fourel1/data/runs/ | head
```

## ③ Use the /e/ repo as the canonical work dir + update .env

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis     # IMPORTANT: always sbatch from here
vim .env
```

Set:
```ini
JUWELS_ACCOUNT=scifi
JUWELS_PROJECT=scifi
HF_TOKEN=hf_xxx_write_scoped_token
GEODML_DATA_ROOT=${SCRATCH}/fourel1
LOCAL_PRECISION=full
MAX_KW=99999
PRIMARY_MODEL=meta-llama/Llama-3.3-70B-Instruct
PROXY_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_DATASET_REPO=ValerianFourel/geodml-papersize
HF_HOME=${PROJECT}/fourel1/GEODML_Analysis/hf_cache
```

Validate:
```bash
jutil env activate -p scifi
set -a; source .env; set +a
echo "GEODML_DATA_ROOT=$GEODML_DATA_ROOT"   # /p/scratch/scifi/fourel1 on login
echo "HF_HOME=$HF_HOME"                      # /p/project1/scifi/fourel1/...
ls "$GEODML_DATA_ROOT/data/main/" | head
```

## ④ Verify visibility from a compute node

```bash
salloc --nodes=1 --gres=gpu:4 --time=00:30:00 --account=scifi -p booster
srun --cpu-bind=none --nodes=1 --pty /bin/bash -i

# Now on jpbo-***-** (compute)
cd /e/project1/scifi/fourel1/GEODML_Analysis
jutil env activate -p scifi
set -a; source .env; set +a
echo "GEODML_DATA_ROOT=$GEODML_DATA_ROOT"   # /e/scratch/scifi/fourel1 on compute
ls "$GEODML_DATA_ROOT/data/main/" | head    # MUST succeed (otherwise dataset copy didn't reach /e/)
ls "$HF_HOME" | head                         # MUST show 3 model dirs
nvidia-smi -L                                # 4× GH200 120GB
exit                                         # leave salloc
exit                                         # release allocation
```

All three `ls` MUST succeed. If anything fails → go back to ① or ② and fix.

## ⑤ Patch _common.sh for /p/ ↔ /e/ translation (one-time, then push)

The login node submits sbatch with `SLURM_SUBMIT_DIR=/e/...` (since you cd
to `/e/`-side repo per ③). But for robustness against accidental `/p/`
submits, add a translator. From a login node:

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
python3 - <<'PY'
from pathlib import Path
p = Path("scripts/slurm/_common.sh")
src = p.read_text()
marker = 'jutil env activate -p "$JUWELS_PROJECT"'
insert = '''

# JUPITER compute nodes mount /e/ instead of /p/. If SLURM_SUBMIT_DIR
# was captured under /p/ on a login node, translate to the matching /e/
# path using the dynamic $PROJECT and $SCRATCH set by jutil.
if [ ! -d "$SLURM_SUBMIT_DIR" ] && [ -n "${PROJECT:-}" ]; then
  ALT="$SLURM_SUBMIT_DIR"
  ALT="${ALT/#\\/p\\/project1\\/scifi/$PROJECT}"
  ALT="${ALT/#\\/p\\/scratch\\/scifi/$SCRATCH}"
  if [ -d "$ALT" ]; then
    echo "[common] path-translate /p/ -> $ALT"
    SLURM_SUBMIT_DIR="$ALT"
    export SLURM_SUBMIT_DIR
  fi
fi
'''
src = src.replace(marker, marker + insert)
p.write_text(src)
print("patched")
PY

# Also re-apply the JUPITER smoke partition tweak
sed -i 's/PARTITION="develbooster"/PARTITION="booster"/' scripts/slurm/dispatch_all.sh

# Commit + push
git add scripts/slurm/_common.sh scripts/slurm/dispatch_all.sh
git commit -m "fix(slurm): JUPITER path-translate /p/->/e/ + booster smoke"
git push origin main
```

## ⑥ Real smoke test (the one that actually runs bf16)

Skip-guard caused 12-sec exits before. To force actual inference on a cell
that already has 1009 records, archive the JSONL first:

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

ABS=$GEODML_DATA_ROOT/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2
mv "$ABS/keywords.jsonl"    "$ABS/keywords.jsonl.pre-bf16-bak" 2>/dev/null
mv "$ABS/.rerank_ckpt.json" "$ABS/.rerank_ckpt.json.pre-bf16-bak" 2>/dev/null
ls "$ABS/"

MAX_KW=5 LOCAL_PRECISION=full \
  ./scripts/slurm/dispatch_all.sh --smoke --partition booster \
    --only rerank --variant biased \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --engines searxng --pools 20

squeue -u $USER --format='%.10i %.32j %.2t %.10M'
JID=$(squeue -u $USER -h -o '%i' | head -1)
tail -f logs/*${JID}*.out
```

**Pass criteria** (5-10 min runtime, not 12 seconds):
- `[common] path-translate /p/ -> /e/...` (if your sbatch was submitted from `/p/`)
- `[LocalRanker] model=meta-llama/Llama-3.3-70B-Instruct precision=bf16-full`
- `[LocalRanker] hf_device_map: ... devices`
- `[LocalRanker]   cuda:N allocated=~35-70 GiB` per GPU
- 5 keywords processed

Verify records carry bf16:
```bash
python <<'PY'
import json
p = "/e/scratch/scifi/fourel1/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2/keywords.jsonl"
recs = [json.loads(l) for l in open(p) if l.strip()]
print(f"records={len(recs)}, precision_set={set(r['llm_parameters']['precision'] for r in recs)}")
PY
# expect: records=5, precision_set={'bf16-full'}
```

If smoke fails — paste the log and we fix before proceeding. **Do not
launch the full run until smoke is bf16-full ✓.**

## ⑦ Clear all 4-bit snippet cells (the bf16 redo replaces them)

The smoke verified bf16 works. Now clear the existing 4-bit JSONLs so the
full run processes every keyword fresh. (`--resume` would otherwise skip
the 1009 existing keywords.)

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

for V in biased neutral ; do
  for D in $GEODML_DATA_ROOT/data/runs/*_top10_${V}/phase2 ; do
    [ -f "$D/keywords.jsonl" ] && \
      mv "$D/keywords.jsonl" "$D/keywords.jsonl.4bit-bak-$(date +%Y%m%d)"
    rm -f "$D/.rerank_ckpt.json"
  done
done

# Verify all snippet cells are now empty (no keywords.jsonl)
find $GEODML_DATA_ROOT/data/runs -path "*_top10_biased/phase2/keywords.jsonl" 2>/dev/null
find $GEODML_DATA_ROOT/data/runs -path "*_top10_neutral/phase2/keywords.jsonl" 2>/dev/null
# Both should return nothing.
```

(The `_rag` variants are also re-done in bf16 because `INCLUDE_RAG_REDO=1`
in the full launch will overwrite their existing api-hf records.)

## ⑧ Full launch (~3-5 days, scifi accounting)

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

tmux new -s gpurun
INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 \
  ./scripts/finish_on_gpu.sh \
    2>&1 | tee logs/finish_on_gpu_$(date +%Y%m%d_%H%M%S).log
# Detach: Ctrl-B then D
```

Submits ~96 sbatch jobs:
- 16 snippet rerank (biased + neutral × 2 models × 2 engines × 2 pools)
- 16 RAG rerank (bf16 redo of the API-produced records)
- 64 order_probe (4 variants × 2 models × 2 engines × 2 pools × 2 seeds)
- ~10 Stage F gap fills (probing for neutral; full Stage F for both `_rag`)

Each cell processes every keyword (no MAX_KW cap because we set
`MAX_KW=99999` and no cell has more than ~1500 keywords).

## ⑨ Monitor (intermittent, days)

```bash
watch -n 30 'squeue -u $USER --format="%.10i %.32j %.2t %.10M" | head -40'
.venv/bin/python scripts/audit_status.py | tail -30
squeue -u $USER -h | wc -l    # 0 = all done

# Recent failures
sacct -u $USER -S now-24hours -X --format=JobID,JobName%30,State,ExitCode,Elapsed \
  | grep -vE "COMPLETED|PENDING|RUNNING"
```

## ⑩ Standardize after all jobs finish

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

# Idempotent backfill (covers any records the new code path missed)
.venv/bin/python scripts/backfill_precision.py \
  --root "$GEODML_DATA_ROOT" --include-recent

# Audit
.venv/bin/python scripts/audit_status.py | tail -20

# Confirm precision is unified bf16-full across rerank cells
python <<'PY'
import json, glob, os
root = os.environ['GEODML_DATA_ROOT']
prec = set()
for p in glob.glob(f"{root}/data/runs/*/phase2/keywords.jsonl"):
    with open(p) as f:
        for line in f:
            if line.strip():
                prec.add(json.loads(line)['llm_parameters']['precision'])
                break
print('Precision labels:', prec)
assert prec == {'bf16-full'}, f"NOT standardized: {prec}"
print('STANDARDIZED')
PY
```

## ⑪ Push to HF (#2 — post-cluster, pre-DML)

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

cat >> "$GEODML_DATA_ROOT/CHANGELOG.md" <<EOF

## $(date -u +%Y-%m-%d) — bf16 reconciliation complete (JUPITER scifi, full-keyword)

Snippet + RAG rerank/order_probe re-run in LOCAL_PRECISION=full on JUPITER
Booster GH200 under scifi accounting. All Stage A records now \`bf16-full\`.
Every keyword processed (no MAX_KW cap). Stage F gaps closed.

EOF

hf upload-large-folder ValerianFourel/geodml-papersize \
  "$GEODML_DATA_ROOT" \
  --repo-type dataset \
  --num-workers 1 \
  --exclude "data/runs/*/phase2/html_cache/**" \
  --exclude "data/runs/*/phase2/html_cache.tar.gz" \
  --path-in-repo data
```

## ⑫ DML analysis (Stages B + C + D + figures + order_probe summary)

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

bash scripts/continue_pipeline.sh

# Headline sanity vs pre-bf16 baseline
python <<'PY'
import pandas as pd, os
root = os.environ['GEODML_DATA_ROOT']
for v in ['biased','neutral','biased_rag','neutral_rag']:
    df = pd.read_parquet(f"{root}/data/dml_results/dml_results_long_{v}.parquet")
    r = df.query(
        "subset=='POOLED' and method=='plr' and learner=='lgbm' "
        "and outcome=='rank_delta' and treatment=='T7_source_earned'"
    ).iloc[0]
    print(f"{v:14s}  T7 coef={r.coef:+.3f}  p={r.p_val:.4f}  n={int(r.n_obs):,}")
PY
# Pre-bf16 reference (for sanity):
#   biased       T7 = -1.607***
#   neutral      T7 = -0.417***
#   biased_rag   T7 = -1.268***
#   neutral_rag  T7 = -0.496***
# Acceptable: within ±15% of these, same sign, still p<0.001.
```

## ⑬ Generate `RESULTS_SUMMARY.md`

```bash
.venv/bin/python scripts/make_results_summary.py \
  --data-root "$GEODML_DATA_ROOT" \
  --variants biased neutral biased_rag neutral_rag \
  --output "$GEODML_DATA_ROOT/RESULTS_SUMMARY.md" \
  --title "GEODML — Final (bf16 reconciliation, JUPITER scifi, $(date -u +%Y-%m-%d))"

head -80 "$GEODML_DATA_ROOT/RESULTS_SUMMARY.md"
```

## ⑭ Push to HF (#3 — final, with DML + summary)

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
set -a; source .env; set +a

# Data tree (excludes html_cache — unchanged from prior push)
hf upload-large-folder ValerianFourel/geodml-papersize \
  "$GEODML_DATA_ROOT" \
  --repo-type dataset \
  --num-workers 1 \
  --exclude "data/runs/*/phase2/html_cache/**" \
  --exclude "data/runs/*/phase2/html_cache.tar.gz"

# Stage F outputs (CSVs + plots, outside the data tree)
hf upload-large-folder ValerianFourel/geodml-papersize \
  /e/project1/scifi/fourel1/GEODML_Analysis/interpretability/output \
  --repo-type dataset \
  --num-workers 1 \
  --path-in-repo interpretability/output
```

## ⑮ Mac-side: pull final dataset + analyze

```bash
# On your Mac
cd ~/Hamburg/GEODML_Analysis

.venv/bin/python <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'ValerianFourel/geodml-papersize',
    repo_type='dataset',
    local_dir='/Users/valerianfourel/Hamburg/geodml-dataset',
    allow_patterns=[
        'data/main/**', 'data/dml_results/**', 'data/features/**',
        'data/order_probe/order_probe_summary.parquet',
        'interpretability/output/**',
        'RESULTS_SUMMARY.md', 'README.md', 'PROVENANCE.md', 'CHANGELOG.md',
    ],
)
PY

$EDITOR ~/Hamburg/geodml-dataset/RESULTS_SUMMARY.md
```

---

## Failure modes + recovery

### A. Smoke gives `precision=4bit-nf4`
- Cause: `LOCAL_PRECISION` not exported by sbatch.
- Fix: re-submit with `--export=ALL,LOCAL_PRECISION=full,...`. Discard the
  contaminated cell:
  ```bash
  V=biased; CELL=searxng_Llama-3.3-70B-Instruct_serp20_top10_${V}
  mv $GEODML_DATA_ROOT/data/runs/$CELL/phase2/keywords.jsonl{,.bak}
  rm $GEODML_DATA_ROOT/data/runs/$CELL/phase2/.rerank_ckpt.json
  ```

### B. `slurmstepd: couldn't chdir to /p/...`
- Cause: SLURM_SUBMIT_DIR captured on login under `/p/`; compute can't see it.
- Fix: confirm ⑤ patch landed (`grep "path-translate" scripts/slurm/_common.sh`),
  OR always `cd /e/project1/scifi/fourel1/GEODML_Analysis` before sbatch.

### C. OOM on Qwen-72B bf16
- Cause: 72B bf16 + RAG 2400-char prompts at edge of 4×96 GB.
- Fix:
  ```bash
  sed -i 's/--top-k-rag 3/--top-k-rag 2/' scripts/slurm/run_rerank.sbatch
  ```

### D. HF push interrupted
- `hf upload-large-folder` is resumable. Just re-run the same command.

### E. DML coef wildly different from pre-bf16 reference
- E.g. T7 jumps from −1.61 to −0.5 or flips sign.
- Don't push (step ⑭) until you understand why. Likely:
  - Precision class genuinely shifts the LLM (paper-worthy — document in CHANGELOG).
  - Stage B features parquet was stale (re-run `interpretability.pipeline.features`).
  - LLM model checkpoint changed (check `meta.json` in `rag_index`).

### F. Compute node missing /e/<your dir>
- Cause: scifi project provisioning incomplete for your user.
- Fix: mkdir on login first (login sees the right /e/ mount); compute then sees it.

---

## Budget

| Step | Wall time | scifi GPU-hr | $$ |
|---|---|---:|---|
| ① cancel + verify | 1 min | 0 | 0 |
| ② dataset copy /p→/e | ~5 min | 0 | 0 |
| ③ .env update | 2 min | 0 | 0 |
| ④ verify on compute | 5 min | 0.5 (idle salloc) | 0 |
| ⑤ patch _common.sh | 2 min | 0 | 0 |
| ⑥ smoke | ~10 min | 1 | 0 |
| ⑦ clear 4-bit cells | 1 min | 0 | 0 |
| ⑧ full launch + ⑨ monitor | **~3-5 days** | ~230 | 0 |
| ⑩ standardize | 5 min | 0 | 0 |
| ⑪ push HF #2 | ~30 min | 0 | 0 |
| ⑫ DML | ~30 min | 0 | 0 |
| ⑬ summary | 1 min | 0 | 0 |
| ⑭ push HF #3 | ~10 min | 0 | 0 |
| ⑮ Mac analysis | ~10 min | 0 | 0 |

End-to-end: ~4-5 days. scifi quota burn: ~230 GPU-hours.

---

## Quick-reference command index

```bash
# JUPITER login
ssh -i ~/.ssh/id_ed25519 fourel1@login.jupiter.fz-juelich.de

# Always work from here
cd /e/project1/scifi/fourel1/GEODML_Analysis
source .venv/bin/activate
jutil env activate -p scifi
set -a; source .env; set +a

# Interactive compute node (for debugging)
salloc --nodes=1 --gres=gpu:4 --time=02:00:00 --account=scifi -p booster
srun --cpu-bind=none --nodes=1 --pty /bin/bash -i

# Kill all your jobs
scancel -u $USER

# Quota
sshare -A scifi -u $USER
du -sh $GEODML_DATA_ROOT $HF_HOME
```

End.
