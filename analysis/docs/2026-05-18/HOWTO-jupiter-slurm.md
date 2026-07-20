# HOWTO — Running GEODML SLURM jobs on JUPITER (scifi project)

**Single-source knowledge transfer doc.** Covers the JUPITER-specific quirks,
every issue we hit during setup (2026-05-17/18), how each was fixed, and the
exact commands to reproduce or extend the work.

Designed to be shared via claude.ai projects so future Claude conversations
land with full context.

---

## 0. TL;DR

We're running a 96-job SLURM campaign on JUPITER Booster (Grace-Hopper GH200s)
under the `scifi` project. The campaign re-runs 32 LLM rerank cells × multiple
variants at **full-precision bf16** to standardize on one precision across the
GEODML DML experiment. After ~4 fights with infrastructure, it's now working:

- ✅ **Smoke test passed 2026-05-18**: 5 keywords, Llama-3.3-70B bf16, sharded
  across 4× GH200 (32-33 GB allocated per GPU), zero errors.
- 🚧 **Full launch pending**: ~96 sbatch jobs, ~3-5 days wall time at scifi.

---

## 1. What this experiment is

**Goal**: measure how SEO-style prompt bias affects 70B/72B LLM reranking of
SERP results, via Double Machine Learning (DML) over a 32-cell factorial:

```
2 engines (searxng, ddg)
× 2 LLMs (Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct)
× 2 pool sizes (20, 50)
× 2 prompts (biased, neutral)
× 2 augmentation modes (snippet, RAG)
= 32 cells × ~1000 keywords each = ~32k re-rank LLM calls
```

The headline finding (`T7_source_earned`, the biased-prompt earned-media
demotion): **−1.61\*\*\*** (biased) vs **−0.42\*\*\*** (neutral), pooled. Replicated
across snippet and RAG arms with shrinking gap.

**The reason for the bf16 redo on JUPITER**: snippet cells were originally
processed in 4-bit nf4 on JUWELS (the older default); RAG cells were
full-precision via HF Inference API. To make snippet↔RAG comparisons
identifiable at one precision class, we re-run everything in `bf16-full` on
GH200 GPUs.

Full project context: `docs/long-term-project-arc.md`,
`docs/work-log-2026-05-08.md`, `docs/runbook-full-cycle-2026-05-17.md`.

---

## 2. JUPITER infrastructure cheatsheet

### Login + SSH

```bash
ssh -i ~/.ssh/id_ed25519 fourel1@login.jupiter.fz-juelich.de
# Lands on jpbl-s0X-XX (JUPITER booster login node).
# Two-factor: SSH key passphrase + JSC TOTP code from JuDoor app.
```

> The `from="<ip>/<mask>"` clause on your JuDoor SSH key restricts which IPs
> can use it. If your home/cafe IP changes, edit it in JuDoor → SSH keys →
> Edit. Use `from="X.Y.Z.0/24"` for a forgiving range.

### Filesystem map — the most important JUPITER quirk

**JUPITER login** mounts BOTH `/p/` (JUWELS-shared GPFS) and `/e/` (ExaSTORE,
JUPITER-native).
**JUPITER compute** mounts ONLY `/e/`.

| Path | Login | Compute | Use for |
|---|---|---|---|
| `/p/project1/scifi/`  | ✓ | ✗ | (JUWELS legacy — DO NOT use for new work) |
| `/p/scratch/scifi/`   | ✓ | ✗ | (JUWELS legacy) |
| `/e/project1/scifi/`  | ✓ | ✓ | **Repo, venv, HF cache** |
| `/e/scratch/scifi/`   | ✓ | ✓ | **Dataset, run outputs** |
| `/e/software/default/stages/2026/...` | ✓ | ✓ | Stage Python + modules |

`jutil env activate -p scifi` sets `$PROJECT` and `$SCRATCH` to the **right**
prefix per node — `/p/` on login, `/e/` on compute. So always reference via
the variables when possible.

Your work paths (this run):
- Repo: `/e/project1/scifi/fourel1/GEODML_Analysis/`
- Venv: `/e/project1/scifi/fourel1/GEODML_Analysis/.venv/`
- HF cache: `/e/project1/scifi/fourel1/GEODML_Analysis/hf_cache/` (~459 GB)
- Dataset: `/e/scratch/scifi/fourel1/data/` (~15 GB)
- Run outputs: `/e/scratch/scifi/fourel1/data/runs/<cell>/phase2/keywords.jsonl`

### SLURM accounting

```bash
sacctmgr show association where user=$USER format=Account,Partition%20,QOS%30
# Expected row on JUPITER:
#   scifi   booster   normal
```

If empty → ticket sc@fz-juelich.de (your account isn't yet in SLURM).

Available partitions (via `sinfo`):
- `booster` — 4× GH200 (96 GB HBM each per node, 4 GPUs/node) — what we use
- *(no `develbooster` exists — that's a JUWELS-only convenience)*

QOS time limits: scifi/normal has no MaxWall set in sacctmgr (unlimited on
paper), but our experience shows ≥06:00:00 walltime sometimes triggers
`QOSMaxWallDurationPerJobLimit` from group-level limits. Use `--time=05:55:00`
for safety; chains will keep submitting via afterany dependency.

### Module stack

JUPITER does NOT have `Stages/2024` (that's JUWELS). It has:
- `Stages/2025`
- `Stages/2026` ← **default**, use this

```bash
module load Stages/2026 GCC Python CUDA
which python3    # /e/software/default/stages/2026/software/Python/3.13.5-GCCcore-14.3.0/bin/python3
python3 --version  # Python 3.13.5
```

**Never** use `/usr/bin/python3` (system 3.9) for the venv — it won't have
torch/numpy on compute and will silently break sbatch jobs.

### No internet on compute nodes

JUPITER Booster compute nodes can't reach huggingface.co, pypi.org, or any
external URL. Consequences:
- `pip install` must happen on the **login node** (which has internet).
- `transformers.from_pretrained('meta-llama/...')` will fail unless models
  are pre-cached locally AND `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`
  are set AND `HF_HUB_CACHE` points at the right directory.

---

## 3. One-time setup (already done — keep here for reproducibility)

### 3.1 SSH key on JUPITER

JuDoor → My Profile → SSH keys → upload `~/.ssh/id_ed25519.pub` with
`from="<your.ip>/24"` clause. Wait 15 min for propagation.

### 3.2 Repo + dataset placement on /e/

```bash
# On JUPITER LOGIN (sees both /p/ and /e/)
mkdir -p /e/project1/scifi/$USER /e/scratch/scifi/$USER

# Copy repo (~500 MB)
rsync -a /p/project1/scifi/$USER/GEODML_Analysis /e/project1/scifi/$USER/

# Copy dataset (~15 GB)
nohup rsync -a /p/scratch/scifi/$USER/data /p/scratch/scifi/$USER/*.md \
  /e/scratch/scifi/$USER/ > /tmp/copy.log 2>&1 &
```

If `/e/project1/scifi/<user>/` doesn't exist when you try, `mkdir` it
yourself (the parent dir is writable per project membership).

### 3.3 Venv (build on LOGIN with Stages Python)

```bash
cd /e/project1/scifi/$USER/GEODML_Analysis
rm -rf .venv .venv-*

module purge
module load Stages/2026 GCC Python CUDA

python3 -m venv .venv
source .venv/bin/activate

# Confirm venv uses Stages Python, NOT /usr/bin
cat .venv/pyvenv.cfg
# expect:  home = /e/software/default/stages/2026/software/Python/3.13.5-GCCcore-14.3.0/bin

pip install -U pip wheel
pip install -r requirements.txt
pip install lightgbm doubleml rank_bm25 sentence-transformers textstat accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Sanity:
```bash
python -c "import numpy, torch, transformers, huggingface_hub; print('all imports OK')"
```

### 3.4 HF model cache

Compute nodes can't download from HF. Pre-populate on LOGIN:

```bash
export HF_HOME=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache
mkdir -p "$HF_HOME"

python <<'PY'
from huggingface_hub import snapshot_download
import os
for m in ['meta-llama/Llama-3.3-70B-Instruct',
          'Qwen/Qwen2.5-72B-Instruct',
          'meta-llama/Llama-3.1-8B-Instruct']:
    snapshot_download(m, cache_dir=os.environ['HF_HOME'])
PY
```

~459 GB total, ~30 min on JSC's network. Run in tmux to survive disconnects.

> **The cache-layout gotcha**: `cache_dir=` (above) places models *directly*
> under `HF_HOME` (`hf_cache/models--<repo>/...`). But transformers' default
> resolver expects `HF_HOME/hub/models--<repo>/...`. Fix by setting
> `HF_HUB_CACHE` to point at the `hf_cache` dir directly — see §4.

### 3.5 `.env` file

```bash
cat > /e/project1/scifi/$USER/GEODML_Analysis/.env <<EOF
JUWELS_ACCOUNT=scifi
JUWELS_PROJECT=scifi
HF_TOKEN=hf_xxx_write_scoped_token

# Use /e/ paths (login mounts /e/ too; compute only sees /e/).
GEODML_DATA_ROOT=/e/scratch/scifi/$USER
HF_HOME=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache
HF_HUB_CACHE=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache
HUGGINGFACE_HUB_CACHE=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache

LOCAL_PRECISION=full
MAX_KW=99999

PRIMARY_MODEL=meta-llama/Llama-3.3-70B-Instruct
PROXY_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_DATASET_REPO=ValerianFourel/geodml-papersize
EOF
```

`HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` point at the directory containing
the `models--*` dirs (NOT at its parent). This is what makes the offline
cache resolution actually work.

### 3.6 Patch `scripts/slurm/_common.sh` for the /p→/e translation

`SLURM_SUBMIT_DIR` is captured on the login node when you `sbatch`. If you
submitted from a `/p/` path, the compute node can't `cd` there. Patch
`_common.sh` to translate the path using `$PROJECT`/`$SCRATCH` (set by
jutil on the compute side).

Already applied if `grep "path-translate" scripts/slurm/_common.sh` returns
output. If not, see `docs/runbook-jupiter-2026-05-18.md` for the python
patcher script.

Same script also fixes:
- `module load Stages/2024` → `Stages/2026`
- adds safe defaults for `$SCRATCH`/`$PROJECT` so the translate block doesn't
  error with `unbound variable` when jutil hasn't run yet.

---

## 4. How to send a SLURM job (canonical sbatch invocation)

### Minimum working command

This is the exact invocation that passed smoke on 2026-05-18. **Single line — no backslash continuations** because stray spaces after `\` break it:

```bash
cd /e/project1/scifi/$USER/GEODML_Analysis

sbatch --account=scifi --partition=booster --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=48 --gres=gpu:4 --job-name=geo-smoke-bf16 --output=logs/smoke-%j.out --error=logs/smoke-%j.err --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,ENGINE=searxng,POOL=20,PROMPT_VARIANT=biased,LOCAL_PRECISION=full,MAX_KEYWORDS=5 scripts/slurm/run_rerank.sbatch
```

**Each flag in plain English:**

| Flag | Why |
|---|---|
| `--account=scifi` | bills compute to scifi project |
| `--partition=booster` | the GH200 partition (NOT develbooster — doesn't exist on JUPITER) |
| `--time=00:30:00` | wall limit (smoke = 30 min; full cells use 05:55:00) |
| `--nodes=1 --ntasks=1 --cpus-per-task=48` | 1 node, 1 task, 48 cores |
| `--gres=gpu:4` | all 4 GH200s on the node |
| `--job-name=...` | shows in squeue + log filename pattern |
| `--output=logs/smoke-%j.out` | `%j` = jobid; one file per job |
| `--error=logs/smoke-%j.err` | python tracebacks land here |
| `--export=ALL,...` | export ALL env vars + comma-separated overrides |
| `scripts/slurm/run_rerank.sbatch` | the sbatch script to execute |

### Multi-line version (if you need to script it)

```bash
cat > /tmp/submit.sh <<'BASH'
#!/bin/bash
cd /e/project1/scifi/$USER/GEODML_Analysis
mkdir -p logs
sbatch \
  --account=scifi \
  --partition=booster \
  --time=00:30:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=48 \
  --gres=gpu:4 \
  --job-name=geo-smoke-bf16 \
  --output=logs/smoke-%j.out \
  --error=logs/smoke-%j.err \
  --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,ENGINE=searxng,POOL=20,PROMPT_VARIANT=biased,LOCAL_PRECISION=full,MAX_KEYWORDS=5 \
  scripts/slurm/run_rerank.sbatch
BASH
bash /tmp/submit.sh
```

> Heredoc-in-file is the safe pattern. Inline `\`-continuations in interactive
> bash are fragile — a single trailing space after `\` breaks the whole command
> (then `--export=...` runs as a separate command, which produces the very
> confusing error chain we saw twice during setup).

### Interactive node (for debugging)

```bash
salloc --nodes=1 --gres=gpu:4 --time=02:00:00 --account=scifi -p booster
# Wait for: "Granted job allocation NNN"
srun --cpu-bind=none --nodes=1 --pty /bin/bash -i
# Now on jpbo-XXX-XX — full GH200 node for 2 hours
# Exit with `exit` (twice — once for srun shell, once for salloc)
```

### Verify a submitted job

```bash
# 1. Is it queued/running?
JID=464549   # whatever sbatch printed
squeue -j $JID --format='%.10i %.32j %.2t %.10M %.20S %.30R'

# 2. What happened (after it leaves the queue)?
sacct -j $JID --format=JobID,JobName%30,State,ExitCode,Elapsed,Reason%30

# 3. Did the python actually run?
ls -la logs/smoke-${JID}.*
tail -50 logs/smoke-${JID}.out
tail -50 logs/smoke-${JID}.err

# 4. Did it write data?
ls -la /e/scratch/scifi/$USER/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2/
wc -l /e/scratch/scifi/$USER/data/runs/searxng_Llama-3.3-70B-Instruct_serp20_top10_biased/phase2/keywords.jsonl
```

### Pass criteria in the log

```
[common] HF_HOME=/e/project1/scifi/.../hf_cache
[common] HF_HUB_CACHE=/e/project1/scifi/.../hf_cache    ← required for offline model load
[common] GEODML_DATA_ROOT=/e/scratch/scifi/...
[common] visible GPUs: 4× NVIDIA GH200 120GB
[rerank] cached SERP rows=13,555 keywords=1,009
[load] CUDA_VISIBLE_DEVICES gives 4 GPUs
[LocalRanker] model=... precision=bf16-full                ← THE precision check
[LocalRanker] hf_device_map: ... 4 devices
[LocalRanker]   cuda:N allocated=~32 GiB                   ← model sharded successfully
rerank ...: 5/1009 [00:30, ...]                            ← inference running
[rerank] done: new=5 errors=0
[rerank] python rc=0
```

---

## 5. Issues we hit and how each was fixed

Chronological log of every blocker during setup, exact root cause, exact fix.
Future-Claude can pattern-match here.

### 5.1 `Invalid account or account/partition combination`

**Symptom**: `sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified`

**Diagnosis**:
```bash
sacctmgr show association where user=$USER format=Account,Partition%20,QOS%30
# returned empty rows
```

**Root cause**: User had no SLURM associations on JUWELS. The scifi project
is on JUPITER, not JUWELS (verified via JuDoor → Systems showing
`JUPITER_BOOSTER: scifi`). The obdifflearn project was on JUWELS but
associations got purged after 2026-04-30 CVE-2026-31431 maintenance window.

**Fix**: Switch to JUPITER:
```bash
ssh fourel1@login.jupiter.fz-juelich.de
# Then on JUPITER:
sacctmgr show association where user=$USER format=Account,Partition%20,QOS%30
# returned:   scifi  booster  normal   ✓
```

### 5.2 `/p/project1/scifi/` doesn't exist on compute

**Symptom**:
```
slurmstepd: error: couldn't chdir to `/p/project1/scifi/fourel1/GEODML_Analysis`:
No such file or directory: going to /tmp instead
```

**Root cause**: JUPITER login mounts `/p/` (JUWELS GPFS) but compute nodes
only mount `/e/` (ExaSTORE). `SLURM_SUBMIT_DIR` was captured on login as a
`/p/` path; compute couldn't `cd` there.

**Fix**: Two-part:
1. Move all data/code to `/e/` paths (rsync from `/p/` on login).
2. Patch `_common.sh` to translate `/p/...` → `/e/...` using
   `$PROJECT`/`$SCRATCH` (which jutil sets correctly per node).

### 5.3 `/e/project1/scifi/fourel1/` doesn't exist either

**Symptom**: `ls /e/project1/scifi/fourel1/` returned nothing on compute,
even though other scifi members had their dirs there.

**Root cause**: User's directory under `/e/project1/scifi/` was never
provisioned (or self-creation was needed).

**Fix**: `mkdir -p /e/project1/scifi/$USER` on a login node. Worked
immediately — no admin ticket needed for scifi.

### 5.4 `module load Stages/2024` fails on JUPITER

**Symptom**: `Lmod has detected the following error: The following module(s) are unknown: "Stages/2024"`

**Root cause**: `Stages/2024` is JUWELS-only. JUPITER has `Stages/2025` and
`Stages/2026` (default).

**Fix**:
```bash
sed -i 's|module load Stages/2024 GCC Python CUDA|module load Stages/2026 GCC Python CUDA|' \
  scripts/slurm/_common.sh
```

### 5.5 `SCRATCH: unbound variable` cascade

**Symptom**: `scripts/slurm/_common.sh: line 46: SCRATCH: unbound variable`

**Root cause**: Because `module load Stages/2024` failed (5.4), `jutil env
activate` couldn't set `$SCRATCH`/`$PROJECT`. The path-translate block then
used `$SCRATCH` directly under `set -u` (nounset) and errored out.

**Fix**: Use safe defaults in the path-translate:
```bash
_PROJ="${PROJECT:-/e/project1/scifi}"
_SCR="${SCRATCH:-/e/scratch/scifi}"
if [ ! -d "${SLURM_SUBMIT_DIR:-}" ]; then
  ALT="${SLURM_SUBMIT_DIR:-}"
  ALT="${ALT/#/p/project1/scifi/$_PROJ}"
  ALT="${ALT/#/p/scratch/scifi/$_SCR}"
  if [ -n "$ALT" ] && [ -d "$ALT" ]; then
    SLURM_SUBMIT_DIR="$ALT"; export SLURM_SUBMIT_DIR
  fi
fi
```

### 5.6 venv built with `/usr/bin/python3` (system 3.9) — broken on compute

**Symptom**:
```
File "/usr/lib64/python3.9/runpy.py", ...
ModuleNotFoundError: No module named 'numpy'
```

**Root cause**: When `module load Stages/2024` silently failed, the
`python3 -m venv .venv` step used `/usr/bin/python3` (3.9.25). Its
site-packages live at `/usr/lib64/python3.9/site-packages` (not the venv).
On compute, neither location had numpy. Also Python 3.9 doesn't have the
new features used by modern transformers.

**Fix**: Rebuild venv on LOGIN with the Stages Python:
```bash
rm -rf .venv
module purge
module load Stages/2026 GCC Python CUDA
which python3    # /e/software/default/stages/2026/.../python3 (3.13.5)
python3 -m venv .venv
cat .venv/pyvenv.cfg   # confirm home = /e/software/...
source .venv/bin/activate
pip install -r requirements.txt
pip install lightgbm doubleml rank_bm25 sentence-transformers textstat accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Stage Python's binary at `/e/software/default/stages/2026/...` IS visible
from compute (same /e/ mount), so the venv is portable.

### 5.7 `pip install` fails on compute with `Network is unreachable`

**Symptom**: From inside salloc on compute:
```
WARNING: ... NewConnectionError ... Network is unreachable
ERROR: No matching distribution found for huggingface_hub>=0.24
```

**Root cause**: JUPITER Booster compute nodes have no outbound internet.

**Fix**: Never run `pip install` on compute. Always do it on a login node.

### 5.8 `OSError: We couldn't connect to 'https://huggingface.co'` at runtime

**Symptom**: At sbatch runtime, `LocalRanker.__init__` errors loading the model:
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

**Root cause**: HF cache layout mismatch. We downloaded models with
`snapshot_download(cache_dir=$HF_HOME)`, which puts them directly at
`$HF_HOME/models--<repo>/`. But transformers' default resolver looks for them
at `$HF_HOME/hub/models--<repo>/`.

**Fix**: Tell transformers where to look by setting `HF_HUB_CACHE` and
`HUGGINGFACE_HUB_CACHE` to point AT the `hf_cache/` dir (containing
`models--*` subdirs):

```bash
# Add to .env
echo "HF_HUB_CACHE=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache" >> .env
echo "HUGGINGFACE_HUB_CACHE=/e/project1/scifi/$USER/GEODML_Analysis/hf_cache" >> .env
```

(Also possible: restructure with `mkdir hub && mv models--* hub/`. The env
var approach is non-destructive.)

`_common.sh` already exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
so the network attempt is suppressed once the cache resolution works.

### 5.9 `--export=ALL,...` forwards LITERAL /p/ paths

**Symptom**: Job runs on compute, but log shows
`outdir=/p/scratch/scifi/...` (not `/e/`). Then python rerank fails because
`/p/` doesn't exist on compute.

**Root cause**: `${SCRATCH}/fourel1` in `.env` was expanded at
`set -a; source .env` time on LOGIN, where `$SCRATCH=/p/scratch/scifi`. The
literal expanded value was carried via `--export=ALL` to compute.

**Fix**: Use **literal** `/e/...` paths in `.env`:
```ini
GEODML_DATA_ROOT=/e/scratch/scifi/fourel1
HF_HOME=/e/project1/scifi/fourel1/GEODML_Analysis/hf_cache
```

Both `/e/scratch` and `/e/project1` ARE mounted on login too, so the literal
`/e/` paths work everywhere. No dynamic expansion needed.

### 5.10 `QOSMaxWallDurationPerJobLimit` on chain-resubmit

**Symptom**:
```
sbatch: error: QOSMaxWallDurationPerJobLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy
```

**Root cause**: The `chain_resubmit` helper in `_common.sh` re-submits with
`--time=23:55:00` (the JUWELS default). scifi/normal QOS on JUPITER has a
tighter cap from group limits.

**Fix**: Lower the sbatch wall limit:
```bash
sed -i 's/#SBATCH --time=23:55:00/#SBATCH --time=05:55:00/' \
  scripts/slurm/run_*.sbatch
```

`05:55:00` is enough for ~1400 keywords/cell at ~15s/keyword. If you hit
walltime mid-cell, the chain auto-resubmits with `--resume` and picks up where
it stopped (rerank.py honors the existing JSONL).

### 5.11 Bash `\` line-continuation followed by stray space

**Symptom**:
```bash
sbatch --account=scifi --time=01:00:00 \    # ← stray space after \
    --gres=gpu:4 ...
# Errors:
-bash: --gres=gpu:4: command not found
-bash: --export=ALL,MODEL=...: No such file or directory
```

**Root cause**: Bash treats `\<space><newline>` as `\<space>` (literal
backslash-space). The newline DOESN'T continue the line. So `--gres=...`
runs as its own command on the next line.

**Fix**: Either use a single-line invocation (the canonical pattern in §4),
or write the sbatch in a heredoc file and execute via `bash file.sh`.

### 5.12 Python `python -c` with leading whitespace

**Symptom**:
```bash
python -c "
  import torch
  ...
"
IndentationError: unexpected indent
```

**Root cause**: `python -c` doesn't strip leading whitespace like docstring
dedent. Indented heredoc bodies break Python parsing.

**Fix**: Either write code with NO leading whitespace, or use a `python <<'PY'` heredoc and write code at column 0:
```bash
python <<'PY'
import torch
print(torch.cuda.device_count())
PY
```

### 5.13 f-string backslash escape in Python 3.12+

**Symptom**:
```python
print(f'{v} {df[\"col\"].value_counts()}')
SyntaxError: f-string expression part cannot include a backslash
```

**Root cause**: Python ≥3.12 forbids `\` inside f-string `{}`. The `\"`
escapes inside the f-string expression aren't valid.

**Fix**: Use single quotes outside, double inside (no escape needed):
```python
print(f'{v} {df["col"].value_counts()}')
```

### 5.14 Setup commit too large for GitHub (611 MB push)

**Symptom**: `git push origin main` errors with `error: RPC failed; HTTP 400`.

**Root cause**: `git add .` picked up `archives/` (606 MB zips), `logs/`,
and other large dirs.

**Fix**:
```bash
git reset --soft HEAD~1     # uncommit, keep changes staged
git restore --staged .      # unstage everything
# Add big dirs to .gitignore
cat >> .gitignore <<'EOF'
archives/
logs/
hf_cache/
.venv/
.venv-*/
geodml_data/
*.zip
*.tar.gz
*.npy
*.parquet
*.csv
*.jsonl
EOF
# Selectively re-add only code
git add .gitignore interpretability/ scripts/ docs/
git commit -m "feat: ..."
git push origin main          # ~80 KB now
```

---

## 6. Current state (2026-05-18)

### ✅ Working

- JUPITER login (login.jupiter.fz-juelich.de)
- scifi account / booster partition / normal QOS
- Repo + venv + HF cache + dataset all on `/e/`
- Smoke test (5 keywords, Llama-3.3-70B bf16, sharded across 4× GH200): **PASSED**
  - Allocated 32-33 GB per GPU
  - `precision=bf16-full` in records
  - `[rerank] done: new=5 errors=0`

### 🚧 Pending

1. **Clear 4-bit snippet cells** (so full launch re-processes everything in bf16):
   ```bash
   cd /e/project1/scifi/fourel1/GEODML_Analysis && set -a; source .env; set +a
   for V in biased neutral ; do
     for D in $GEODML_DATA_ROOT/data/runs/*_top10_${V}/phase2 ; do
       [ -f "$D/keywords.jsonl" ] && mv "$D/keywords.jsonl" "$D/keywords.jsonl.4bit-bak-$(date +%Y%m%d)"
       rm -f "$D/.rerank_ckpt.json"
     done
   done
   ```

2. **Submit full campaign** (~96 sbatch jobs, ~3-5 days wall time):
   ```bash
   tmux new -s gpurun
   INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 \
     ./scripts/finish_on_gpu.sh 2>&1 | tee logs/finish_on_gpu_$(date +%Y%m%d_%H%M%S).log
   ```
   *Note: `finish_on_gpu.sh` was written for JUWELS layout. May need partition
   tweaks. If it errors with "invalid partition develbooster" → already fixed
   via sed, but verify with `grep develbooster scripts/`.*

3. **Standardize after all jobs complete** (backfill precision, audit):
   ```bash
   .venv/bin/python scripts/backfill_precision.py --root "$GEODML_DATA_ROOT" --include-recent
   .venv/bin/python scripts/audit_status.py | tail -30
   ```

4. **DML + figures + summary**:
   ```bash
   bash scripts/continue_pipeline.sh
   .venv/bin/python scripts/make_results_summary.py --data-root "$GEODML_DATA_ROOT"
   ```

5. **Push to HF**:
   ```bash
   hf upload-large-folder ValerianFourel/geodml-papersize "$GEODML_DATA_ROOT" \
     --repo-type dataset --num-workers 1 \
     --exclude "data/runs/*/phase2/html_cache/**" \
     --path-in-repo data
   ```

### 🔍 Known throughput limit

Smoke showed **~15.30 s/keyword** for Llama-3.3-70B-Instruct bf16 at
`temperature=0.1, max_tokens=500, top_p=1, do_sample=True` with ~3k input
tokens. Math:
- 1 cell × ~1000 keywords × 15s = ~4.2 hours
- 32 cells × ~4.2 hours sequential = ~5.6 days

In practice, multiple sbatch jobs run in parallel (queue permitting), so
wall time is ~3-5 days for the full campaign. Qwen2.5-72B is somewhat slower
per-step due to MoE-style mix.

If throughput becomes a blocker:
- Reduce `max_tokens` from 500 → 250 (output is just a ranked list, rarely
  needs more than ~150 tokens). Halves the per-keyword time.
- Use `do_sample=False` for greedy decoding. May change the output
  distribution; benchmark before adopting.
- Use FlashAttention-2 (if not already enabled). Check
  `model.config._attn_implementation` after load.

---

## 7. Quick reference

### Essential paths

```
Login:          ssh fourel1@login.jupiter.fz-juelich.de
Repo:           /e/project1/scifi/fourel1/GEODML_Analysis
Venv:           /e/project1/scifi/fourel1/GEODML_Analysis/.venv
HF cache:       /e/project1/scifi/fourel1/GEODML_Analysis/hf_cache
Dataset:        /e/scratch/scifi/fourel1/data
Run outputs:    /e/scratch/scifi/fourel1/data/runs/<cell>/phase2/
HF repo:        ValerianFourel/geodml-papersize
GitHub:         https://github.com/ValerianFourel/GEODML_Analysis
```

### Essential commands

```bash
# Login
ssh -i ~/.ssh/id_ed25519 fourel1@login.jupiter.fz-juelich.de

# Setup shell
cd /e/project1/scifi/fourel1/GEODML_Analysis
jutil env activate -p scifi
module load Stages/2026 GCC Python CUDA
source .venv/bin/activate
set -a; source .env; set +a

# Submit one rerank cell (replace VARIANT, ENGINE, POOL, MODEL)
sbatch --account=scifi --partition=booster --time=05:55:00 --nodes=1 --ntasks=1 --cpus-per-task=48 --gres=gpu:4 --job-name=geo-rerank --output=logs/rerank-%j.out --error=logs/rerank-%j.err --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,ENGINE=searxng,POOL=20,PROMPT_VARIANT=biased,LOCAL_PRECISION=full scripts/slurm/run_rerank.sbatch

# Submit all 96 jobs
tmux new -s gpurun
INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 ./scripts/finish_on_gpu.sh 2>&1 | tee logs/finish_$(date +%s).log

# Monitor
squeue -u $USER --format='%.10i %.32j %.2t %.10M %.20S'
sacct -u $USER -S now-24hours -X --format=JobID,JobName%30,State,ExitCode,Elapsed | head -40
tail -f logs/$(ls -t logs/ | head -1)

# Diagnose failed job
sacct -j <JID> --format=JobID,JobName%30,State,ExitCode,Reason%30
cat logs/<jobname>-<JID>.out
cat logs/<jobname>-<JID>.err

# Cancel everything
scancel -u $USER

# Interactive node for debugging
salloc --nodes=1 --gres=gpu:4 --time=02:00:00 --account=scifi -p booster
srun --cpu-bind=none --nodes=1 --pty /bin/bash -i
```

### Env var reference

| Var | Required value | Set in |
|---|---|---|
| `JUWELS_ACCOUNT` | `scifi` | `.env` (export via `set -a; source .env`) |
| `JUWELS_PROJECT` | `scifi` | `.env` |
| `HF_TOKEN` | `hf_...` write-scoped | `.env` |
| `GEODML_DATA_ROOT` | `/e/scratch/scifi/fourel1` | `.env` (literal /e/, not ${SCRATCH}) |
| `HF_HOME` | `/e/project1/scifi/fourel1/GEODML_Analysis/hf_cache` | `.env` |
| `HF_HUB_CACHE` | same as `HF_HOME` | `.env` (the critical fix from 5.8) |
| `HUGGINGFACE_HUB_CACHE` | same as `HF_HOME` | `.env` |
| `LOCAL_PRECISION` | `full` | `.env` (compiles to `bf16-full`) |
| `MAX_KW` | `99999` | `.env` (sentinel for "no cap") |
| `HF_HUB_OFFLINE` | `1` | exported by `_common.sh` |
| `TRANSFORMERS_OFFLINE` | `1` | exported by `_common.sh` |
| `SLURM_SUBMIT_DIR` | translated to `/e/` by `_common.sh` | auto by sbatch |
| `PROJECT`, `SCRATCH` | `/e/project1/scifi`, `/e/scratch/scifi` | auto by `jutil env activate` on compute |

### Per-cell sbatch env (passed via `--export=ALL,...`)

| Var | Example value |
|---|---|
| `MODEL` | `meta-llama/Llama-3.3-70B-Instruct` or `Qwen/Qwen2.5-72B-Instruct` |
| `ENGINE` | `searxng` or `ddg` |
| `POOL` | `20` or `50` |
| `PROMPT_VARIANT` | `biased`, `neutral`, `biased_rag`, `neutral_rag` |
| `MAX_KEYWORDS` | omit for full (no cap), or `5` for smoke |
| `SEED` | required for order_probe: `42` or `123` |

---

## 8. When you're stuck — diagnostic flowchart

```
Job failed/missing/produced wrong output?
  ↓
sacct -j <JID> --format=State,ExitCode,Elapsed,Reason
  ↓
State = ?
├─ PENDING/RUNNING → just wait
├─ COMPLETED + ExitCode=0:0 → success; check data on disk
├─ FAILED + ExitCode=1:0 → cat logs/*<JID>*.err (most common)
├─ OUT_OF_MEMORY → drop --top-k-rag or use bigger node
├─ TIMEOUT → increase --time, but check QOSMaxWall
├─ CANCELLED → you scancel'd it, or admin did
├─ NODE_FAIL → just resubmit, transient
└─ Invalid account/partition → check sacctmgr show association

logs/*.err shows what?
  ↓
├─ ModuleNotFoundError: numpy → venv broken, see 5.6
├─ OSError: couldn't connect to huggingface.co → cache layout, see 5.8
├─ slurmstepd: couldn't chdir → /p/ submit dir, see 5.2
├─ Lmod: module not found → stage name wrong, see 5.4
├─ SCRATCH: unbound variable → cascade from 5.4, see 5.5
├─ Network is unreachable → ran pip on compute, see 5.7
├─ torch.cuda.OutOfMemoryError → reduce model size or --top-k-rag
└─ Unable to open file → bash continuation, see 5.11

sbatch rejected the submission?
  ↓
├─ Invalid account → see 5.1
├─ Invalid partition → use 'booster' not 'develbooster' (JUWELS-only)
├─ QOSMaxWallDurationPerJobLimit → see 5.10
└─ Unable to open file (blank) → bash continuation, see 5.11
```

---

## 9. Useful URLs

- JuDoor (project + SSH management): https://judoor.fz-juelich.de
- JUPITER docs: https://apps.fz-juelich.de/jsc/hps/jupiter/
- JUPITER status: https://go.fzj.de/status-jupiter
- llview job reports: https://go.fzj.de/llview-juwelsbooster (also has JUPITER)
- JSC support email: `sc@fz-juelich.de`

---

## 10. Glossary of project-specific terms

- **GEODML** — the experiment (DML over LLM rerank features).
- **Stage A** — rerank phase (model produces a ranking per keyword).
- **Stage A'** — order_probe (re-rank with shuffled candidates to test
  position sensitivity).
- **Stage B** — feature extraction from page HTML (deterministic, no LLM).
- **Stage C** — merge SERPs + ranks + features → single Stage C parquet.
- **Stage D** — DoubleML estimation: per-treatment causal effect on
  `rank_delta`.
- **Stage F** — mechanistic interp: ablation, saliency, probing, weights
  (uses an 8B PROXY model, not the 70B).
- **Snippet variant** — `biased`, `neutral`: LLM sees only title + 150-char
  snippet per result.
- **RAG variant** — `biased_rag`, `neutral_rag`: per-result `passage:` field
  filled with top-3 retrieved chunks from page body (~2400 chars).
- **Cell** — one `(model, engine, pool, variant)` combination = one sbatch job.
- **Precision regime** — `bf16-full`, `4bit-nf4`, `api-hf`, `api-openai` —
  captured per record in `llm_parameters.precision`.

---

End. See companion docs:
- `docs/runbook-jupiter-2026-05-18.md` — same content as a step-by-step runbook.
- `docs/runbook-full-cycle-2026-05-17.md` — Mac↔JUPITER↔HF full cycle.
- `docs/long-term-project-arc.md` — what the experiment is trying to prove,
  what the paper claim is, what comes after EMNLP.
