# HF reconciliation — Verify phase (2026-07-20)

Completes the reconciliation started 2026-07-18 (workflow `wf_521a9555-859`, Compare
phase cached in its journal). Every claimed remote-vs-local mismatch was re-derived
from scratch: remote bytes downloaded (anonymously — all three repos are public),
sha256 both sides, row counts for parquets. Full evidence:
`verify_report.json` in the session scratchpad; per-file compare JSONs
(`emnlp2026_compare.json`, `papersize_compare.json`, `reviewer_compare.json`)
copied forward from the 07-18 session.

## Verdict per claim

### 1. `geodml-emnlp-2026` vs `~/geodml-emnlp-2026` (the 1.8 GB bundle) — REAL, local newer
| file | remote | local | verdict |
|---|---|---|---|
| `scripts/make_fig_saliency.py` | 5,889 B, sha `a7cdfd15…` | 5,889 B, sha `37fff406…` | REAL — 12-line diff |
| `scripts/build_hf_dataset.py` | 26,591 B | 34,819 B | REAL — 199-line diff |

Both local copies are **byte-identical to the current GEODML_Analysis working tree**
(which still shows them as uncommitted modifications). The final HF commit
(2026-05-25 18:14 UTC) only added README/docs and never carried these two script
updates. **Action: commit the two scripts in git, push both to the HF repo.**

### 2. `geodml-emnlp-2026` vs `GEODML_Analysis/geodml_data` — REAL but expected (era gap)
Sampled README.md, `regression_dataset.parquet`, one order-probe jsonl: all
genuinely differ (local mtimes 2026-04-26 … 05-07). `geodml_data/` is a mirror of a
**predecessor** dataset (May-07 era) and predates the repo's creation (May-22).
Remote is newer and canonical on every sampled file. **No sync action** — treat
`geodml_data/` as historical, not as a mirror of this repo.

### 3. `geodml-papersize` overlap: 227 claims → **202 artifacts, 25 real**
The compare agent matched files by basename+size across directories, so 202 of the
227 "mismatches" compared a remote file from one experimental cell against a local
file from a *different* cell (e.g. remote `saliency_Llama…biased_passage/…full.csv`
vs local `saliency_Qwen…neutral/…full.csv`). Spot-downloads confirmed: hash differs
because the cells differ. **All 9 claimed hash-mismatches are such artifacts** —
dismissed, no data corruption anywhere.

The 25 genuine same-path divergences:
- **4× `data/main/full_experiment_data_{biased,neutral,biased_rag,neutral_rag}.parquet`**
  vs `~/geodml-emnlp-2026` — REAL. Identical row counts (96,778 / 125,613 /
  103,073 / 106,392) but different bytes: papersize carries the **May-17-era
  Stage-C inputs**, the bundle carries the **May-24 paper-final rebuild** (larger:
  added columns from the bridge fix). No data loss — the paper-final versions are
  safely on HF in `geodml-emnlp-2026`. Papersize's README/MANIFEST must state its
  data era so nobody mistakes its `data/main` for the paper inputs.
- **21 vs `geodml_data/`** (May-07 mirror): README, 16 hidden `.…_passage_…_ckpt.json`
  order-probe checkpoints, `order_probe_summary.parquet`, 4 searxng
  `html_cache.tar.gz` (remote tarballs are the fuller, newer caches — e.g.
  918 MB vs 404 MB local). Same era-gap story as claim 2. No action.

### 4. `geodml-emnlp-2026-reviewer` — perfect sync
48/48 files match by hash; only local extra is `.DS_Store`. No claims to verify.

## Standing discoveries (from the cached Compare, re-confirmed today)
- **`geodml-emnlp-2026` is ALREADY PUBLIC** (API `private: False`) — plan step
  "flip public" is moot. Secret scan was clean (only the author's own email).
- Remote bloat in `geodml-emnlp-2026`: 167 stale pre-refactor files (3.89 GB,
  `interpretability/output/**`, old `data/dml_results/*`, `MANIFEST.json`, …)
  contradict the README's 1.8 GB claim → delete from remote.
- Papersize is internally complete: 44 run dirs = full 8-cell grid × 5 variants
  + 4 legacy dirs; every base cell has a complete `html_cache.tar.gz`. The
  "45 run dirs" in DATA_POINTERS was likely a miscount or an archived `_passage`
  dir. Regenerated MANIFEST content: `papersize_manifest_regen.json`
  (19 groups, 41,882 files, 37,599,189,390 bytes).
- Reviewer pack: published README was hand-expanded AFTER the build —
  **port it back into `build_reviewer_pack.py` before any rebuild**, else the
  rebuild regresses it. None of the 16 `rebuttal/out/` artifacts are in the pack yet.
- Both HF tokens from 07-18 (`horeka_read`, `juwels_write`) are still valid.

## Updated plan
1. Commit/push `rebuttal/` + untracked docs + the two modified scripts in GEODML_Analysis.
2. `geodml-emnlp-2026` hygiene: push the 2 updated scripts, delete the 167 stale
   files (3.89 GB), verify README size claim then holds. (Already public — nothing to flip.)
3. Papersize hygiene: upload regenerated MANIFEST.json + fresh AUDIT/README
   (state the May-17 era of `data/main`; point to `geodml-emnlp-2026` for paper-final).
4. Reviewer pack: port published README into `build_reviewer_pack.py`, add
   rebuttal section, rebuild, re-upload.
5. Monorepo unification (unchanged).
