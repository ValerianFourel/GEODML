# Repair pipeline — how to fill data gaps on JUPITER

*2026-05-23.* Resumable two-script flow that audits keyword-level gaps and resubmits the missing cells.

## What it covers

| Stage | Template | Per-cell axes |
|---|---|---|
| `rerank` | `scripts/slurm/run_rerank.sbatch` | variant × engine × pool × model |
| `order_probe` | `scripts/slurm/run_order_probe.sbatch` | variant × engine × pool × model × seed |
| `probing` | `scripts/slurm/run_probing.sbatch` | variant × model |

The ground-truth keyword set comes from the **SERP pool parquets** (`data/serp/phase0_top{20,50}_{ddg,searxng}.parquet`). For each cell, the audit computes:

- `target_kw` — unique keywords in the canonical SERP pool for that (engine, pool) combination.
- `actual_kw` — unique keywords with completed output for that cell.
- `gap = target_kw − actual_kw`.

For RAG variants, the gap captures keywords that originally failed retrieval. Re-running the cell with `--resume` will retry those keywords from scratch (rerank.py / order_probe.py both honour `--resume` and append new keywords).

## Quick start on JUPITER

```bash
cd /e/project1/scifi/fourel1/GEODML_Analysis
jutil env activate -p scifi
set -a; source .env; set +a

# 1) Audit everything — produces manifests/repair_manifest.parquet
.venv/bin/python scripts/repair_audit.py

# 2) Submit jobs for cells with gap > 0 (max 32 per invocation by default)
.venv/bin/python scripts/repair_dispatch.py

# 3) Wait, then re-run both — audit re-reads files (closes DONE cells),
#    dispatch polls SLURM (closes RUNNING cells) and resubmits anything still missing.
.venv/bin/python scripts/repair_audit.py
.venv/bin/python scripts/repair_dispatch.py
```

Repeat steps 3 every few hours until the audit shows zero `gap > 0` cells.

### Just check status without submitting

```bash
.venv/bin/python scripts/repair_dispatch.py --status
```

### Dry-run before submitting

```bash
.venv/bin/python scripts/repair_dispatch.py --dry-run
```

### One stage at a time

```bash
.venv/bin/python scripts/repair_dispatch.py --stage probing --max-submissions 8
```

## How resumability works

- `repair_audit.py` writes / refreshes `manifests/repair_manifest.parquet`. Each row is `(stage, variant, engine, pool, model, seed, run_id, target_kw, actual_kw, gap, status, last_jobid, last_submitted, last_check)`. When re-run, it preserves `status / last_jobid` for cells you already submitted, recomputes `actual_kw / gap` fresh, and auto-marks `status=DONE` for any cell where `gap == 0`.
- `repair_dispatch.py`:
  1. polls `squeue -j <jobid>` for every `SUBMITTED/RUNNING` row; any whose jobid is no longer in the queue is marked `PENDING_RECHECK` (the next audit run finalises it as `DONE` or `FAILED`).
  2. submits `sbatch` for cells in `{TODO, FAILED, PENDING_RECHECK}` with `gap > 0`, biggest gaps first, up to `--max-submissions`.
  3. persists the manifest after each cycle, so a Ctrl-C in the middle just leaves the partial state and the next invocation continues.

If the cluster scheduler delays acceptance, the underlying sbatch `chain_resubmit` self-resubmits with `afterany` dependency. Each underlying job uses `--resume`, so partial keywords.jsonl is appended, not overwritten.

## Status reference

| status | meaning |
|---|---|
| `TODO` | never submitted via this manifest |
| `SUBMITTED` | `sbatch` accepted; jobid recorded |
| `RUNNING` | jobid present in `squeue` |
| `PENDING_RECHECK` | jobid no longer in `squeue`; next audit decides DONE / FAILED |
| `DONE` | audit confirmed `gap == 0` |
| `FAILED` | sbatch rejected the submission |

## What this does NOT fix

- **Keywords that always fail RAG retrieval** (e.g. backend timeouts on certain queries). These will stay in `gap > 0` after multiple cycles. After 2–3 retries, inspect `logs/op-*-*.err` for the underlying error and decide whether to (a) blacklist the keyword, (b) widen the retrieval timeout, or (c) document them as known-failures in the paper limitations.
- **Stage B–D rebuilds.** Once Stage A is complete, run `scripts/build_unified_main.py` (or the existing `unify_precision.py`) to rebuild `data/main/full_experiment_data_{variant}.parquet`, then `scripts/dml_summary.py` for Stage D. Those are not in the repair scope because they're fast and not job-shaped.
- **The unified.parquet RAG bug** — fixed separately by `scripts/bridge_dataset_gaps.py` (see `docs/dataset_gap_bridge_2026-05-23.md`).

## Throughput estimate

Per cell at bf16 on 4× GH200: ~5 min model load + ~2–4 ms/keyword → 8–25 min wall for 1000 keywords. With JUPITER's `booster` partition holding 16–32 concurrent jobs typically, a full grid resubmit (96 cells) finishes inside one wall-clock day. Probing is more expensive (~5 h per cell).

## Auditing locally (Mac)

You can run the audit script against the HF dataset mirror:

```bash
GEODML_DATA_ROOT=$HOME/geodml_data .venv/bin/python scripts/repair_audit.py --print-only
```

This won't write the manifest but tells you what's missing in the published dataset (useful before pushing).
