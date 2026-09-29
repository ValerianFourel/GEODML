# Leaving JUPITER: handoff for tomorrow (2026-09-30)

Condensed follow-up to [the Leave JUPITER log](2026-09-29_leave-jupiter_handoff.md),
which holds the detailed history, pasted evidence and every intermediate
decision. HoreKa and Nemotron work continues separately in
[the Nemotron handoff](2026-09-29_nemotron-tomorrow_handoff.md).

Valerian's professor asked him to stop using JUPITER. Everything still on JUPITER
that the publication needs has to reach Hugging Face before access ends.
**Deadline:** JSC deletes from `$FSCRATCH` (files older than 30 days) and
`$SCRATCH` (older than 90 days) starting **2026-10-01**. `/e/project1` is not purged.
The publication is the **new, improved version** (Experiment V2 on the GeoAxis
readiness axis); the earlier EMNLP/DML paper is finished and already public.

## 1. Done and verified

| What | Where | Evidence |
| --- | --- | --- |
| Llama generator results: 310,374 of 310,380 cells complete, 6 terminal failures to finish later elsewhere | `geodml-experiment-v2-paper-private` (registry and exchange) | `LLAMA_FULLY_DONE=yes`; round 8 (job 2102836) published, `already_synced 56` |
| Qwen JUPITER results: 24,218 cells | same repo, `coordination/qwen-results.json` | `new_cells 0` |
| 26k prompt population: prompts, axis map, final Qwen/Mistral embeddings of the 26,009 prompts, maps, battery, plan, final checkpoint (~740k candidates and validation), registration, selection manifest, pointers, global-merge provenance | private `ValerianFourel/geoaxis-prompts-generation-26k` | `GEOAXIS_26K_COMPLETE 271 files, 9.66 GB`; contract hashes `1718321f…`/`43189f68…` match |
| Axis construction: subspace root (bundle in both scopes, embeddings, maps, robustness, comparisons, confirmations), task bank and codebook, packed judge queue (85,955 files), corpus, acquisition runs (manifests/logs), analyses | private `ValerianFourel/geoaxis-readiness-axis` | `GEOAXIS_AXIS_COMPLETE 952 files, 1.28 GB` |
| Compute and storage accounting summary | Mac: `~/Hamburg/GEODML_Unified/jupiter-archive/slurm-accounting-2026-summary.md` | 13,830 jobs, 3,966 node-h, 15,862 GPU-h; 1,134,232 core-h = 1.25 % of SCIFI use (72 % of an equal share for the whole period) |

The restore checks (download and hash compare) for both GeoAxis repos have not
been run yet. They are optional.

Owner decisions recorded in the READMEs and the detailed log:
- The restricted-local final embeddings, the maps and the WildChat
  restricted-scope rows (3,200) are in the private repos, against the runbook rule.
- The EMNLP probing outputs are to be public.

## 2. In flight or to start (Valerian runs these; paste output back)

Both are on the Mac, open in Safari, and assume the HF token is already
exported in the shell. They run in two JUPITER login shells on the same node
(the last used node was `jpbl-s03-01`); tmux sessions only show on the node that
started them.

**`finish-shell-a.html`: general archive** to private
`ValerianFourel/geodml-jupiter-archive-private`, with pinned archiver `062f06a`
(`analysis/scripts/archive_jupiter_to_hub.py`).

1. Plan (30–60 min). Roots are uploaded in this order: home, fscratch-scifi,
   scratch-scifi, project1-scifi. Only files ≤50 MB go in, and no
   `restricted-local` paths. It excludes:
   - caches, venvs and model downloads
   - `geodml/src` checkouts
   - `compile-cache` and `jupiter-llama-*`
   - Llama wave data shards, which are already on the Hub
   - EMNLP `html_cache` (on the Hub as tarballs)
   - `probing_*/*_chunks_*` (Shell B uploads those)
   - `GEODML_Analysis/hf_stage`
   - tokens and files containing tokens

   **Expect `PLAN_READY` around 50–100 GB. Review the output before uploading**,
   including the names of files that mention LMSYS or WildChat. If they are
   acceptable, set `ACCEPT=--accept-restricted` in the upload block.
2. Upload in tmux `hf-archive`: attach, paste the token, detach. Units of about
   10 GB are tar.gz-verified on the Hub and resumable.
3. Watch until `ARCHIVE_COMPLETE=yes`.

The archive is insurance, not a paper requirement. It holds provenance
(logs, manifests, receipts, ledgers), the per-job accounting TSV
(`audits/slurm-accounting/`), the pilots including the ACL ARR document pilot
(70 MB, exported nowhere else) and `pilot-3` (37 GB, 770k files), the unpublished
AxisGEO exports, the late EMNLP results (order-probe top-off 0.6 GB, probing,
ablation and saliency summaries), and the untracked May repair reports.

**`finish-shell-b.html`: public EMNLP interpretability mirror**, approved by
Valerian on 30 Sep.

1. Start tmux `emnlp-activations`. It uploads all of
   `/e/project1/scifi/fourel1/GEODML_Analysis/interpretability/output`
   (551.7 GB: probing `.npz` activation chunks, about 2,000 GPU-h to recompute,
   plus all Stage F results) to the new public dataset
   `ValerianFourel/geodml-emnlp-2026-probing-activations`.
   - `SCOPE=all` takes about 1.5–3 days.
   - `SCOPE=without-t7-full` takes about 80 GB and 5–12 h.
   - It uses `upload_large_folder` with 4 workers and is resumable. It ends with
     a per-file size comparison: `ACTIVATIONS_UPLOAD_COMPLETE`, or
     `ACTIVATIONS_INCOMPLETE` (then rerun to resume).
   - Hugging Face may want to be told about a public repository this large.
2. Watch.

## 3. Tomorrow, in order

1. Check both shells: Shell A step 3 and Shell B step 2. If the Shell A plan is
   still waiting for review, review it, then start the upload.
2. When `ARCHIVE_COMPLETE=yes` and `ACTIVATIONS_UPLOAD_COMPLETE` are both shown,
   run `leave-jupiter.html` step 9 (remove stored tokens:
   `$BASE/huggingface-acl-arr/token`, `~/.cache/huggingface/token`,
   `~/.git-credentials`), then step 3, the checklist. Want `LEAVE_READY=yes`.
   Revoke the JUPITER HF token on huggingface.co; create a separate HoreKa token
   first if the two share one.
3. Optional, still open:
   - **Locate the 1.5 TB on `/e/scratch`:** `jutil` reports about 1.5 TB on
     `exa_scratch` (last updated 2026-09-21), but only 26 GB was found under
     `/e/scratch/scifi/fourel1`. Scratch is purged too. This command is read-only:
     ```
     for d in $(find /e/scratch -maxdepth 3 -user "$USER" -type d 2>/dev/null | head -40); do timeout 600 du -sh "$d" 2>/dev/null; done | sort -h | tail -20
     ```
   - Restore checks: `geoaxis-26k.html` step 4, `geoaxis-readiness-axis.html` step 4.
   - Courtesy cleanup of project1 (about 1.8 TB, mostly `hf_cache` 429 GB,
     `geodml/models` 343 GB, `.venv`, interpretability) after everything is
     verified. No page exists yet. It needs a confirm-each-step design, and
     nothing may be deleted before the Hub copies are verified.
4. Follow-ups that are not tied to JUPITER:
   - The 6 Llama straggler cells (terminal failures, packages blocked).
   - AGENTS.md still names JUPITER as owner of the frozen population, the
     reference timings and plan republication. Moving these roles to HoreKa is a
     reviewed change still to make.

## 4. Code, pins and pages

- Branch `threehour-relaunch-fix`, pushed to `origin/codex/pilot-continuation`
  at `750d309` and this handoff's commit.
- Archivers, all with fake-Hub tests (12 archive tests pass):
  - `analysis/scripts/archive_jupiter_to_hub.py`
  - `analysis/scripts/archive_geoaxis_prompts_26k.py`: shared staging, manifest
    and private-upload helpers
  - `analysis/scripts/archive_geoaxis_readiness_axis.py`: `--group`,
    `--manifests-only`, `--pack` and `--pack-manifests`
- Pins checked out on JUPITER: `062f06a` (general archiver and 26k) and
  `9e4b2a0` (axis), under `$BASE/src/archive-<sha>`.
- Mac pages in `~/Hamburg/GEODML_Unified/`:
  - `finish-shell-a.html`, `finish-shell-b.html`: the current minimal pages
  - `leave-jupiter.html` (full steps 0–9), `geoaxis-26k.html`,
    `geoaxis-readiness-axis.html`, `jupiter-publication-audit.html`
- The page builders live in the session scratchpad only. The command blocks are
  embedded in the HTML, and the scripts they run are committed.
- JUPITER outputs:
  - `$BASE/audits/leave-jupiter/`: staging trees, run and upload logs,
    receipts, `hf-archive/`
  - `$BASE/audits/publication-audit/`: hub comparison, interp and runs reports,
    activation upload logs
  - `$BASE/audits/slurm-accounting/`
- `$BASE` = `/e/fscratch/scifi/fourel1/geodml`.

## 5. Facts worth remembering

- The Hub refuses more than 10,000 files in one folder. The archivers pack
  file-heavy folders as tarballs.
- Hub transfers from JUPITER have run at about 2–5 MB/s.
- `$HOME` on JUPITER is over its **inode** quota (81,960 of 80,000 soft,
  82,000 hard), not its size quota. Write nothing to `$HOME`.
- Other login nodes can't be reached from a login node by ssh (publickey).
  Check tmux on the node that started it.
- Existing public repos (`geodml-papersize`, `geodml-emnlp-2026`,
  `geodml-semantic-readiness`) were not modified. `AxisGEO` was never published.
