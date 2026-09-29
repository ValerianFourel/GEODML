# Leave JUPITER page, 2026-09-29

Valerian asked for the commands to leave JUPITER for good, on a dedicated HTML
page. Created `/Users/valerianfourel/Hamburg/GEODML_Unified/leave-jupiter.html`
(local operator page, not in git) and opened it in Safari. Nothing was run on
any cluster; no allocation, cancellation or deletion was prepared.

The page uses the existing JUPITER setup file
`/e/fscratch/scifi/fourel1/geodml/audits/llama-prep/sweep-env.sh` (pin `71ace0c`):

0. Set up the shell and ask for the HF write token.
1. Read-only leave checklist, rerun at the end: live Slurm jobs, dispatcher
   lock, tmux/python processes on this node and on `jpbl-s01-04`/`jpbl-s02-03`
   (BatchMode ssh), cron/scrontab, dirty or unpushed checkouts under `$BASE/src`,
   unpublished Llama attempts, Hub packages owned by `cluster == "jupiter"`,
   `LLAMA_DONE`, Qwen `new_cells` (publisher dry run), the archive and stored token
   files. Ends with `LEAVE_READY=yes|no`.
2. Publish every Llama round (the relaunch page's discover + `sync-llama` step).
3. `publish_qwen_results.py --apply` on `experiment-v2-incremental-v1`.
4. Archive every `local-only/` folder of the JUPITER dataset roots (population
   registration needed by the axis analysis, verification caches, input-owner
   records; never published to the Hub) plus audit files under 50 MB into
   `$BASE/audits/leave-jupiter/jupiter-archive-<stamp>.tar.gz` with SHA-256.
   Stops if LMSYS/WildChat/MS MARCO names appear in local-only data, or above 20 GB.
5. Mac: renew `ssh jupiter`, rsync the archive folder to
   `~/Hamburg/GEODML_Unified/jupiter-archive`, verify the hash.
6. List token files and `rm -i` them; revoke the JUPITER HF token afterwards
   (make a new HoreKa token first if they share it).

Round 8 (the approved 1 × 5 h Llama leftover relaunch) and `LLAMA_DONE` were
not confirmed in any handoff. `LLAMA_DONE=no` blocks the exit: Llama was not
validated unchanged on HoreKa A100s.

Verification: every command block passes `bash -n` (macOS bash 3.2), embedded
Python passes `ast.parse`, the copy-button JavaScript passes `node --check`, and
the HTML blocks match the checked sources exactly. Linux-only flags (`find
-printf`, `du -b`, `pgrep -a`) were not executed locally.

Next: inspect the step 1 output. After leaving, AGENTS.md still gives JUPITER
ownership of the frozen population, reference timings and plan republication;
moving those roles to HoreKa is a separate change that needs review.

## Follow-up: explicit Llama completion check

At Valerian's request the page gained a read-only step 1, "Is Llama fully
done?" (later steps renumbered 2–7). It lists live Llama Slurm jobs, end states of
every Llama round (including whether round 8 was ever submitted), local attempts
without a `released` sync receipt, and Hub registry totals: planned, completed,
terminal-failed and remaining Llama cells, unfinished packages by status and
owned packages. Output is saved to `$PREP/llama-done-<time>.txt`. Verdict
`LLAMA_FULLY_DONE=yes` requires 0 live Llama jobs, 0 unpublished attempts,
0 cells left and 0 owned packages; terminal-failed cells are reported separately.
All blocks pass `bash -n`/`ast.parse`; the verdict logic was exercised locally
with synthetic yes/no inputs only. Not run on JUPITER.

## Llama status from the user's JUPITER output (2026-09-29)

Round 8 ran as job 2102836 (1 × 5 h, jpbo-037-48, 28 Sep 12:28–15:17, 02:49:02)
and ended FAILED, exit 1. The cause is the runner's deliberate
`RuntimeError: 3 agentic-search cells failed after bounded retry`; the
EngineDeadError in the server log appears at shutdown. Ledger for writer
`jupiter-llama-5hour-e3b2bb8d4e6e-1`: completed 4,578, terminal_failed 3,
which accounts for all 4,581 cells the registry still listed as left. Before
publishing, the registry showed 305,796 completed and 3 terminal failed of
310,380 planned; the attempt was `not synced` and owned 17 packages. After
page step 3 publishes it, expect about 310,374 completed and 6 terminal-failed
cells. Valerian decided to leave those 6 stragglers for later; they are known
failures, not a reason for automatic retries.

JUPITER login banner: $FSCRATCH retention is 30 days, and cleanup starts on
2026-10-01. The page now warns to finish publish/archive/download before then.
Next, in order: page step 3, step 1 (want 0 unpublished, 0 owned, 0 left),
step 4, step 5, step 6 on the Mac, step 7, then step 2.

## Round 8 published; inventory step added

User-pasted `sync-llama` output: round 8 attempt released as bundle
`bundle-c5918b31d82c…` with `verified_tasks_in_checkpoint 4581`; summary
`already_synced 55, released 1, live 0, blocked 0`. Every Llama result cell
is therefore on the Hub. Not yet confirmed by a fresh Llama-status run.

Valerian asked to go back to the start: inventory everything on JUPITER and
scratch, then push everything worth keeping to Hugging Face, Llama first. The
page gained a read-only step 1, "Inventory everything on JUPITER" (steps now
0–8). It prints `jutil user dataquota` and walks `$HOME` plus every
`/e/*/*/$USER` and `/e/*/$USER` root once (nested roots deduplicated). For each
root: total bytes, files, bytes older than 30 days, newest mtime; then folders by
depth with size/file thresholds. Output saved to
`audits/leave-jupiter/inventory-<time>.txt`. Walker tested on a local synthetic tree.
Home contains many readiness/axis pointer files (`geodml-readiness-*`,
`geodml-axis-v2-*`, `important_commands.txt`, `nltk_data`), per the user's `ls`.
The HF upload design for non-result data waits on the inventory sizes.

## Llama confirmed complete; Hugging Face archive replaces the local tar

User-pasted results on 2026-09-29 (jpbl-s02-01):
- Llama status: registry revision `64d79e59fd40`, 1,008 packages
  (1,006 complete, 2 blocked), 310,374 of 310,380 cells completed, 6 terminal
  failed, 0 left, 0 owned, 0 unpublished attempts: `LLAMA_FULLY_DONE=yes`.
  Rounds: 1 (4 COMPLETED/1 FAILED), 5 (9/1), 6 (18/2), 7 (5/1), 8 (0/1, 4,578
  cells), llama-five-872875d (5/0); rounds 2–4 never submitted.
- Qwen: 24,218 verified on disk, 24,218 already published, `new_cells 0`.
- Llama publish rerun: `already_synced 56`, nothing else to do.
- Checklist: no Slurm jobs, lock free, no crontab. A tmux session
  `geodml-agentic-resume` (created 13 Sep) is still running on jpbl-s02-01. Other
  login nodes can't be reached by ssh (publickey). Token files:
  `$BASE/huggingface-acl-arr/token`, `~/.cache/huggingface/token`, `~/.git-credentials`.
- Old tar step: local-only 372 MB, audit files under 50 MB 58.7 GB, plus many
  0.8–1.8 GB recovery forensics files (`audits/recovery-*`); over the 20 GB
  guard, so it stopped.

Valerian chose to put this data in a Hugging Face "jupiter archive" dataset.
Page changes (steps now 0–9, all on JUPITER, no Mac step):
- Fixes: the llama-done `tee` path (was `$PREP` outside the subshell);
  "no crontab for USER" no longer counts as an entry; tmux sessions are now
  flagged; the checklist knows the fscratch token path and checks archive
  completion instead of a tar file.
- Step 6 writes `audits/leave-jupiter/hf-archive/hf_archive.py` (embedded in the
  page) and plans roots `fs=/e/fscratch/scifi/fourel1` and `home=$HOME`. It
  excludes `geodml/src`, `geodml/python`, `geodml/huggingface-acl-arr`,
  `geodml/audits/leave-jupiter`, and home's `.cache`, `.ssh`, `.local`, `.config`,
  `nltk_data`, `.vscode-server`. Also excluded: secret file names, files ≤2 MB
  containing token patterns, and `*.safetensors`/`*.gguf`. Files are grouped
  into ~10 GB units. It reports mentions of LMSYS/WildChat, and the run refuses
  them without `--accept-restricted`. 150 GB guard.
- Step 7 uploads in tmux `hf-archive` (the token is typed inside tmux) to the
  private dataset `ValerianFourel/geodml-jupiter-archive-private`: per unit,
  `tar | gzip -1` into parts of ≤5 GB, members list, manifest, one commit, then
  size/sha256 checked on the Hub before local parts are deleted. Resumable via
  local done markers. It finishes with `index/` (file→unit TSVs, plan summary)
  and a README.
- Step 8 watches; step 9 removes tokens after `ARCHIVE_COMPLETE=yes`.

Verification: archiver tested locally end to end with a fake Hub (secret and
weight exclusion, restricted gate, transient-error retry, resume skip,
byte-identical restore including symlinks); all page blocks pass `bash -n`,
embedded Python parses, the embedded script is identical to the tested one,
and the generated tmux `run.sh` passes `bash -n`. Not run on JUPITER.

## Old tmux session and the project1 filesystem

`geodml-agentic-resume` on jpbl-s02-01 is the idle shell of `salloc` job
1776656 (13 Sep). Its allocation was revoked at its time limit ("exceeded its
time limit and its allocation has been revoked"); its last Qwen/Llama retry
steps exited 1 (5 and 10 cells failed after bounded retry). There is no live
allocation (`squeue --me` empty), so closing the session releases nothing.

That job ran from `/e/project1/scifi/fourel1/geodml/src/...`, so data also
lives on the project1 filesystem. Page changes: step 6 now plans `home` plus
every `/e/<fs>/<project>/$USER` root (label `<fs>-<project>`, e.g.
`fscratch-scifi`, `project1-scifi`), excluding `geodml/src`, `geodml/python`,
`geodml/huggingface-acl-arr` and `geodml/audits/leave-jupiter` on each. The
checklist's checkout check now covers `/e/*/*/$USER/geodml/src/*`. Multi-root
planning was tested locally on a synthetic tree.

## Archive sizing, the GeoAxis 26k repository and committed archivers

**Sizes (step 6a breakdown, user-pasted).** fscratch has 3,326.7 GB: small files
(≤50 MB, not restricted) 192.2 GB, large non-restricted 1,402.6 GB, restricted-local
1,731.8 GB. Largest parts: `runs/readiness-30k-axis1` 2,272.5 GB (1,727 GB
restricted-local per-round cumulative `question_embeddings.restricted-local.npz`),
`runs/readiness-30k-high-axis-v2` 674.7 GB (`ten-section-20260902T131831Z/sections-*`
621.8 GB large), `audits/recovery-*` about 200 GB forensics tars, `compile-cache`
72.2 GB, `audits/llama-five-872875d` 59.3 GB, 56 `jupiter-llama-*` folders of
0.3 GB. project1 has `GEODML_Analysis` 955 GB (interpretability 515, hf_cache 429)
and `geodml/models` 343 GB; scratch has 26 GB. The first run refused at 150 GB.

**Decisions (Valerian, 2026-09-29).**
- The final 26,009-prompt embeddings and axis maps go to a **private** HF repo
  despite the restricted-local label, which is a deliberate deviation from the
  runbook rule. Exemplar text and other restricted-local files stay out.
- Key embeddings means the final audit only. The general archive keeps small
  files only, trimmed, and includes project1 and scratch now.

**GeoAxis discovery (user-pasted).**
- Final audit:
  `runs/readiness-30k-high-axis-v2/ten-section-20260902T131831Z/global-merge-hf-20260903T210857Z/checkpoint/final-audit-4gpu-a452213`
  (595 MB, commit a452213, `final-audit-summary` PASS). Both population files have
  26,009 rows and match the contract hashes. The embeddings are 8 shards per view,
  26,009 × 4096 float32.
- Subspace root on project1:
  `runs/semantic-readiness-subspace/7679b314…-20260820T104733Z`, with both maps and
  the battery.
- Registration: `datasets/experiment-v2-incremental-v1/local-only/population-registration-v1`.
- The global merge has no `hf-publication-receipt.json`, so AxisGEO was never
  published. `axisgeo-unified-hf-dataset` is 260 MB. `huggingface_hub` is 1.30.0.
- Not found at the guessed paths: `runs/readiness-30k-axis1/plans/*` and
  `runs/acl-arr-search-experience/pilot-500-*/selection-manifest.json`. The plan
  pointer is `$GEODML_PROJECT_ROOT/geodml-readiness-30k-v2-plan-latest.txt`; the
  selection manifest path comes from the registration manifest.

**Code (commits `eefaa85`, `062f06a`; push to origin was blocked by the
permission system; Valerian must push before running on JUPITER).**
- `analysis/scripts/archive_jupiter_to_hub.py`: the page archiver, now in git,
  with `--max-file-mb`, `--skip-name-contains` (default `restricted-local`) and
  `--exclude-glob`, all pruned during the walk. It prints filter counts and the
  names of files that mention LMSYS/WildChat.
- `analysis/scripts/archive_geoaxis_prompts_26k.py`:
  - builds an allowlist into hard-linked staging (final audit including the 16
    shards, maps without exemplars, battery, plan, checkpoint without nested
    final-audit/restricted/oversize files, registration, selection manifest,
    pointers, and `--extra` provenance) with `MANIFEST.tsv` and README
  - checks the contract hashes; defaults to a dry run
  - `--apply` refuses a non-private repo, uploads with `upload_large_folder`,
    verifies each file by LFS sha256 or git blob sha1, and writes a receipt
- Tests: `analysis/tests/test_archive_jupiter_to_hub.py` (4) and
  `test_archive_geoaxis_prompts_26k.py` (4), 8 passed with fake Hubs.

**Pages.**
- `geoaxis-26k.html`: 0 setup, 1 discovery (done), 2 dry run (writes
  `audits/leave-jupiter/geoaxis-26k.sh` with the exact sources, finding the
  plan/selection manifest itself), 3 upload in tmux `geoaxis-26k`, 4 restore check.
- `leave-jupiter.html` steps 3 and 6–8 now use the pinned archiver. The replan
  covers home, fscratch, project1 and scratch with ≤50 MB files and the agreed
  excludes; the checklist also checks `GEODML_Analysis`.
- The generated `geoaxis-26k.sh` was simulated and passes `bash -n`, and every
  block passes `bash -n`. Nothing has run on JUPITER.

Next: push `062f06a`, then run geoaxis steps 2–4, then leave-jupiter step 6
(replan), 7 and 8.

## GeoAxis 26k upload started; readiness-axis archive prepared

- GeoAxis 26k dry run (user-pasted), pin `062f06a`:
  - 271 files, 9.66 GB staged; `EMBEDDING_SHARDS 16`, both contract hashes match, `DRY_RUN_OK`
  - the plan (`…/question-populations/readiness-30k-v2-20260822T102133Z-58192518/plan-support-aware`) and the selection manifest (`/e/project1/…/runs/acl-arr-search-experience/pilot-500-axis-balanced-a559b7058277/selection-manifest.json`) were found
  - `checkpoint/` is 8.7 GB: all ~740k candidates and their validation
  - the upload started in tmux `geoaxis-26k` on jpbl-s03-01 at 16:21; completion is not yet confirmed
- Pushed to GitHub: `961ad0b..9b6e5c7`, then `..7cfd94f`.
- Readiness-axis construction:
  - **Decisions (Valerian):** a new private repo `ValerianFourel/geoaxis-readiness-axis`. The restricted-local scope (about 3,200 WildChat, MS MARCO and LMSYS rows, with their grades and embeddings) is included by owner decision, against the runbook rule and LMSYS terms. Upstream content is the raw judge outputs, the corpus, task bank and codebook, and source-acquisition manifests and logs only. Earlier map variants are excluded.
  - **Pipeline:** `semantic_readiness_hf_dataset_jupiter_runbook.md`. Everything is under project1 `$GEODML_RUNS_ROOT`. The subspace root (887 MB) holds `bundle`, `embeddings`, `maps`, `robustness`, `comparisons`, `confirmations`, `question-populations` and `huggingface-datasets`.
  - **Code, commit `7cfd94f`:** new `archive_geoaxis_readiness_axis.py`, driven by `--group DEST=PATH` and `--manifests-only DEST=PATH` arguments. Staging, `MANIFEST.tsv` and private upload with verification are shared with the 26k archiver, refactored into `stage`, `write_manifest` and `upload_private` with no change in behaviour.
  - **Tests:** 11 archive tests pass.
- Page `geoaxis-readiness-axis.html` has read-only step 1 so far. It prints the inputs recorded in each map, embedding and assembly manifest, the sizes of the readiness runs, and the 20k judge queue, task bank and codebook. The dry run and upload steps follow once that output is back.

## Readiness-axis discovery and staging plan

User-pasted discovery:
- **Subspace root** `…/runs/semantic-readiness-subspace/7679b314…-20260820T104733Z`, 887 MB:
  - `bundle` 222 MB (restricted-local 121, huggingface-safe 102)
  - `embeddings` 321 MB (qwen3-8b 160 MB, mistral runtime-v3 161 MB)
  - `maps` 31 MB, `huggingface-datasets` 223 MB, `question-populations` 91 MB, plus robustness, comparisons, confirmations
- **Maps:** qwen v2 (commit 9ec2ff4) and mistral v3 (commit 116417e) were both fit on restricted-local `prompts.jsonl` sha256 `9cbefcc5…` and `annotations.jsonl` `965c1319…`, judges primary-frontier / replicate-frontier-a / -b, ridge 1.0.
- **Bundle** (commit 7679b31) inputs:
  - corpus `semantic-readiness-incremental/40527b0…/incremental-corpus/semantic_readiness_expanded_corpus.jsonl` `b4d6339d…`
  - codebook `…/semantic-readiness-20k-abstention/4c9cd20…-four-judge-v2/task-bank-four-judge-v2/readiness_label_codebook_private.jsonl` `063e0fc4…`
  - queue root `…/judge-queue` (121 MB, 85,955 files; judges gemma4-31b primary, qwen3-32b-a, ministral3-8b-b, llama3.3-70b-c)
  - restricted sources: WildChat only (3,200 prompts)
- **Other runs:**
  - `fc361dd…` (an earlier aborted 20k root) is not archived.
  - A stray directory `4c9cd20…-four-judge-\nv2` with a newline in its name is ignored.

Code `9e4b2a0` (pushed): `--pack` / `--pack-manifests` store file-heavy folders as `DEST.tar.gz` plus `DEST.members.tsv` (per-member sha256), because the Hub allows at most 10,000 files per folder. 12 archive tests pass.

Page `geoaxis-readiness-axis.html`: steps 0 setup, 1 discovery (done), 2 dry run, 3 upload in tmux `geoaxis-axis`, 4 check against the recorded input hashes. The dry run writes `audits/leave-jupiter/geoaxis-axis.sh`:
- **groups:** subspace, task-bank, corpus, the judge run's `*.txt`
- **packs:** judge-queue, the incremental run
- **manifests-only packs:** judge slurm; phase1, expansion and base-axis acquisition runs; readiness-axis-reports / validation / factor-stress; the fscratch HF export 141c9c0b

The generated script was simulated and passes `bash -n`. Nothing has run on JUPITER.

## Publication audit before leaving (read-only page `jupiter-publication-audit.html`)

Valerian's professor asked him to stop using JUPITER. Before anything expires,
we check what a publication could need that exists only on JUPITER.

Already safe (Hub listing from the Mac, 29 Sep):
- `geodml-papersize` (public, 37.8 GB, 305 commits; last JUPITER sync
  2026-05-24; runs with 12 HTML-cache tarballs, rag_index 6.1 GB, order_probe
  953 MB, interpretability/output 5.8 GB)
- `geodml-emnlp-2026` (public, 1.9 GB, the reproducibility set, 20 Jul)
- `geodml-semantic-readiness` (public, v1 pilot), `-20k` and `-all-results` (private)
- `geodml-experiment-v2-paper-private` (coordination, exchange, recovery snapshot)
- `geoaxis-prompts-generation-26k`
- `AxisGEO` does not exist.

The Mac's `~/Hamburg/{GEODML,geodml-dataset,GEODML_Analysis}` hold about 100 GB of
the EMNLP work (see `DATA_POINTERS.md`).

At risk on JUPITER:
- `/e/scratch/scifi/$USER/data` (26 GB): the bf16 EMNLP dataset from May. It
  falls under the 90-day scratch retention; cleanup starts 2026-10-01.
- `GEODML_Analysis` on project1 (955 GB): interpretability 515 GB versus 5.8 GB
  on the Hub, hf_cache 429 GB of model downloads, logs 763 MB, hf_stage 5.4 GB.
- The 24 May session log said the Stage F probing (8 jobs) and 24 order-probe
  top-off jobs were still running, with a re-push planned afterwards; nothing
  confirms that re-push.
- The ACL ARR document pilot and search-experience pilots on project1 have no
  confirmed export.
- The Slurm accounting history.

The audit page covers these in five steps:
- setup writes `audits/publication-audit/pub_audit.py` (tested on the Mac
  against the real public repos and fake folders)
- `hub`: per-file comparison of `$SCRATCH/data` and `interpretability/output`
  with both EMNLP repos; HTML caches matched by prefix to the per-cell tarballs
- `interp`: breakdown of the 515 GB by folder and file type, plus the
  `GEODML_Analysis` git state, `hf_stage`, logs, and `exports`/`staging`/`manifests`
- `runs`: the pilots and every fscratch run not yet archived
- `sacct`: exports 2026 jobs to `audits/slurm-accounting/` with a GPU-hour summary

No decisions about further uploads until the output is back.

Scope decision (Valerian, 29 Sep): the publication is the **new, improved version**
(Experiment V2 on the GeoAxis readiness axis), not the finished EMNLP/DML paper.
The audit page is reordered to match:
0. setup
1. Experiment V2 pilots and other runs (priority)
2. Slurm accounting (priority)
3. EMNLP Hub comparison (optional legacy check)
4. 515 GB interpretability (optional legacy check)

EMNLP data is already public on the Hub. It only needs uploading if small final
results (for example probing tables) turn out to be missing and are wanted as a
cited baseline.

## Publication audit results (user-pasted, 29–30 Sep) and archive plan update

- Axis upload started in tmux `geoaxis-axis` on jpbl-s03-01 at 16:50;
  completion is not yet confirmed, and neither is the 26k upload.
- **EMNLP `$SCRATCH/data`:** 1,304 files (16.7 GB) are on the Hub. 130 files
  (0.52 GB) are not on the Hub and 79 files (0.065 GB) changed since the Hub copy,
  dated 17 May – 1 Jun: the order-probe top-off, features, per-cell run files.
  The 2 extracted HTML caches (9 GB) are covered by Hub tarballs.
- **`GEODML_Analysis/interpretability/output`:** 547.8 GB is not on the Hub. Almost
  all of it is `.npz` probing activation chunks (`t7_chunks_full` 58.9 GB × 8,
  `adm_chunks_full` 7.4–11.7 GB × 8, `t7_chunks_rw` 0.2–0.3 GB × 8), which would take
  about 2,000 GPU-h (`prob-*` jobs) to recompute. The small final results are also
  missing: `probing_results.csv` (changed), probing/ablation/saliency/weights
  summaries and plots, 23–25 May.
- **`GEODML_Analysis` code:** no unpushed commits; 46 untracked files (repair
  reports, DML summaries, `audits/`).
- **project1 exports:** unpublished `AxisGEO` exports (1.4 GB).
- **Pilots:** `pilot-3-627ad348c2a0` is 36.95 GB in 770,600 files; its recovery
  export is on the Hub. The other pilots are small.
- **General archive plan (page step 6, code pin `062f06a` unchanged):**
  - roots now in order home, fscratch, scratch, project1
  - new exclusions: `data/runs/*/phase2/html_cache` (scratch),
    `GEODML_Analysis/interpretability/output/probing_*/*_chunks_*`, `GEODML_Analysis/hf_stage`
  - the patterns were checked with the archiver's `walk()`
- **Public mirror of the complete interpretability output:** Valerian proposed
  `ValerianFourel/geodml-emnlp-2026-probing-activations`, 551.7 GB, about 1.5–3 days
  of upload. A script was drafted and tested with a fake Hub. Adding it to the page
  was blocked by the permission system as creating a public surface, so it needs
  Valerian's explicit permission or a private alternative.

Valerian approved the **public** mirror on 30 Sep ("make it public"). The audit
page gained step 5 and step 6:

- **Step 5** writes `audits/publication-audit/activations_upload.py` and starts
  tmux `emnlp-activations`. It creates the public dataset
  `ValerianFourel/geodml-emnlp-2026-probing-activations` with a README
  (provenance, companion repos, Llama/Qwen attribution) and runs
  `upload_large_folder` over `GEODML_Analysis/interpretability/output`, 4 workers,
  resumable. It then compares every file's size on the Hub with JUPITER and
  ends with `ACTIVATIONS_UPLOAD_COMPLETE`.
  - `SCOPE=without-t7-full` leaves out the 471 GB `t7_chunks_full` (about 80 GB).
  - Expected 1.5–3 days for the full 548 GB.
  - Start it after the general archive; `geodml-papersize` and
    `geodml-emnlp-2026` stay untouched.
- **Step 6** watches the upload.

The script was tested with a fake Hub for both scopes; the generated tmux script
passes `bash -n`.

## Status 30 Sep and the two finishing pages

User-pasted status:
- `GEOAXIS_26K_COMPLETE`: 271 files, 9.66 GB verified, `UPLOAD_EXIT=0`.
- `GEOAXIS_AXIS_COMPLETE`: 952 files, 1.28 GB verified, `UPLOAD_EXIT=0`.
- General archive: still the old 391-unit plan (`RUN_EXIT=1`, 0 units).
- Activation mirror: not started.

Two finishing pages on the Mac, built only from blocks already tested on the
other pages:
- `finish-shell-a.html`:
  0. setup
  1. close the finished GeoAxis tmux windows (only if the log shows completion)
  2. 26k restore check
  3. axis restore check
  4. find the 1.5 TB on `/e/scratch`
  5. replan the general archive (paste before uploading)
  6. upload
  7. watch
  8. tokens (after both shells finish)
  9. final checklist
- `finish-shell-b.html`:
  0. start the public mirror `geodml-emnlp-2026-probing-activations` (`SCOPE=all`
     or `without-t7-full`)
  1. watch

The general archive is insurance: provenance, pilots (including the ACL ARR
document pilot), ledgers, the accounting table, and the late EMNLP results.
Everything the new paper needs is already on the Hub.
