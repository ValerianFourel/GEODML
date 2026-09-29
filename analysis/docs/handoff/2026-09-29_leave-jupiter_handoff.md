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
