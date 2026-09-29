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
