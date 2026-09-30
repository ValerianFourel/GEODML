# Reuse existing Gemma preparation and shorten checks

User repeated preparation of the already existing run and hit the intentional
output-exists refusal. Then the admission check reported 62 seconds remaining
under the retained ten-minute start-gap rule. No evidence of a new allocation
or model start. The wait itself is not a bug and its rule was not waived.

Fixed preparation in `horeka_gemma_si.py`: a repeat verifies the requested
settings, preserved input hash/baseline bytes, original clean/pinned execution
checkout and exact run.sh command/interpreter, then reports status reused.
Does not overwrite, migrate the code pin, refresh approval or allocate. Conflicts,
partial preparation and tampered inputs/run.sh still fail. New-run behavior and
scientific settings remain unchanged. Factored run.sh rendering for both paths.

Moved the existing HTML admission program into a `check` subcommand, with account
passed explicitly. The scoped five-active/no-pending exception, 295-job account
limit, storage checks and ten-minute observed-start gap remain. It never
allocates, cancels, holds or changes jobs. Kept salloc as a separate HTML block.

Code fix commit: 788ef7d (full SHA embedded in HTML). The page's step 2 fetches
this helper in a separate worktree and verifies/reuses the existing preparation.
Its saved execution remains pinned to ddc93fe. Step 3 is now a short command
without inline Python/heredoc. Start at step 2 once, then repeat only step 3 for
wait/full-queue responses. Do not delete the existing run or redownload models.

Verification: regression reproduced output-exists failure before the fix, then
passed with byte-identical artifacts. Tests cover settings conflicts, changed
inputs/run.sh and the actual admission CLI: 28 running + pending allowed, 295
account jobs / recent start / unsafe storage blocked. Focused owner and adjacent
tests: 51 passed; after account CLI extraction adjustment, owner suite 9 passed.
HTML command syntax and embedded Python parsing checked, copies match, diff
whitespace check passed. No remote access or live cluster changes performed.
