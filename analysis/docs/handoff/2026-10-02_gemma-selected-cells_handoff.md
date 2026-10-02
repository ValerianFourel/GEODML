# Gemma selected-cell workflow

Valerian requested a replacement for the repeatedly pasted f256b688 admission
check, proposed deleting locks, and explicitly asked to offer random cells and
let him choose which ones Gemma judges. The obsolete check correctly protects
an already attempted preparation. Deleting its marker would not repair the old
9aa5d90 source hashes or the saved launcher. This change adds a separate selected
SI-v4 development run and preserves original inputs, attempts and results.

Read the last three indexed handoffs and the supplied cluster evidence. No live
cluster query was executed. Job 5175111 was historically granted for one hour on
dev_accelerated, hkn0402; its current state and time remaining are unknown. The
old source-hash import failed, then rerunning its launcher encountered the
existing attempt directory. Previously pending 5175056 is also unverified.
CAP=294 and account count 208 are historical observations, not current admission.

## Published code

Release branch: `codex/gemma-selected-cells`, isolated from unpublished manuscript
and other active-checkout history. Helper commit:
`d47418b81b93057a02cc54e0383492835400be79`, following initial selection commit
`3f6e4fc42730349ef91c6f660c15aedae6cc32db`. The helper commit was pushed and is the
exact pin used by the page. A following documentation commit publishes the page,
monitor regression and this handoff without changing the helper code.

Active checkout: `.worktrees/threehour-relaunch-fix`.
Release checkout: `.worktrees/gemma-selected-cells-release`.
Do not push unrelated active-checkout history merely to deliver this workflow.

`analysis/scripts/horeka_si_v4.py` now provides:
- `select`: verify saved source bundles, sample up to 20 development cells with
  seed 20261002, show requests/answer previews, accept numbered choices through
  the terminal, and exclusively create selected inputs with provenance and IDs.
- `fresh`: create a unique `reviews/gemma-selected-*` directory, offer the cells,
  query running owned Gemma jobs, choose or ask for the existing job, prepare
  only the selected cells, and save/print its exact compute-shell command.
  No running job or insufficient remaining time preserves the selection and
  stops without allocating. Preparation errors are saved explicitly.
- `prepare --selected-inputs --existing-job-id`: validate sample/provenance and
  saved contents; preserve model/settings; freeze only SI-v4 inputs; derive
  missing hashes using the existing bridge and run a real Coordinator import
  before inference. Require a matching owned running allocation and >=20 minutes
  remaining. Use a unique cache prefix and record the selected workload mode.
- `run-picked`: recheck bound allocation, protect concurrent selected runners
  with a nonblocking allocation-scoped advisory lock, preserve unresolved and
  duplicate attempts, reject overlapping prior saved/running/done/blocked tasks,
  refresh quota/storage evidence, then use the existing exclusive-node runner.
  The saved selected-mode run.sh enters this path too. The lock inode is retained;
  releasing the process lock on close does not require deleting a marker.
- `review`: selected mode runs exactly one v4 pass. The historical 40-cell,
  five-pass comparison remains unchanged. Unknown workload modes fail closed.

The sample is from the saved development source bundles, not the full corpus.
This is a user-selected diagnostic; scientific_result remains false. No uploads,
model downloads, allocation submissions, extensions, cancellations, inference,
cluster repairs or marker deletions were executed by Codex. Commands use the
existing allocation only. If expired, retain selected inputs; obtaining another
allocation remains a separate estimate and approval step.

## Operator page

Rewrote `analysis/docs/horeka-si-v4.html` with one short published-pin bootstrap,
instructions to paste its generated compute command in the matching existing
shell, and a read-only audit. The audit includes live Gemma queue, three newest
selection directories, original failed run, selected IDs, preparation errors,
bounded console tails, accounting, trial receipts and honest task counts. It
handles integer/mapping inventories and single/legacy pass sets. Saved successful
results are reported separately from committed completions.

The former `analysis/docs/horeka-si-v4-existing-job.html` links to the canonical
page. Matching exports in the workspace root and Downloads:
`horeka-si-v4.html`, `horeka-si-v4-recovery.html`, and new
`horeka-gemma-select.html`; the former-entry-point export is also updated.
Other Qwen dataset/audit and interactive-status pages are unchanged.

Estimate: conservative 20–50 minutes including historical 9–11 minute startup;
SI-v4 throughput remains unmeasured. The current allocation's remaining time is
the only budget: one exclusive four-A100 node, 32 requested CPUs, whole-node
memory, actual Slurm end time and drain/cleanup margins. This estimate does not
promise completion or authorize a replacement allocation.

## Verification

Applied HoreKa and test-audit skills and concise-writing guidance. Added tests at
CLI/input/ownership boundaries rather than source-only assertions. Terminal
verification caught the real nonseekable `/dev/tty` read/write-mode failure;
separate reader/writer handles fix it, with a retained PTY regression. The
preceding terminal check accepted actual input `2,3` and wrote the selected file.

66 focused tests passed in the final release checkout:
`PYTHONDONTWRITEBYTECODE=1 /Users/valerianfourel/miniconda3/bin/python -m pytest -q analysis/tests/test_horeka_si_v4.py analysis/tests/test_horeka_gemma_si.py`

Proof includes preserved legacy behavior; deterministic choice/content checks;
real frozen-input I/O and Coordinator import; preserved old artifacts; no-job,
pending, multiple-job, wrong-job and short-time paths; duplicate/unresolved/
overlapping attempts; storage refusal; and lock contention during execution.
Scheduler, quota and unavailable model/runtime boundaries use fixtures; no GPU
work or scientific validation occurred. A tightened shared scheduler fixture
initially omitted sacct, causing two legacy-test failures; the fixture now
models that read explicitly and the full focused run passes.

Both copied Bash blocks, embedded Python and page JavaScript parse. HTML IDs,
ARIA/navigation references, exact bootstrap text and matching exports were
checked. Browser verification remains unavailable from the preceding discovery;
no visual/copy-button browser test is claimed. `git diff --check` passes.

Next: Valerian pastes the new bootstrap in a HoreKa login shell, chooses cells,
then uses only its generated command in the matching still-live compute shell.
Return GEMMA_EXIT and the page audit output. No cluster result is inferred from
local tests or published code.
