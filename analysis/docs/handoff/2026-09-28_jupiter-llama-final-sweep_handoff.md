# JUPITER final Llama sweep handoff — 2026-09-28

Operator: Valerian runs every cluster command in his own shells and pastes the
output back; the agent prepares paste-ready blocks and never fabricates cluster
output. Continues
[2026-09-27_llama-round6-qwen-published_handoff.md](2026-09-27_llama-round6-qwen-published_handoff.md)
for JUPITER. The HoreKa state in
[2026-09-27_horeka-a100-bouts_handoff.md](2026-09-27_horeka-a100-bouts_handoff.md)
is unchanged by this session.

## 1. Code state

Branch `threehour-relaunch-fix`, pushed to `origin/codex/pilot-continuation`.
**JUPITER pin: `71ace0c567814665660af9e2f01fcafd17670e5e`** (pushed).

| Commit | Change |
|---|---|
| `11587e0` | `dispatch_threehour_wave.py llama` accepts `--walltime 01:00:00` and `--sweep`: every eligible Llama package is reserved and split into contiguous keyword-priority groups of about equal remaining cells (`groups(..., budget=None)`). Three new tests. |
| `71ace0c` | `--walltime 07:00:00` accepted for Llama (the one- and seven-hour paths share parametrized tests); 109 focused tests pass |

Focused suites (dispatcher, Llama five, wave coordination, agentic hours,
shared-hour reservation and scheduler owners): 107 passed on the Mac.

## 2. Operator page (local, not in git)

`/Users/valerianfourel/Hamburg/GEODML_Unified/jupiter-llama-sweep.html` replaces
the four older JUPITER pages (`llama-next-runs.html`, `llama-status-next.html`,
`qwen-publish.html`, `qwen-results-to-hf.html`; deleted at Valerian's request).
Steps: 0 setup (`audits/llama-prep/sweep-env.sh`, pin `71ace0c`), 1 is round 6
over, 2 publish every finished wave (writes `llama-sites.txt`, `next-round.txt`;
a round whose dispatch never submitted keeps its number), 3 audit with the
dispatcher's own sweep split, 4 the sweep, 5 finite watch loop (5 min, ≤100
rounds), 6 publish the sweep, per-job yields, `LLAMA_DONE=yes|no`.

## 3. What ran and the approved allocation

- Round 6 ended 18 COMPLETED, 2 FAILED (error logs not yet read). Page step 2
  published it after one Hub commit cooldown (HTTP 429, waited 3,660 s and
  retried once): `already_synced 29, released 20, live 0, blocked 0`.
- Audit at registry `1bf7fe7`: **LLAMA_COMPLETED 284,267, failed 3, 89 eligible
  packages, 26,110 cells**; the sweep split is 4,602 / 4,260 / 4,199 / 4,511 /
  4,200 / 4,338 cells.
- The first approval (6 × 1 h) could not cover this. Valerian then approved on
  2026-09-28: **6 × 1 node × 4 GH200 × 07:00:00 = at most 42 node-hours / 168
  GPU-hours**, booster, account scifi; every eligible package split evenly; cap
  10 concurrent (exception to the five-allocation default), simultaneous starts
  (exception to the ten-minute gap), stale-quota exception; no automatic
  retries, extensions or replacements. Estimate: round-6 members verified about
  1,460–1,490 cells/h, so each job needs about 3.2 h (4.3 h at the slowest 1,100
  cells/h) and ends by itself on `queue_exhausted`; expected use ≈20–26
  node-hours. The agent recommended 5 h as sufficient; Valerian chose 7 h.
- **Not yet submitted** at handoff.

## 4. Next actions, in order

1. Page step 0 again (new pin `71ace0c`), then step 4. Read round 6's two
   FAILED error logs.
2. Watch with step 5.
3. After all six end: step 6. `LLAMA_DONE=no` means a leftover that needs its own
   estimate and approval.
4. After Llama is done on JUPITER: refresh `coordination/progress.json`, the axis
   report `--registration` work, and the JUPITER home quota diagnosis remain open
   (see the 2026-09-27 handoffs).
