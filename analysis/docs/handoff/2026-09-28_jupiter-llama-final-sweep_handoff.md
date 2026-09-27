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
**JUPITER pin: `11587e088759d15f4792c9f19d222c9879a33c19`** (must be pushed before
the page's setup step can create its checkout).

| Commit | Change |
|---|---|
| `11587e0` | `dispatch_threehour_wave.py llama` accepts `--walltime 01:00:00` and `--sweep`: every eligible Llama package is reserved and split into contiguous keyword-priority groups of about equal remaining cells (`groups(..., budget=None)`). Three new tests. |

Focused suites (dispatcher, Llama five, wave coordination, agentic hours,
shared-hour reservation and scheduler owners): 107 passed on the Mac.

## 2. Operator page (local, not in git)

`/Users/valerianfourel/Hamburg/GEODML_Unified/jupiter-llama-sweep.html` replaces
the four older JUPITER pages (`llama-next-runs.html`, `llama-status-next.html`,
`qwen-publish.html`, `qwen-results-to-hf.html`; deleted at Valerian's request).
Steps: 0 setup (`audits/llama-prep/sweep-env.sh`, pin `11587e0`), 1 is round 6
over, 2 publish every finished wave (writes `llama-sites.txt`, `next-round.txt`;
a round whose dispatch never submitted keeps its number), 3 audit with the
dispatcher's own sweep split, 4 the sweep, 5 finite watch loop (5 min, ≤30
rounds), 6 publish the sweep, per-job yields, `LLAMA_DONE=yes|no`.

## 3. Approved allocation

Valerian approved on 2026-09-28: **6 × 1 node × 4 GH200 × 01:00:00 = 24 GPU-hours**
(6 node-hours), booster, account scifi; every eligible Llama package split
evenly; cap 10 concurrent (exception to the five-allocation default),
simultaneous starts (exception to the ten-minute gap), stale-quota exception; no
automatic retries, extensions or replacements. Estimate: measured 55-minute
Llama jobs did 1,042–1,395 cells, so 6 jobs ≈ 6,300–8,400 cells; Valerian
expected about one 5-hour job's worth (~8,000 cells) to remain.

**Not yet run**: round 6's final state, the remaining cell count and the sweep
submission are unknown at handoff.

## 4. Next actions, in order

1. Push `11587e0` (and this handoff) to `origin/codex/pilot-continuation`.
2. Page steps 0 → 1 → 2 → 3; send the audit output back. Step 4 only with
   `READY_FOR_SWEEP=yes`.
3. After all six end: step 6. `LLAMA_DONE=no` means a leftover that needs its own
   estimate and approval.
4. After Llama is done on JUPITER: refresh `coordination/progress.json`, the axis
   report `--registration` work, and the JUPITER home quota diagnosis remain open
   (see the 2026-09-27 handoffs).
