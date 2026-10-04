# Fable plan: tests on the Qwen/Llama generator corpus before Gemma

Fable 5.1 answered `analysis/docs/fable-pre-gemma-tests-prompt.md` in this
checkout (`9e23a3d`). Planning only: no cluster command, job or code edit.
Deliverable: `analysis/docs/fable-pre-gemma-tests-plan-20261004.md` with the
measurement inventory, an eight-group test catalogue, the Gemma decision, an
ordered plan and a paste-ready implementation prompt for milestones 1 and 2.

Decision: hold the auto-submitting Gemma sender (Valerian decides; he offered a
pause method that cancels nothing). Reasons from files: SI-v4 r2 misses its
frozen thresholds and fails semantically (`si-v4-r2-semantic-review-20261004.md`;
`si_v4_evaluation.json` has `production_launch_authorized: false`); its known
failure mode (request fulfilment leaking into support grades) moves with the
axis, so more cells make a biased RQ2 association more precise; keyword-cluster
uncertainty depends on keywords, not cells; Qwen is unfrozen and keyword-ordered.
Gates before any large pass: validity gate on fresh axis-stratified cells
(G8.4), eligibility census and frozen population (G8.1, G1.1, G1.8), precision
table from generator-level variance (G8.3), Qwen missingness (G1.2). Default
sampled design: natural condition, matched prompts, both generators, ~200
keywords ≈ 40,000 cells ≈ 41–48 five-hour bouts ≈ 820–960 GPU-hours ceiling,
arithmetic on `si-v4-gemma-five-hour-cost-20261004.json` rates.

Contradictions recorded: the root AGENTS.md names Nemotron as the judge while
the files show Nemotron v4 unusable and Gemma unaccepted; the paper's Llama
registry (310,380, 2026-10-01) is a stale snapshot next to HoreKa's 26,008 × 12
− 6 = 312,090; `report_axis_permutation_study.py` requires identical candidate
pools and is the wrong driver for agentic data (reuse `fit_axis_rankings`).

Open questions for Valerian are listed at the top of the plan: the Gemma
preparation freeze root (it is the eligibility census), axis-map and
registration paths on HoreKa, the Qwen dataset contract, the outcome of
page-extract 5179545 / page-relocate 5179548, archived prompt embedding format,
the unreviewed text report, the original-500 Nemotron judgments, keyword text.

Local checks: `python3 -m pytest -q` on `test_report_generator_outputs.py`,
`test_answer_readiness.py`, `test_source_importance_pipeline.py` passed (51
tests; miniconda Python, NumPy 2.4.2, SciPy 1.17.0; no `.venv` in this worktree).
Next: Valerian pauses the sender, answers the questions, and pastes the plan's
§7 prompt into a coding agent for milestones 1 and 2.
