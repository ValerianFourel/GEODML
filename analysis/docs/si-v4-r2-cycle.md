# SI-v4 r2 repair and absolute evaluation

This revision is a development candidate. The deterministic repair has been checked against the exact rejected proctoring output. No Gemma semantic acceptance follows from that check or from CPU tests.

## What changed

Historical `source-importance-task-v4` records reconstruct their original prompts, schemas, IDs and seeds. New preparation uses `source-importance-task-v4-r2`. The six importance anchors, complete answer, source records, citation preprocessing, output budget and pinned Gemma configuration remain unchanged.

The revised map parser can remove a redundant `non_substantive` exclusion for one bare enumeration marker in an unambiguous ordered list. It retains that marker in its original claim. Strict full-map validation runs afterward. The original raw output, validation failure, repaired output and versioned repair receipt remain separate. Substantive overlaps are never repaired automatically. A model corrective retry receives its rejected output and precise error; identical invalid responses stop without additional retries.

Stage A explicitly separates answer importance from fulfilment and list position. Stage B checks decisive qualifications, considers background support before returning zero, respects fixed roles and returns `map_issue` for substantive map defects. These are revised instructions requiring empirical evaluation, not automatic corrections of model grades.

Fixed-map repeats import and verify repair receipts. Failed and quarantined maps remain in the repeated cohort and block dependent scoring. The repeated run never replaces them with a newly generated map.

## Finite workload and resource boundary

The approved cycle is one candidate with at most **32 GPU-hours**, split into **16 development and 16 fresh**. Each phase permits at most four identified allocations of at most one hour on one four-A100 node with 32 requested CPUs and node-default memory. A common budget ledger binds both phase configurations. Submission checks current scheduler admission, storage, immutable inputs/configuration and previous ownership. It neither chains allocations nor modifies existing jobs. Uncertain submission ownership stops further admission.

Historical startup was 8.9–10.9 minutes. Prior warm measurements were 5.87 minutes for 20 cells and 1.30 minutes for the two-map retry. The provisional full-cycle estimate is 4–7 node-hours including repeated startup. Each segment checkpoints the finite backlog using Slurm's actual end time and existing drain margins. Actual allocation occupancy is read from accounting; warm request latencies are never summed as node-hours.

Development contains 40 cells/214 supplied sources, 48 original constructed pairs, five cyclic proctoring map controls and four fixed-map role controls. A predetermined 24-cell subset contains 12 Qwen and 12 Llama cells. It receives three end-to-end executions, including the initial run, and three additional fixed-map source executions. The matched 4096-token v3 bridge measures cost only.

Fresh evaluation is blocked until the complete development gate passes. It requires 96 core cells from 48 paired prompts, 24 stress cells from 12 paired prompts, 48 held-out constructed pairs and repeats of 24 preselected cells in both modes. Selection requires matching generator pairs and verified development prompt/keyword exclusions. Missing historical keyword metadata must be recovered from the verified population by prompt ID; it is never guessed. Known axis bins are reported as bins, not fabricated continuous readiness values. Any post-freeze semantic revision needs a later cycle and different fresh data.

## Entry points

All commands below use an already-open cluster shell, the existing environment and a clean checkout at the published revision. They do not install models or change the serving environment.

1. `run_si_v4_cycle.py prepare-development --source-run PATH --output PATH --list-cell-id ID --list-spec JSON` verifies the saved bundle hashes, freezes r2 and cost inputs, creates the controls, selects repeats, and writes independent grade packets. The list specification names the five independent factors and must reconstruct the masked answer exactly. The generated `queue/evaluation-plan.json` is immutable.
2. `horeka_si_v4.py prepare --evaluation-plan PLAN --cycle-budget LEDGER --workspace WORKSPACE --output RUN --account ACCOUNT --walltime 01:00:00 --approval TEXT` validates the installed pinned runtime/model and creates the existing server launcher. Both phases must use the same budget ledger. This step does not allocate resources.
3. `run_si_v4_cycle.py submit --config RUN/config.json` submits one identified segment after fresh admission checks. Repeat this same command only after a terminal allocation and reconciliation if the queue remains incomplete. The ledger enforces the four-segment phase cap. No completed calls are repeated.
4. `run_si_v4_cycle.py resource-usage --budget LEDGER --output NEW_JSON` captures accounting for every submitted segment. Live/unknown allocations remain explicitly incomplete.
5. `run_si_v4_cycle.py export --config RUN/config.json --output NEW_ARCHIVE.tar.gz` exports frozen inputs, complete saved results, attempt receipts and logs while the queue is idle. It excludes serving caches and authentication files. Preserve the printed SHA256 with the archive.

`prepare_si_v4_evaluation.py` also exposes `subset`, `repeats`, `fresh`, and `packets`; use `--help` for their exact arguments. Fresh selection consumes a verified population freeze, never the already-reviewed forty-cell pool. `run_si_v4_cycle.py plan` assembles the finite queue from explicit candidate, constructed, repeat and matched cost inputs. It refuses a fresh queue without a passing development report.

## Independent reference workflow

Use separate fresh-context Astra and Sol assessments at high effort. Send each reviewer only the packet's instructions and body, retaining packet hashes and returned raw assessments separately. Do not include surrounding conversation, generator metadata, other sources, Gemma maps or Gemma grades in independent source-grade assessment.

- `packets --phase grade`: complete original request, answer and one supplied source. Freeze both reviewers' ranges and unresolved judgments before exposing candidate outputs.
- `packets --phase map`: request, complete answer and map, with no source. Assess content fidelity and importance roles separately. The role-control fixed map can be reviewed before GPU execution.
- `packets --phase support --frozen-grades FILE`: candidate findings and full original source, after independent grades are frozen. Review whole-source support, missed support, role consistency and collective witness sufficiency separately. Zero-finding outputs still require complete-source review.

Reference records bind to exact answer/source/map/output/packet hashes. Preserve reviewer identities, available configuration, raw assessments and disagreements. Compatible grade ranges use their intersection; disjoint ranges remain unresolved. Reference-model consensus is not ground truth. Human annotation is not an added prerequisite.

## Reporting and acceptance

Use `run_si_v4_cycle.py assess --config RUN/config.json --resource-usage USAGE.json --output NEW_REPORT.json` for a progress report. It resolves the saved report pointers, repeat selection, constructed controls and supplementary evidence from the frozen queue, and writes adjacent JSON and HTML. With no references, semantic gates remain unavailable. Add `--references REFERENCES.jsonl --reference-packets PACKETS --supplement-references SUPPLEMENT_REFERENCES.jsonl --supplement-packets SUPPLEMENT_PACKETS` after the independent reviews are frozen. Fresh assessment rechecks the saved development supplementary evidence against the unchanged candidate settings.

`compare_source_importance_runs.py --absolute --inputs FREEZE --candidate REPORT --references JSONL --reference-packets DIRECTORY --repeat-cohort SELECTION_JSON --repeat-end-to-end REPORT ... --repeat-fixed-map REPORT ... --baseline COST_REPORT --resource-usage JSON --controls JSON --phase development --output REPORT.json` writes JSON and a portable HTML report. The repeat arguments are complete lists of three executions for each mode; include the initial candidate report as end-to-end execution one. Reports live under `evaluation-results/NAME/reports/`; `reports/latest.json` points to the complete latest report, without relying on UUID sorting.

Every frozen cell/source remains in the report. Missing, failed, blocked, quarantined, uncertain and ineligible cases are distinct. Report all-attempt coverage alongside conditional agreements, both repeat modes, all-zero and incomplete top groups, and available generator/style/readiness strata. Do not total reused results as additional unique completion.

The absolute policy is `analysis/config/si_v4_absolute_evaluation.json`. Historical comparative configuration and reporting remain available. The numeric absolute thresholds remain unchanged: integrity errors 0; retained validity 100%; eligible completion 99%; resolved claim–witness support 95%; unresolved pairs at most 5%; grade-range agreement 90%; exact repeats 95%; within-one repeats 99%; nonempty top-group agreement 95%; map fidelity 95%; essential omissions/reversals 0; unsupported constructed grades 4/5 0; warm GPU-hours per completed eligible cell at most three times the matched bridge. Comparative superiority over v3 is not an acceptance gate.

Missing evidence prevents readiness. Formatting success does not establish semantic quality. The final disposition is ready to scale, a named further repair, or reconsideration of the judge configuration when substantive support errors persist with faithful maps. No code path launches production evaluation automatically.
