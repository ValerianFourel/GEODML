# Axis-2 readout: the paper's stage readout on a second direction (2026-10-12)

Labels: **F** read from a file or command output, **A** arithmetic, **H** estimate or interpretation. Nothing ran on a
cluster; no cluster output is quoted. Branch `codex/axis2-readout-20261012` (pushed).

## Questions for Valerian

1. Run the HoreKa blocks below (one CPU allocation of at most 45 minutes on an accelerated node) to add the agent's
   queries (Q) and the answer text (A); everything else is done on the Mac.
2. The random-direction control (below) bears on how the body reads the size of the axis-1 stage slopes. Decide whether
   the body needs a sentence; this task kept the body to one pointer sentence.
3. Side finding: `tab:axis-checks` (A2) prints the evaluation row of "Spearman of axis 1 with each rubric score" as
   -0.061 / -0.152; those are the battery's axis_1_linear predictions. Axis 1 itself correlates +0.061 (Qwen) and +0.151
   (Mistral, aligned) with the evaluation score on the 3,570 confirmation items [F]. Relabel the row or print the direct values.

## What exists where [F]

| Input | Mac | HoreKa |
|---|---|---|
| Maps (both views) | `~/Hamburg/geodml-inputs/topic/maps/{qwen,mistral}` (sha256 equal to the archive manifests) | `$W/geoaxis-archive/maps` |
| Battery | `analysis/paper/arr_oct2026_first_draft/evidence/saved-summaries/battery__readiness_robustness_battery.json` (sha256 = battery manifest) | `$W/geoaxis-archive/battery` |
| Prompt embeddings, 26,009 x 4,096, both views | `~/Hamburg/geodml-inputs/topic/prompts/final-audit/projections` | `$W/geoaxis-archive/final-audit/projections` |
| Page embeddings, 21,384, both views | `ARR_ACL_CycleOct2026/hf-min/derived/snippet-embeddings/snippet-embeddings-corpus-v1` | `$R/snippet-embeddings-corpus-v1` |
| Stage rows, 624,132 answers | `~/Hamburg/geodml-inputs/release-traces-hub/derived/release-traces-v1` | `$RUN/release-traces-v1` |
| Prompt-text replay, 26,008 prompts x 2 engines | `~/Hamburg/geodml-inputs/funnel-replay-v1` (reproduces the stored R0 exactly) | `$RUN/funnel-replay` |
| Calibration coordinates and labels (21,458 items) | `~/Hamburg/geodml-inputs/axis2-calibration-v1` (downloaded 2026-10-12 from the private `geoaxis-prompts-generation-26k`, revision 74e64ff6; restricted-local texts) | `$W/geoaxis-archive/maps/*` (probe) |
| Queries, 304,567 texts: raw axis 1 and 2 per view, no embeddings | none | `$RUN/embed-queries-{qwen,mistral}.merged/question_projections.jsonl` |
| Answers, 207,354 texts: raw axis 1 and 2 per view, no embeddings | none | `$RUN/embed-answers-{qwen,mistral}.merged/question_projections.jsonl` |

The full-run embedding jobs did not keep query or answer vectors (`--save-embeddings` off), so random directions cannot be
read on Q and A anywhere without re-embedding (GPU; not proposed for the paper).

## Code [F]

`analysis/scripts/axis2_readout.py` (`run`, `texts`, `macros`), `analysis/tests/test_axis2_readout.py` (7 synthetic tests),
`analysis/docs/horeka-axis2-readout.sh`. Reuses `steelman.chain.chain`, `within_keyword_ols`, `steelman.tables.stage_values`
and `replay_values`, `intent_stages.keyword_draws` (seed 20261007, keyword codes in the funnel order) and `shuffle_draws`
(seed 20261008), `PromptScale`, the battery alignment. Random directions: five per view, seed 20261012, orthogonal to
span(a_v1, a_v2) to 3e-17. Tests: 77 pass (7 new, 70 related existing: intent stages, release export, steelman, page readiness, paper numbers, answers).

Checks of the Mac run [F]: the axis-1 readout reproduces the stored held-out chain (slopes, 95% intervals, answer counts,
four cells) to 7e-17; stage values to 2e-14; x exactly; page positions to 3e-14 once the page scale uses all 26,009
embedded prompts (the slim 26,008-row final-axis-map copy on the Mac is off by up to 3.5e-5; fixed in 76b9c1a); the
battery's axis-2 confirmation agreement exactly.

## Results, Mac run `analysis/out/axis2/mac-1fef39e` (commit 1fef39e, 881 s) [F]

- Sign: the Qwen view's a_2 loads -2.68 on evaluation, so axis 2 is flipped; the Mistral view agrees after rotation.
- Two-view agreement of axis 2 (Spearman): confirmation items 0.696 (Pearson 0.743), prompts 0.900, pages 0.870
  (axis 1 on pages 0.742). Queries, answers: pending HoreKa.
- Rubrics, confirmation items (3,570): evaluation +0.469, selection commitment +0.163, action implementation -0.140,
  information seeking reversed -0.034, overall -0.027 (development, in sample: +0.735, +0.003, -0.531, -0.420, -0.435).
- Relation to x: Spearman -0.588, within keyword -0.642; binned mean of x2 rises from 0.58 (x < 0.05) to 0.73
  (0.20-0.25) and falls to 0.18 (x >= 0.95); within-keyword quadratic R^2 0.458.
- Within-keyword spread: median SD 0.26 (x: 0.29), IQR 0.40 (0.49); range >= 0.3 in 100% of keywords (x 100%),
  >= 0.5 in 99.1% (x 100%); 82% of the variance within keywords (x 98%). Writer correlation -0.05 (x -0.05).
- Stage slopes, held-out, percentile units, x and x^2 as controls (LP, LR, QP, QR): R0 +0.129, +0.130, +0.130, +0.127;
  R +0.072, +0.050, +0.101, +0.101; P +0.105, +0.090, +0.113, +0.126; I +0.122, +0.095, +0.119, +0.121;
  K +0.142, +0.110, +0.125, +0.122 (all 95% intervals exclude 0; raw slopes within 0.03 of these; shuffle p 0.005).
  Axis 1 (stored chain): R0 +0.014 to +0.017, K +0.058 to +0.094.
- SD units at K: axis 1 +0.093, +0.075, +0.057, +0.071; axis 2 (controls) +0.152, +0.120, +0.136, +0.134; ten random
  directions (no controls) median +0.070, +0.057, +0.054, +0.061, largest 0.094, 0.082, 0.080, 0.074.
  Axis 2 lies above all ten in 20/20 stage-cell pairs; axis 1 inside their range in 20/20, below their median at R0 in
  4/4 cells and above it at K in 4/4.
- Reading [H]: the size of a stage slope does not single out intent; part of any prompt direction is mirrored in the pages
  shown and cited. What distinguishes directions is where the slope arises: axis 2 at the prompt's own words (R0), axis 1
  later (queries, shortlist).

## Name [H]

"Weighing options" (high) versus "carrying out a task" (low). Evidence and limits: `analysis/out/axis2/naming_evidence.md`.
No person rated axis 2; the population was written along axis 1 only.

## Paper material (not integrated; the paper session integrates it)

`analysis/out/axis2/{paper_paragraph.tex, paper_table.tex, numbers-axis2.tex, method_box_sentence.tex, naming_evidence.md}`
(untracked; every number is a macro; compiled clean with the paper's acl.sty in a scratch document). Regenerate the macro
and table files with `python3 -m analysis.scripts.axis2_readout macros --results <results.json> --output numbers-axis2.tex --table paper_table.tex`.
No source file of `paper_v2-revision-oct26` was edited; `check_paper.sh` ran there once as a baseline (GATE: PASS) and
rebuilt its PDF and LaTeX build files.

## HoreKa (paste into an open login shell; no ssh prefix)

Resources [H from F]: the Mac run took 881 s single-threaded (observed RSS about 4.5 GB). On HoreKa the export adds
2-8 minutes and Q and A add about 10%; estimate 17-38 minutes. Request: accelerated, 1 GPU (unused; the partition needs
one), 19 CPUs (one used), 32 GB, 45 minutes, reservation casualnet: at most 0.75 GPU-hours and 14.25 core-hours.
Cheaper: `TEXTS_ONLY=1` with `-t 00:15:00` (export only) and the readout on the Mac. Admission: fewer than five of
your allocations and at least ten minutes since the last observed start (block 1 prints the queue).

PIN = `3f648cef1b37bc8d78e6fc76c11219654a6b89f5` (the final code commit; later commits on the branch change documentation only).

Block 1, login node: variables, admission check and read-only probe (nothing is written)

```bash
export W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen
export R=$W/reviews/page-readiness-20261004 RUN=$W/reviews/page-readiness-20261004/fullrun-v1/run2 A=$W/geoaxis-archive
export PIN=3f648cef1b37bc8d78e6fc76c11219654a6b89f5
export CODE=$W/checkouts/axis2-$PIN
squeue -u $USER -o '%.10i %.9P %.22j %.8T %.10M %.10l %.20S'
df -h $W | tail -n 1
scontrol show reservation casualnet | head -n 4
ls -la $RUN/release-traces-v1 $RUN/funnel-replay $RUN/steelman-confirmation/chain.json $RUN/answers-export/observations.jsonl.gz
ls -la $RUN/embed-queries-qwen.merged $RUN/embed-queries-mistral.merged $RUN/embed-answers-qwen.merged $RUN/embed-answers-mistral.merged
ls -la $R/funnel-features-v1/rows.parquet $R/snippet-embeddings-corpus-v1 $R/snippet-embeddings-corpus-v1/embeddings $W/containers/geodml-cpu.sif
ls -la $A/maps/qwen $A/maps/mistral $A/battery $A/final-audit/final-axis-map.jsonl $A/final-audit/projections/qwen $A/registration/population-registration-v1/population-prompts.jsonl
du -sh $RUN/embed-queries-*.merged $RUN/embed-answers-*.merged $RUN/release-traces-v1 $R/snippet-embeddings-corpus-v1
wc -l $A/final-audit/final-axis-map.jsonl $RUN/embed-queries-qwen.merged/question_projections.jsonl $RUN/embed-answers-qwen.merged/question_projections.jsonl
python3 -c 'import json,gzip,os,sys
R,A=os.environ["RUN"],os.environ["A"]
first=lambda p: json.loads((gzip.open if p.endswith(".gz") else open)(p,"rt").readline())
for k in ("queries","answers"):
    for v in ("qwen","mistral"):
        r=first(f"{R}/embed-{k}-{v}.merged/question_projections.jsonl"); print(k,v,sorted(r),sorted(r["projection"]))
print("population",sorted(first(f"{A}/registration/population-registration-v1/population-prompts.jsonl")))
print("observations",sorted(first(f"{R}/answers-export/observations.jsonl.gz")))
for v in ("qwen","mistral"):
    for n in ("readiness_supervised_subspace_coordinates.jsonl","readiness_consensus.jsonl"):
        p=f"{A}/maps/{v}/{n}"; print(p,os.path.exists(p))'
```

Block 2, login node: pinned checkout of exactly 3f648cef1b37bc8d78e6fc76c11219654a6b89f5 and a free output folder

```bash
BASE=$(ls -d $W/checkouts/*/ | head -n 1)
git -C "$BASE" fetch origin codex/axis2-readout-20261012
git -C "$BASE" worktree add --detach "$CODE" "$PIN"
test "$(git -C "$CODE" rev-parse HEAD)" = "$PIN" && test -z "$(git -C "$CODE" status --porcelain)" && echo CHECKOUT OK
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-${PIN:0:7}
export OUT=$W/runs/axis2-readout/$RUN_ID
mkdir -p $W/runs/axis2-readout && test ! -e $OUT && echo "OUTPUT FREE $OUT"
```

Block 3, the one allocation (a child shell; the step ends the allocation; at most 45 minutes)

```bash
bash -c 'srun -p accelerated -A hk-project-p0026831 --reservation=casualnet --gres=gpu:1 -n1 -c 19 --mem=32G -t 00:45:00 \
  -J axis2-readout bash "$CODE/analysis/docs/horeka-axis2-readout.sh" "$OUT"; echo "srun exit $?"'
```

Block 4, status and logs (login node)

```bash
cat $OUT/run_manifest.json; tail -n 25 $OUT/stdout.log; tail -n 25 $OUT/stderr.log
sacct -n -X --name=axis2-readout -S today --format=JobID,State,Start,End,Elapsed,AllocCPUS,MaxRSS
ls -la $OUT/results/texts $OUT/results/readout; du -sh $OUT
python3 -c 'import json,os; o=os.environ["OUT"]; t=json.load(open(o+"/results/texts/manifest.json")); print(t["counts"])
r=json.load(open(o+"/results/readout/results.json")); c=r["checks"]
print(c["chain_values_max_abs_difference"], c.get("text_values_axis1_z_max_abs_difference"), c.get("text_values_nan_mismatch"), c["stored_chain_heldout"])
print({k: v["view_agreement"]["axis2"] for k, v in r["positions"].items()})'
```

Block 5, login node: upload the sealed results to the private Hub (write token at a hidden prompt; the face-validity file stays on HoreKa)

```bash
source $W/geodml-nemotron-env.sh
read -rsp "HF write token: " HF_TOKEN; echo
HF_TOKEN=$HF_TOKEN "$RT/bin/python" -c "import os; from huggingface_hub import HfApi; print(HfApi(token=os.environ['HF_TOKEN']).upload_folder(repo_id='ValerianFourel/geodml-experiment-v2-paper-private', repo_type='dataset', folder_path='$OUT', path_in_repo='derived/axis2-readout/$RUN_ID', ignore_patterns=['**/face-validity.txt'], commit_message='axis2-readout $RUN_ID'))"
unset HF_TOKEN
```

After the upload, on the Mac: download `derived/axis2-readout/<RUN_ID>` and either use its `results/readout/{numbers-axis2.tex,
table-axis2.tex}` or rerun `run ... --text-coordinates <download>/results/texts/text_coordinates.parquet` locally, then
regenerate the macro files and uncomment the Q and A sentence in `paper_paragraph.tex`.
