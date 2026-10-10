# Anonymous release of code and data for "Measuring Where Intent Enters" (ACL ARR, October 2026)

Written 2026-10-10 by the release session (a branch of the paper-revision conversation); second round the same day,
after Valerian asked for "a summary of all traces and results that can reconstruct Figure 3": a preprocessed version
with the processed traces, and a link to the full dataset. Evidence labels: **F** read from a file or command output;
**P** plan; **H** estimate. Nothing was pushed or uploaded: the authors follow `intent-release/UPLOAD_STEPS.md`.

## 1. Where it is [F]

| What | Where |
|---|---|
| Release repository: one root commit `fed8482`, author and committer `Anonymous <anonymous@example.com>`, both dates `2026-10-10 00:00 +0000` | `ARR_ACL_CycleOct2026/intent-release/` |
| Reviewer archives built from that commit (before the cluster export) | `dist/code.zip` (33.7 MB), `dist/data.zip` (24.2 MB); digests in `dist/MANIFEST.md` |
| Dataset folder (local traces only until the export is integrated) | `dist/hf_dataset/` |
| Export of the processed traces, for HoreKa (not committed: see section 4) | `.worktrees/acl-figures/analysis/scripts/export_release_traces.py` and `analysis/tests/test_export_release_traces.py` |
| Integration on the Mac after the export | `intent-release/.release-work/integrate_traces.sh <export folder>` |

## 2. What changed in this round [F]

- **Data tiers** (README, section Data). Sample in the repository (20 exploration keywords, under 20 MB); reviewer pack
  `data.zip` (preprocessed, under 50 MB): the small tables, `answers_natural`, and `trace_answers_natural` (float32, one row
  per natural answer of both generators); full dataset on the Hub (link placeholder `<hf-user>/<hf-dataset>`): every
  trace table, all three conditions, float64. Raw traces (about 110 GB of JSON with infrastructure paths) stay private.
- **`trace_answers`, the summary of all traces.** One row per answer (624,132): x, keyword, split, page counts per stage,
  the values Figure 3 plots (Q, R0, Rk, R, C, P, K, A on the consensus-z scale; R_u, C_u, P_u, K_u on the prompt scale,
  exactly as `intent_stages_study.py analyze` computes them) and the stage chain's values `chain_R0` ... `chain_K` (exactly
  as `steelman/tables.py` computes them). Also `queries` (304,567 distinct agent queries with their axis positions) and
  `answer_queries` (1,631,398 links), plus `stage_items`, `rerank_events`, `stage_u`, `shown` for both generators.
- **Checks.** `verify.py` B13: every bin mean, keyword-clustered SE and count of Figure 3's stored curves recomputed from
  `trace_answers` (1e-12; pack, float32: 1e-6). B10 now recomputes the chain of the **held-out** run (733 keywords) as well
  as the exploration run (278) from the chain columns, with the chain's own functions, and checks the columns against the
  trace rows. `make figure3` redraws the figure from the recomputed curves; `make rerun-chain SPLIT=confirmation` reruns
  the held-out chain end to end.
- **The export is safe to run on the cluster code.** Since `cf5c452` (the commit of the stored intent-stages and chain
  runs) neither `steelman/` nor `funnel_study.py` nor `intent_stages_study.py` changed; `intent_stages.py` changed only in
  `selection_rows` (the reranker fix of `de5d8fb`), not in the stage values or curves. `--relocation` only adds a validity
  record, `--gemma` only the development stage G, which the check skips.
- **Local results.** Release tests: all pass (`pytest tests`). `verify.py` on the local dataset folder: PASS 134, FAIL 0,
  KNOWN 3, INFO 30, SKIP 7; reviewer pack PASS 133, FAIL 0, KNOWN 3; sample PASS 38, FAIL 0, KNOWN 3. The local exploration
  chain agrees with its stored run at 1e-9 in every slope, share and count, and `make rerun-chain` reproduces all 1,118
  numbers of that run. Export tests (synthetic full run): the export reproduces the stored stage curves with a maximum
  difference of 0.0, and its chain values equal `steelman.tables.load` by answer id.
- **Pack budget.** Float columns of distinct values are no longer dictionary-encoded (it doubled them); a synthetic
  `trace_answers_natural` of 208,044 rows then takes 18.8 MB [H for the real one], so the integrated pack should be about
  45 MB. Without the cluster export the pack carries the Qwen exploration rows instead, for the local chain check.

## 3. Findings the paper must handle [F] (unchanged)

1. `\MultiSnippetURLs` = 15%; the snapshot gives 20.0% of shown URLs. P: change to 20\%.
2. `\NPrompts` = 26,009; the archive holds 26,008 (axis rank 2,636 absent; x is exact). P: keep with a sentence, or 26,008.
3. Capture location: 138 of 23,767 rows mention Hamburg, 433 are `.de` URLs; released as study data, flagged.
4. The second package in `paper_v2-revision-oct26/release/` cannot detect a wrong held-out number. P: submit `intent-release/`.

## 4. Next: the cluster export [P]

### 4.1 Mac: commit and push the export (Valerian; this session's guard refuses git in this worktree)

```bash
cd ~/Hamburg/GEODML_Unified/.worktrees/acl-figures
python -m pytest -q analysis/tests/test_export_release_traces.py          # 3 passed
git diff analysis/docs/handoff/README.md                                  # only the 2026-10-10 line
git add analysis/scripts/export_release_traces.py analysis/tests/test_export_release_traces.py \
        analysis/docs/handoff/2026-10-10_anonymous-release_handoff.md analysis/docs/handoff/README.md
git commit -m "Release export: per-answer trace summary (Figure 3 and stage-chain values) and trace tables; handoff"
git push origin revision-oct26:codex/acl-figures-20261006
git rev-parse HEAD                                                        # PIN for HoreKa
```

The worktree's other uncommitted edits (`acl_figures.py`, `paper_numbers_fullrun.py`) belong to the paper session; the
commit above leaves them out.

### 4.2 HoreKa: estimate and allocation

- Inputs [F, manifests]: 624,132 answers; 22,085,285 retrieval items; 27,239,437 scored candidates; 7,334,994 own-keyword
  rows; 3,384,383 shown rows; 304,567 distinct queries.
- Prior timing [F]: the steelman chain on the same funnel extract took 67 s (exploration) and 127 s (confirmation) on CPU.
- Estimate [H]: 10–30 minutes, peak memory 10–30 GB, output 0.8–1.5 GB (`rerank_events` dominates).
- Allocation: one allocation-bearing `srun` on `accelerated` (project rule: never cpuonly), 1 GPU (unused; the partition
  needs one), 19 cores, 120 GB, 45 minutes, reservation `casualnet`: at most 0.75 GPU-hours. It is released when the step
  ends. Cheaper alternative: `dev_accelerated` with the same request, if free. Admission: fewer than five allocations of
  yours running and about ten minutes since the last start (block 1 prints `squeue`).

### 4.3 HoreKa: commands (an open login shell; run block 2 inside tmux if the connection may drop)

Block 1, login node: variables, pinned checkout, inputs.

```bash
export W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen
export R=$W/reviews/page-readiness-20261004 RUN=$W/reviews/page-readiness-20261004/fullrun-v1/run2
export PIN=<full sha from git rev-parse HEAD>
export CODE=$W/checkouts/fullrun-$PIN OUT=$RUN/release-traces-v1 LOGS=$RUN/release-traces-logs
squeue -u $USER -o '%.10i %.9P %.22j %.8T %.10M %.10l %.20S'
df -h $W | tail -n 1
BASE=$(ls -d $W/checkouts/*/ | head -n 1)
git -C "$BASE" fetch origin codex/acl-figures-20261006
git -C "$BASE" worktree add --detach "$CODE" "$PIN"
test "$(git -C "$CODE" rev-parse HEAD)" = "$PIN" && test -z "$(git -C "$CODE" status --porcelain)" && echo CHECKOUT OK
ls -d $RUN/trace-extract $RUN/intent-replay $RUN/embed-queries-qwen.merged $RUN/embed-queries-mistral.merged \
  $RUN/embed-answers-qwen.merged $RUN/embed-answers-mistral.merged $RUN/answers-export/observations.jsonl.gz \
  $RUN/funnel-extract $RUN/funnel-assembled $RUN/funnel-replay $RUN/intent-stages/results.json \
  $R/snippet-embeddings-corpus-v1 $W/geoaxis-archive/battery $W/geoaxis-archive/final-audit/final-axis-map.jsonl \
  $W/containers/geodml-cpu.sif
test ! -e $OUT && mkdir -p $LOGS && echo "OUTPUT FREE"
```

Block 2, the export (one allocation, at most 45 minutes; exit 0 = Figure 3 reproduced from `trace_answers`, 3 = not).

```bash
srun -p accelerated -A hk-project-p0026831 --reservation=casualnet --gres=gpu:1 -n1 -c 19 --mem=120G -t 00:45:00 -J release-traces \
  bash -c 'cd $CODE && apptainer exec --no-home --bind $W --env GEODML_GIT_COMMIT=$PIN,PYTHONPATH=$CODE $W/containers/geodml-cpu.sif \
    python -u -m analysis.scripts.export_release_traces --run $RUN --corpus-package $R/snippet-embeddings-corpus-v1 \
      --battery $W/geoaxis-archive/battery --final-axis-map $W/geoaxis-archive/final-audit/final-axis-map.jsonl --output $OUT; \
    echo "export exit $?"' 2>&1 | tee $LOGS/export-$(date +%Y%m%dT%H%M%S).log
```

Block 3, login node: results, then the upload to the private Hub (only after `"passed": true`).

```bash
cat $OUT/figure_check.json
python3 -c "import json; m = json.load(open('$OUT/manifest.json')); print({k: v['rows'] for k, v in m['tables'].items() if isinstance(v, dict)}); print(m['validity'].get('chain'), m['seconds'])"
du -sh $OUT
source $W/geodml-nemotron-env.sh
"$RT/bin/python" -c "from huggingface_hub import HfApi; print(HfApi().upload_folder(repo_id='ValerianFourel/geodml-experiment-v2-paper-private', repo_type='dataset', folder_path='$OUT', path_in_repo='derived/release-traces-v1', commit_message='release-traces-v1 from $PIN'))"
```

Expected [P]: `trace_answers` 624,132 rows, `stage_items` 22,085,285, `rerank_events` 27,239,437, `stage_u` 7,334,994,
`shown` 3,384,383, `queries` 304,567, `answer_queries` 1,631,398; chain alignment `trace_answers_without_funnel_answer`
0, `keyword_mismatches` 0, `x_max_abs_difference` 0.0. If the figure check fails (exit 3), keep the folder and send
`figure_check.json`: the tables are written for inspection, and the Mac integration refuses them.

### 4.4 Mac: integrate, recommit, package, clean room (`UPLOAD_STEPS.md` steps 0a.3, 1 and 1b)

```bash
hf download ValerianFourel/geodml-experiment-v2-paper-private --repo-type dataset \
    --include "derived/release-traces-v1/*" --local-dir ~/Hamburg/geodml-inputs/hub
cd ~/Hamburg/GEODML_Unified/ARR_ACL_CycleOct2026/intent-release
bash .release-work/integrate_traces.sh ~/Hamburg/geodml-inputs/hub/derived/release-traces-v1
```

Then check the summary it prints (FAIL 0 in all three `verify.py` runs, Figure 3 passed for dataset and pack, both chain
reruns equal), check that the pack stays under 50 MB and the sample under 20 MB (the scripts refuse otherwise), compare
the row counts of `docs/DATA_CARD.md` with `dist/hf_dataset/trace_export.json`, then amend, package and run the clean room.

## 5. Not done, and why [F]

- The cluster export has not run (needs the push in 4.1 and Valerian's HoreKa session). Until it is integrated the
  archives in `dist/` are an intermediate build: B13 and the held-out B10 are SKIP, and the README's data tiers describe
  the integrated state.
- No push, no public upload, no account (task rule).
- This handoff and the export script are not committed (worktree guard; see 4.1).
