# HoreKa runbook: the full run

Plan: `analysis/docs/handoff/2026-10-08_full-run-plan_handoff.md`; what it is for: `analysis/steelman/IMPORTANT.html`.
Approved by Valerian on 2026-10-08: CPU allocations of 6 hours on `cpuonly` (whole node), at most 4 at once, at least
10 minutes between observed starts, cap 8 CPU allocations; GPU allocations of 2 hours on `accelerated` (4 A100, the
validated LLM2Vec profile), cap 3. Every block below is for an already-open HoreKa login shell.

Estimate [H, from the Mac steelman run, the funnel handoff and the planner's per-task estimates]: about 630 CPU-hours of
tasks in total on the CPU path; with `"gpu_stats": true` (section 5b) about 60 CPU-hours plus 1–2 four-GPU allocations (both keyword splits run every part); one 76-core node delivers about 400 core-hours in 6 hours, so
3 to 5 CPU allocations; 1 GPU allocation of about 1.5 hours. New disk: 20–40 GB plus the Hub import (5–15 GB).

## 0. Variables and status

```bash
W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen; R=$W/reviews/page-readiness-20261004
PIN=<full commit sha given in the session reply>
CODE=$W/checkouts/fullrun-$PIN; FR=$R/fullrun-v1; LEDGER=$FR/ledger
squeue -u $USER -o '%.10i %.9P %.22j %.8T %.10M %.10l %.20S'
df -h $W | tail -n 1; ws_list 2>/dev/null | head -n 20
apptainer --version
ls -d $W/shared-hours/dataset $W/llama-hf/dataset $R/funnel-features-v1 $R/snippet-embeddings-corpus-v1 $W/geoaxis-archive/final-audit/final-axis-map.jsonl
```

## 1. Pinned checkout and the container

```bash
BASE=$(ls -d $W/checkouts/*/ | head -n 1)
git -C "$BASE" fetch origin codex/acl-figures-20261006
git -C "$BASE" worktree add --detach "$CODE" "$PIN"
test "$(git -C "$CODE" rev-parse HEAD)" = "$PIN" && test -z "$(git -C "$CODE" status --porcelain)" && echo CHECKOUT OK
mkdir -p $W/containers
(cd "$CODE/analysis/container" && apptainer build --fakeroot $W/containers/geodml-cpu.sif geodml-cpu.def)
srun -p dev_cpuonly -n1 -c 8 -t 00:15:00 apptainer exec --no-home --bind $W --env GEODML_GIT_COMMIT=$PIN,PYTHONPATH=$CODE \
  $W/containers/geodml-cpu.sif python -m pytest -q -p no:cacheprovider $CODE/analysis/tests/test_fullrun_ledger.py $CODE/analysis/tests/test_fullrun_merge.py
```

If `apptainer build --fakeroot` is refused: on the Mac `docker build -t geodml-cpu:1 analysis/container && docker save geodml-cpu:1 | gzip > geodml-cpu.tar.gz`,
copy the file to `$W/containers/`, then `gunzip geodml-cpu.tar.gz && apptainer build $W/containers/geodml-cpu.sif docker-archive://$W/containers/geodml-cpu.tar`.
Without Apptainer at all: `python3.12 -m venv $W/environment/fullrun-cpu && $W/environment/fullrun-cpu/bin/pip install -r $CODE/analysis/container/requirements-cpu.lock`
and submit with `RUNTIME=venv`.

## 2. Cells only on the Hub (login node; finite)

```bash
cd "$CODE"
$RT/bin/python -m analysis.fullrun hub inventory --cache $FR/hub
$RT/bin/python -m analysis.fullrun hub missing --cache $FR/hub --root $W/shared-hours/dataset --root $W/llama-hf/dataset
$RT/bin/python -c "import json; m=json.load(open('$FR/hub/missing.json')); print(m['bundle_count'], m['missing_unique_cells'])"
$RT/bin/python -m analysis.fullrun hub fetch --cache $FR/hub --import-root $W/hub-import/dataset 2>&1 | tail -n 5
$RT/bin/python -m analysis.fullrun hub missing --cache $FR/hub --root $W/shared-hours/dataset --root $W/llama-hf/dataset --root $W/hub-import/dataset
```

The last line must report no missing cells. A failing bundle is reported once and not retried; rerun `fetch` to retry
only the bundles still missing (resumable). Needs Hub read access on the login node (`HF_TOKEN` or `huggingface-cli login`).

## 3. Configuration and the task list

```bash
mkdir -p $FR
cp $CODE/analysis/fullrun/horeka-config.example.json $FR/config.json
DDG=$(find $W/shared-hours/dataset/artifacts $W/llama-hf/dataset/artifacts -name phase0_top20_ddg.parquet -type f | head -n 1)
SX=$(find $W/shared-hours/dataset/artifacts $W/llama-hf/dataset/artifacts -name phase0_top20_searxng.parquet -type f | head -n 1)
$RT/bin/python - $FR/config.json "$CODE" "$DDG" "$SX" <<'PY'
import json, sys
p, code, ddg, sx = sys.argv[1:]
c = json.load(open(p)); c["code"] = code; c["snapshots"] = {"duckduckgo": ddg, "searxng": sx}; c.pop("_comment", None); c.pop("_gpu_stats_comment", None)
json.dump(c, open(p, "w"), indent=1); print(json.dumps(c, indent=1))
PY
(cd "$CODE" && $RT/bin/python -m analysis.fullrun plan --config $FR/config.json --ledger $LEDGER)
(cd "$CODE" && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER)
```

Drop the `hub-import` sources from the config if block 2 found nothing missing. The ledger is planned once; a
change of plan uses a new ledger folder (`$FR/ledger-2`) and keeps the old one.

## 4. CPU wave (extraction first, then the analyses)

```bash
cd $FR && nohup env CODE=$CODE LEDGER=$LEDGER bash $CODE/analysis/docs/horeka-fullrun-wave.sh 3 > $FR/wave-$(date +%Y%m%d-%H%M).log 2>&1 &
tail -n 5 $FR/wave-*.log; squeue -u $USER -o '%.10i %.9P %.22j %.8T %.10M %.10l %.20S'
```

The first wave runs the 64 extraction shards, the merges, replay, assemble and the answer export, then every analysis
task whose inputs exist. Exit 4 in `$FR/slurm/cpu-<job>.out` means the deadline came with work left: after the job has
ended, run block 6 and admit again. Count every submission against the cap of 8.

## 5. GPU allocation (after `merge-trace` and `answers-export` are done)

```bash
(cd "$CODE" && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER) | grep -A 12 by_stage
sbatch --export=ALL,CODE=$CODE,LEDGER=$LEDGER --output=$FR/slurm/gpu-%j.out $CODE/analysis/docs/horeka-fullrun-gpu.sbatch
```

## 6. Monitor, reconcile, resume

```bash
(cd "$CODE" && $RT/bin/python -m analysis.fullrun reconcile --ledger $LEDGER && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER)
sacct -u $USER -S today --name=geodml-fullrun-cpu,geodml-fullrun-gpu -o JobID,State,Elapsed,MaxRSS,ExitCode
ls -t $LEDGER/logs | head; tail -n 30 $LEDGER/logs/<task>.<job>.log
ls $LEDGER/failed
```

`reconcile` releases only claims of allocations that Slurm reports ended. A task that failed twice stays failed: read its
log, fix the code on the Mac, commit, and plan a new ledger for what remains (never edit a running checkout).

## 5b. GPU statistics (optional; needs Valerian's explicit approval of the allocations)

The heavy fits (funnel stage models, generator decisions, steelman generator and fixed effects; about 570 of the
630 CPU-hours) can run on the PyTorch backend (`analysis/interpretability/pipeline/torch_fits.py`, PREREG computational
note C1). Plan the ledger with `"gpu_stats": true` in `config.json`; each heavy task then takes one A100, and a
4-GPU node runs four at a time. The CPU tasks (extraction, merges, cheap parts, assembly, reports) stay on `cpuonly`.
Only estimators that passed `analysis/fullrun/validation/REPORT.md` may be routed here.

Proposed (to approve): up to 3 allocations of `horeka-fullrun-gpu-stats.sbatch` (accelerated, 4 A100, 152 CPUs,
4 hours) = at most 48 GPU-hours; one 15-minute `dev_accelerated` check first. Estimate [H, to be replaced after the
first allocation by measured task times]: 1–2 allocations.

```bash
mkdir -p $W/containers
(cd "$CODE/analysis/container" && apptainer build --fakeroot $W/containers/geodml-gpu.sif geodml-gpu.def)
srun -p dev_accelerated --gres=gpu:1 -n1 -c 8 -t 00:15:00 apptainer exec --nv --no-home --bind $W \
  --env GEODML_GIT_COMMIT=$PIN,PYTHONPATH=$CODE $W/containers/geodml-gpu.sif \
  python -c "import torch,time; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); a=torch.randn(8000,8000,dtype=torch.float64,device='cuda'); torch.cuda.synchronize(); t=time.time(); (a@a).sum().item(); print('fp64 8000^3 s', round(time.time()-t,3))"
srun -p dev_accelerated --gres=gpu:1 -n1 -c 8 -t 00:15:00 apptainer exec --nv --no-home --bind $W \
  --env GEODML_GIT_COMMIT=$PIN,PYTHONPATH=$CODE $W/containers/geodml-gpu.sif \
  python -m pytest -q -p no:cacheprovider $CODE/analysis/tests/test_torch_fits.py
sbatch --export=ALL,CODE=$CODE,LEDGER=$LEDGER --output=$FR/slurm/gpustats-%j.out $CODE/analysis/docs/horeka-fullrun-gpu-stats.sbatch
```

The embedding tasks start with the relocation check: 512 archived prompts are re-embedded and must reproduce the
archived axis (`relocation-fresh.json`); if it fails, no query or answer is embedded and intent analysis waits.

## 7. Results to bring back

```bash
ls -la $FR/paper-results.tar.gz $FR/paper-results/report.md
(cd "$CODE" && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER) > $FR/final-status.json
tar -C $FR -czf $W/reviews/fullrun-$(date -u +%Y%m%dT%H%M%SZ).tar.gz paper-results.tar.gz final-status.json ledger/done ledger/failed ledger/checkpoint ledger/tasks.jsonl hub/missing.json slurm
```

On the Mac: `scp` the tarball, unpack, run `python3 -m analysis.steelman report --output <steelman-confirmation folder>`
if needed, then update `analysis/steelman/IMPORTANT.html`, `RESULTS.md` and the handoff.
