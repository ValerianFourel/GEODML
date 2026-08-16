# Phase 1 semantic-readiness freeze on Jupyter-JSC

This runbook prepares and executes only **Phase 1 — freeze semantic corpora**
from the semantic-readiness master specification. It performs CPU data parsing,
hashing, deterministic sampling, and manifest validation. It does not run a
judge, LLM2Vec, reranking, or any production model inference.

The active experiment remains the query-free decision-readiness experiment in
`analysis/docs/query_free_decision_readiness_axis.md`. The Phase-1 gate must be
green before its readiness-annotation stage begins.

## Expected outcome

There are two deliberate checkpoints:

1. The five-open-source rehearsal must reproduce 4,370 exact-unique transfer
   prompts and end at `phase1-transfer-snapshots-pending` because MS MARCO,
   WildChat, and LMSYS are absent.
2. The final run may end at `phase1-complete` only after all eight exact source
   snapshots are locally present, revision-pinned, and handled under their
   access and redistribution terms.

Do not substitute another dataset for a missing source. Do not continue to
Phase 2 from the rehearsal result.

## 1. Start the Jupyter service

Use [Jupyter-JSC](https://jupyter.jsc.fz-juelich.de/) with a **JSC Account**;
the federated Helmholtz-ID login does not provide HPC access. Select JUPITER,
JupyterLab 4.3, the correct compute project, and the smallest CPU-only resource
configuration available. Phase 1 does not need a GPU.

JUPITER is still documented as an early-access system. In the Jupyter terminal,
store the repository and persistent inputs under `$PROJECT`, not in `$HOME` or
temporary notebook-job storage. The current JSC documentation says `$HOME` and
`$PROJECT` are available on compute nodes and recommends `$HOME` only for small
user-specific files.

At the time of this runbook update, the JUPITER login banner reported active
filesystem-instability work and announced automatic cleanup for `$SCRATCH`
(90 days) and `$FSCRATCH` (30 days). Keep code, frozen inputs, manifests, and
accepted outputs in `$PROJECT`. Use scratch only for reproducible caches in
later GPU phases. Do not start while `Stale file handle` errors are present or
during an announced filesystem outage.

Official references:

- <https://jupyterjsc.pages.jsc.fz-juelich.de/docs/jupyterjsc/>
- <https://jupyterjsc.pages.jsc.fz-juelich.de/docs/jupyterjsc/authentication/>
- <https://jupyterjsc.pages.jsc.fz-juelich.de/docs/jupyterjsc/users/jupyterlab/4.3/>
- <https://apps.fz-juelich.de/jsc/hps/jupiter/environment.html>

Use a Jupyter terminal for the commands below so the full command history and
errors remain visible. If the default Python lacks `pyarrow`, select the
preinstalled PyHPC kernel/environment or follow JSC's documented HPC virtualenv
kernel setup. Do not install packages into an untracked ad-hoc environment and
then treat that as the scientific run environment.

## 2. Commit and publish from the local computer

The cluster run must use committed code. From the local `geodml-mono` checkout,
finish and test the Phase-1 changes, commit them, push them, and record the full
SHA:

```bash
git status --short
git rev-parse HEAD
```

The ignored corpus inputs are not transferred through Git. Package only the
exact Phase-1 inputs; do not copy nested upstream `.git` directories, the
unused Amazon product table, or the full 3.5 GB acquisition tree. The selected
open-source input set is about 832 MB before compression and contains only a
few hundred files.

On the Mac, build one checksummed transfer archive. The published open-data
bundle for this rehearsal must have the following digest:

`f0422481b094da8f8e4db5e0d4ed7668fed603e7857db0b53069deac0387d344`

```bash
export GEODML_LOCAL_REPO=/absolute/path/to/geodml-mono
export PHASE1_BUNDLE=/tmp/geodml-semantic-readiness-phase1-open-20260817.tar.gz

tar -C "$GEODML_LOCAL_REPO/analysis/output" -czf "$PHASE1_BUNDLE" \
  query_free_surface_corpus_v1/raw \
  query_free_surface_corpus_v2 \
  semantic_readiness_web_v2 \
  semantic_readiness_corpus_v3 \
  semantic_readiness_transfer_sources_v1/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz \
  semantic_readiness_transfer_sources_v1/_upstream_git/ccpe/data.json \
  semantic_readiness_transfer_sources_v1/_upstream_git/taskmaster/TM-1-2019 \
  semantic_readiness_transfer_sources_v1/google-schema-guided-dialogue \
  semantic_readiness_transfer_sources_v1/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet

if tar -tzf "$PHASE1_BUNDLE" | grep -q '/\.git/'; then
  echo "Unexpected nested Git metadata in Phase-1 bundle" >&2
  exit 1
fi
shasum -a 256 "$PHASE1_BUNDLE"
ls -lh "$PHASE1_BUNDLE"
```

Publish the archive to the public Hugging Face dataset repository only after
the digest matches. Keep it under a versioned path and record the immutable Hub
commit returned by the upload. Do not use `main` as the scientific input
revision.

```bash
export HF_DATASET_REPO=ValerianFourel/geodml-papersize
export PHASE1_BUNDLE_REPO_PATH=data/semantic_readiness/phase1-open-20260817/geodml-semantic-readiness-phase1-open-20260817.tar.gz

hf auth whoami
hf upload "$HF_DATASET_REPO" \
  "$PHASE1_BUNDLE" \
  "$PHASE1_BUNDLE_REPO_PATH" \
  --repo-type dataset \
  --commit-message "Add open semantic-readiness Phase-1 bundle"
```

The archive contains only the open-source rehearsal inputs. It does not grant
a new blanket license: the source-specific licenses and attribution retained
inside the manifests continue to apply.

On the JUPITER login node, activate the project and create only the persistent
layout. Record the printed absolute staging path:

```bash
jutil user projects -u "$USER" -o columns
jutil env activate -p '<project-id>'
: "${PROJECT:?Project activation did not export PROJECT}"
export GEODML_PROJECT_ROOT="$PROJECT/$USER/geodml"
mkdir -p "$GEODML_PROJECT_ROOT/staging" "$GEODML_PROJECT_ROOT/src" \
  "$GEODML_PROJECT_ROOT/runs"
printf 'JSC_STAGING=%s\n' "$GEODML_PROJECT_ROOT/staging"
```

Download the public bundle over HTTPS on the JUPITER login node. Pin the exact
Hugging Face commit produced above, write through a temporary filename, and
verify the content digest before accepting the file:

```bash
export HF_DATASET_REPO=ValerianFourel/geodml-papersize
export HF_DATASET_REVISION='<full-hugging-face-commit-sha>'
export PHASE1_BUNDLE_REPO_PATH=data/semantic_readiness/phase1-open-20260817/geodml-semantic-readiness-phase1-open-20260817.tar.gz
export PHASE1_BUNDLE="$GEODML_PROJECT_ROOT/staging/geodml-semantic-readiness-phase1-open-20260817.tar.gz"
export PHASE1_BUNDLE_SHA256=f0422481b094da8f8e4db5e0d4ed7668fed603e7857db0b53069deac0387d344

test ! -e "$PHASE1_BUNDLE"
curl --fail --location --retry 5 --retry-all-errors \
  --output "$PHASE1_BUNDLE.part" \
  "https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/${HF_DATASET_REVISION}/${PHASE1_BUNDLE_REPO_PATH}?download=true"
printf '%s  %s\n' "$PHASE1_BUNDLE_SHA256" "$PHASE1_BUNDLE.part" \
  | sha256sum --check -
mv "$PHASE1_BUNDLE.part" "$PHASE1_BUNDLE"
chmod 600 "$PHASE1_BUNDLE"
```

If HTTPS access to Hugging Face is temporarily unavailable, initiate an `scp`
fallback from the Mac because JUPITER restricts outbound SSH:

```bash
export JSC_LOGIN='<jsc-user>@login.jupiter.fz-juelich.de'
export JSC_STAGING='<absolute-path-printed-above>'

scp -i ~/.ssh/id_ed25519 "$PHASE1_BUNDLE" \
  "$JSC_LOGIN:$JSC_STAGING/"
```

Never put usernames, tokens, project IDs, or absolute infrastructure paths into
committed files.

MS MARCO, WildChat, and LMSYS must be acquired only after their terms and local
handling requirements have been reviewed. Restricted source text, derived
task banks, and judge responses remain in project storage; do not push them to
GitHub, Hugging Face, or another API.

## 3. Jupyter-terminal preflight

Set paths without hard-coding the project or account in the repository:

```bash
set -euo pipefail

: "${PROJECT:?Select/activate the JSC compute project before continuing}"
export GEODML_EXPECTED_COMMIT='<full-commit-sha>'
export GEODML_PROJECT_ROOT="$PROJECT/$USER/geodml"
export GEODML_REPO="$GEODML_PROJECT_ROOT/src/geodml-mono"
export READINESS_INPUT="/dev/shm/$USER/semantic-readiness-phase1/$GEODML_EXPECTED_COMMIT"
export READINESS_RUN="$GEODML_PROJECT_ROOT/runs/semantic-readiness-phase1/$GEODML_EXPECTED_COMMIT"
export PHASE1_BUNDLE="$GEODML_PROJECT_ROOT/staging/geodml-semantic-readiness-phase1-open-20260817.tar.gz"
export PYTHONDONTWRITEBYTECODE=1

test -s "$PHASE1_BUNDLE"
printf '%s  %s\n' \
  f0422481b094da8f8e4db5e0d4ed7668fed603e7857db0b53069deac0387d344 \
  "$PHASE1_BUNDLE" | sha256sum --check -
test ! -e "$READINESS_INPUT"
mkdir -p "$READINESS_INPUT"
chmod 700 "$READINESS_INPUT"
tar -xzf "$PHASE1_BUNDLE" -C "$READINESS_INPUT"

cd "$GEODML_REPO"
test "$(git rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
python3 -c 'import pandas, pyarrow; print("Phase-1 Python imports: OK")'
python3 analysis/scripts/freeze_semantic_readiness_phase1.py --help \
  | grep -- '--surface-source-input'
mkdir -p "$READINESS_RUN"
```

If the checkout is absent, clone or pull the repository over HTTPS and check
out the recorded SHA first. Do not run from an uncommitted notebook copy.

## 4. Five-source rehearsal

Define the exact staged inputs. The two `--surface-source-input` entries are
required because the original provenance records contain the paths of the Mac
that built the corpus; the override verifies the same bytes at their JSC paths.

```bash
export TRANSFER_ROOT="$READINESS_INPUT/semantic_readiness_transfer_sources_v1"

SURFACE_ARGS=(
  --surface-source-input "databricks-dolly-15k=$READINESS_INPUT/query_free_surface_corpus_v1/raw/databricks-dolly-15k.jsonl"
  --surface-source-input "anthropic-hh-helpful-base=$READINESS_INPUT/query_free_surface_corpus_v1/raw/train.jsonl.gz"
)

OPEN_TRANSFER_ARGS=(
  --source-input "openassistant-oasst1=$TRANSFER_ROOT/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz"
  --source-revision "openassistant-oasst1=fdf72ae0827c1cda404aff25b6603abec9e3399b"
  --source-input "google-ccpe-m=$TRANSFER_ROOT/_upstream_git/ccpe/data.json"
  --source-revision "google-ccpe-m=2c9cd30f33f3a154b5a27d015333679262ff36f5"
  --source-input "google-taskmaster-1=$TRANSFER_ROOT/_upstream_git/taskmaster/TM-1-2019"
  --source-revision "google-taskmaster-1=d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
  --source-input "google-schema-guided-dialogue=$TRANSFER_ROOT/google-schema-guided-dialogue"
  --source-revision "google-schema-guided-dialogue=e852981ae34990f4358979625854259302feaa78"
  --source-input "amazon-shopping-queries=$TRANSFER_ROOT/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
  --source-revision "amazon-shopping-queries=7916cdf6ab75a462e77f20ab40428a10923998d5"
)

OPEN_FREEZE_ARGS=(
  --transfer-source-input "openassistant-oasst1=$TRANSFER_ROOT/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz"
  --transfer-source-revision "openassistant-oasst1=fdf72ae0827c1cda404aff25b6603abec9e3399b"
  --transfer-source-input "google-ccpe-m=$TRANSFER_ROOT/_upstream_git/ccpe/data.json"
  --transfer-source-revision "google-ccpe-m=2c9cd30f33f3a154b5a27d015333679262ff36f5"
  --transfer-source-input "google-taskmaster-1=$TRANSFER_ROOT/_upstream_git/taskmaster/TM-1-2019"
  --transfer-source-revision "google-taskmaster-1=d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
  --transfer-source-input "google-schema-guided-dialogue=$TRANSFER_ROOT/google-schema-guided-dialogue"
  --transfer-source-revision "google-schema-guided-dialogue=e852981ae34990f4358979625854259302feaa78"
  --transfer-source-input "amazon-shopping-queries=$TRANSFER_ROOT/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
  --transfer-source-revision "amazon-shopping-queries=7916cdf6ab75a462e77f20ab40428a10923998d5"
)
```

Run deterministic extraction and the read-only freeze audit:

```bash
python3 analysis/scripts/build_semantic_readiness_dataset.py collect-transfer \
  --output-dir "$READINESS_RUN/transfer-open" \
  --maximum-per-source 1000 \
  --master-seed 20260817 \
  "${OPEN_TRANSFER_ARGS[@]}"

python3 analysis/scripts/freeze_semantic_readiness_phase1.py \
  --base-corpus "$READINESS_INPUT/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl" \
  --base-manifest "$READINESS_INPUT/semantic_readiness_corpus_v3/run_manifest.json" \
  --surface-corpus "$READINESS_INPUT/query_free_surface_corpus_v2/surface_coverage_corpus.jsonl" \
  --surface-manifest "$READINESS_INPUT/query_free_surface_corpus_v2/run_manifest.json" \
  --surface-provenance "$READINESS_INPUT/query_free_surface_corpus_v2/source_provenance.json" \
  --web-records "$READINESS_INPUT/semantic_readiness_web_v2/web_text_records.jsonl" \
  --web-manifest "$READINESS_INPUT/semantic_readiness_web_v2/run_manifest.json" \
  --web-raw-responses "$READINESS_INPUT/semantic_readiness_web_v2/raw_responses" \
  "${SURFACE_ARGS[@]}" \
  "${OPEN_FREEZE_ARGS[@]}" \
  --output-dir "$READINESS_RUN/freeze-open"
```

Validate the expected rehearsal result:

```bash
python3 - "$READINESS_RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
transfer = json.loads((root / "transfer-open/run_manifest.json").read_text())
freeze = json.loads((root / "freeze-open/phase1_corpus_freeze_manifest.json").read_text())
assert transfer["transfer_record_count"] == 4370
assert transfer["transfer_records_sha256"] == "f5f98202dbec61e23589b07ac9627ba1edba616f4f69cdd43f31b8da7ffd163a"
assert freeze["base_corpus"]["sha256"] == "c851c31f99bdfecd31238f36d6fe24d1e379a4ebf48c8dadc98ebfb2af1b26a8"
assert freeze["phase_gate"]["status"] == "phase1-transfer-snapshots-pending"
pending = {row["source_id"] for row in freeze["transfer_registry"]["sources"] if not row["frozen"]}
assert pending == {"microsoft-ms-marco-v1", "allenai-wildchat-1m", "lmsys-chat-1m"}
print("Open-source Phase-1 rehearsal: PASS; full Phase 2 remains blocked")
PY
```

These are data-integrity checks, not scientific results.

## 5. Final all-eight-source freeze

After the three access reviews are complete, set local-only snapshot paths and
immutable revisions in the Jupyter terminal:

```bash
export MS_MARCO_PATH='<local-ms-marco-query-tsv>'
export MS_MARCO_REV='<exact-release-or-content-hash>'
export WILDCHAT_PATH='<local-wildchat-parquet-directory>'
export WILDCHAT_REV='<exact-hugging-face-revision>'
export LMSYS_PATH='<local-lmsys-parquet-directory>'
export LMSYS_REV='<exact-hugging-face-revision>'
```

Repeat `collect-transfer` with `OPEN_TRANSFER_ARGS` plus the three new
`--source-input`/`--source-revision` pairs, writing to
`$READINESS_RUN/transfer-all`. Repeat the freeze audit with
`OPEN_FREEZE_ARGS` plus their three corresponding
`--transfer-source-input`/`--transfer-source-revision` pairs, writing to
`$READINESS_RUN/freeze-all`.

The final acceptance check is:

```bash
python3 - "$READINESS_RUN" "$GEODML_EXPECTED_COMMIT" <<'PY'
import json, pathlib, sys
root, expected_commit = pathlib.Path(sys.argv[1]), sys.argv[2]
transfer = json.loads((root / "transfer-all/run_manifest.json").read_text())
freeze = json.loads((root / "freeze-all/phase1_corpus_freeze_manifest.json").read_text())
assert transfer["git_commit_sha"] == expected_commit
assert freeze["git_commit_sha"] == expected_commit
assert transfer["omitted_source_ids"] == []
assert len(transfer["included_source_ids"]) == 8
assert freeze["phase_gate"]["status"] == "phase1-complete"
assert freeze["phase_gate"]["safe_to_begin_full_phase2"] is True
assert all(row["frozen"] for row in freeze["transfer_registry"]["sources"])
assert freeze["annotation_performed"] is False
assert freeze["embedding_performed"] is False
assert freeze["model_inference_performed"] is False
print("Phase 1 all-source freeze: PASS")
PY
```

Record the Jupyter job/allocation metadata and retain both run directories in
project storage. Stop here. Phase 2 is a separate cluster iteration and must
not begin merely because the notebook session is still running.
