# Qwen preparation on HoreKa

This milestone downloads models and frozen inputs. It submits no Slurm job,
reserves no shared hour and does not claim A100 compatibility.
Use the exact committed Git revision on both hosts. Keep runtime files outside
the clean checkout. [Resource snapshot](HOREKA.md).

## HoreKa: weights first

Run `analysis/scripts/slurm/horeka/prepare_qwen_downloads.sh` with
`GEODML_HOREKA_ACCOUNT` set to your project. It reuses the active `geodml-qwen`
workspace or requests a new 60-day workspace. It never restores or deletes
`ragdag`. `GEODML_WORKSPACE_NAME` selects another workspace name;
`GEODML_PYTHON` selects an installed Python >=3.10.

The script queries actual GPFS project-home and personal-work quotas, retains the
raw responses, and includes in-doubt bytes/files in its headroom calculation.
An unrecognized quota report stops preparation. Model inventory includes file
sizes and Hub SHA-256 or Git-blob checksums. Downloads require quota evidence at
most five minutes old and room for the missing files, largest temporary file,
and a further 10% or 5 GiB margin. Rerun the same script after interruption.
Corrupt completed files fail verification rather than being silently replaced.

Only these immutable repositories are downloaded:

| Component | Revision |
|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

BGE is the existing compactor. No other generation or judging weights are needed.
Models use the workspace cache; authentication remains in the existing saved HF
login or `HF_TOKEN`. Tokens are never put in preparation files.

The printed preparation directory contains `models.json`, `models-verified.json`,
`preparation.json`, quota/workspace reports, download logs, a transfer-environment
freeze and `environment.sh`. Source that environment file in later HoreKa blocks.
Check the actual workspace expiry in `workspaces.txt`; the legacy cluster is
undergoing migration, so a workspace lifetime is not a guarantee of cluster service.

## JUPITER: freeze and publish the inputs

First check job 1995245 and reconcile the current source dataset. Preserve pending
and running allocations. A fresh complete scheduler snapshot must show no legacy
GEODML jobs before staging and again immediately before publication. Unreconciled
claims or unverifiable completions stop staging. Do not repeat registration or
recovery acceptance, and do not reuse historical progress counts.

`analysis/scripts/prepare_agentic_qwen_inputs.py stage` accepts `--source`,
`--output`, `--runtime`, `--keyword-priority` and `--scheduler-snapshot`.
Use the saved runtime JSON from the prepared JUPITER attempt. The output must be
a separate mirror, never the live dataset. Preserve the original snapshot bytes,
prompts, selection records, keyword priorities and serving profile.

Staging copies intact Qwen registration shards, shared prompt/membership tables
and verified Qwen completion references. A mixed registration shard stays intact.
Unrelated result shards, models, live locks and forensic archives are omitted.
`artifacts/qwen-preparation.json` binds files, settings and the source profile.
Repeating an unchanged stage is safe; changed source data requires a new mirror.

Verify access to the existing private dataset
`ValerianFourel/geodml-experiment-v2-paper-private`. Use the existing
`manage_agentic_hours.py upload-inputs` on the mirror, with a durable journal and
fresh scheduler snapshot. Save the returned `input_bundle` and publication log.
The first successful upload establishes write access; public HF reachability
alone proves neither private read nor write permission.

## HoreKa: data and environment

`manage_agentic_hours.py download-inputs` accepts `--input-bundle`,
`--dataset-root`, `--quota-evidence` and optional `--stripes`.
It verifies bytes, imports verified terminal outcomes idempotently and reports
current missing work. It does not modify the remote hour registry. Use a new
dataset directory for a changed bundle; do not overwrite conflicting artifacts.

Run `prepare_agentic_qwen_inputs.py verify --dataset-root PATH --report PATH`
after download. This also checks the frozen input manifest. Retain this report
alongside the model verification report and the full committed Git SHA.

Then run `analysis/scripts/slurm/horeka/prepare_qwen_environment.sh` with
`GEODML_HOREKA_WORKSPACE`, `GEODML_QWEN_DATASET`, `GEODML_QUOTA_EVIDENCE` and,
if necessary, `GEODML_PYTHON`. It creates a separate x86_64 runtime using the
reference vLLM version and `sentence-transformers==6.0.1`, installs binary
packages only, runs `pip check`, and records the resolved dependency freeze.
Missing compatible wheels or Python versions stop setup; no automatic runtime
upgrade or source compilation is attempted.

The runtime is installed, not GPU-validated. Next obtain a fresh time approval
for the compatibility allocation. Before that run, add optional `casualnet`
reservation support to the shared-hour launcher. Verify the actual Slurm boundary
on the compute node before loading Qwen, without `unshare`. Preserve the frozen
scientific settings and obtain production approval only after measuring startup
and throughput. No production launcher is supplied by this preparation step.

## Full historical archive

`analysis/scripts/prepare_horeka_archive.py` downloads the full historical
`ValerianFourel/geodml-papersize` revision
`8ba12dcc592e41eda86f349ca3dde97d726e4b4b` independently of shared-hour inputs.
Pass `--root`, `--report-dir`, and fresh `--quota-evidence`; `--inventory-only`
reports size without downloading payloads. Keep it in a separate workspace
archive directory. The live inventory contained 41,911 files and 37,749,713,024
bytes when checked on 24 September 2026. Disk capacity must include transfer
metadata and safety margins.

Downloads resume per file and verify authoritative Hub checksums. Tar archives
remain compressed. Existing conflicting files stop the process. Raw paid API
errors are preserved, not counted as valid measurements. Archive files never
create task-ledger completions. Optional `--publish-catalog` publishes only the
pinned inventory to the private V2 repository and requires write access.

Models passed HoreKa preparation at commit `2ba0cf3`; this does not validate GPU
inference. The current read-only HF login can download inputs and archives but
must be replaced with appropriately scoped write access before publishing results
or claiming hours. Tokens stay in the host's saved HF login, never in configuration.

For initial publication and routine transfers, see the
[fast update workflow](../analysis/docs/agentic_shared_hours.md#fast-update-workflow).

For a shared multi-model bundle, set `GEODML_SHARED_INPUT_MANIFEST` to the
content-addressed manifest path printed by `pull`. The existing
`prepare_qwen_environment.sh` verifies that manifest and uses its frozen Qwen
profile/vLLM version. Without this variable it retains the original Qwen-only
manifest workflow. Neither path validates GPU inference during installation.
