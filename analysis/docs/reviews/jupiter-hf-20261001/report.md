# JUPITER exports on Hugging Face, 1 October 2026

The recorded JUPITER export collections are present on Hugging Face. The general
archive and both GeoAxis exports pass the checks described below. Llama outcomes
are fully accounted for, including six terminal failures. The activation
repository contains 551.71 GB, including the previously outstanding full T7
chunks. This establishes remote publication to the stated depth. It does not
establish that nothing remains on JUPITER.

This was an independent, read-only audit from the Mac using saved authentication.
The five repository revisions stayed unchanged through the final check at
**2026-10-01 05:26:47 UTC**. No SSH, allocation, upload, deletion or registry
mutation ran. Sizes below use decimal GB. Counts include repository metadata
unless described as payload counts. Machine-readable results: [summary.json](summary.json).

| Collection | Files | GB | What was verified |
|---|---:|---:|---|
| General JUPITER archive | 44 | 20.20 | All 12 planned unit manifests agree with the uploaded plan; every archive part matches its recorded size and LFS SHA-256 |
| GeoAxis prompt generation | 272 | 9.66 | All 269 payload files match the saved manifest's sizes and SHA-256 hashes |
| GeoAxis readiness axis | 953 | 1.28 | All 950 payload files match the saved manifest's sizes and SHA-256 hashes |
| Probing activations | 8,022 | 551.71 | Remote inventory includes eight model/style probing folders and 6,488 full T7 chunk files, totaling 471.19 GB |
| Experiment V2 paper repository | 22,321 | 73.33 | Registry, Qwen index, 86 referenced bundle manifests and their object inventories checked |

## Generator publication and recorded errors

The Llama registry lists **310,380 planned cells: 310,374 completed and six
terminal failures**, with zero remaining cells, zero owned packages, no overlap
between successes and failures, and no outcomes outside the plan. Of 1,008
packages, 1,006 have status `complete`; two have status `blocked` because they
contain the six accepted failures. Those statuses do not indicate an active
owner or unaccounted work.

All six failed outcome records say `LLM transport failed` after two attempts.
This audit did not diagnose the underlying serving/transport cause or rerun them.
The registry's `failed` list correctly corresponds to bundle events named
`terminal_failed`, as specified by `finish_hours` in `agentic_hours.py`.

The JUPITER Qwen publication index contains **24,218 unique completed cells and
zero terminal failures across 26 bundles**. This is the JUPITER publication
scope, not the total Qwen experiment or the separate HoreKa backlog.

All 60 unique Llama checkpoint bundles and 26 Qwen bundles passed their canonical
manifest content hashes. Task identity hashes, outcome-to-registry joins and
record-reference shard names were checked. Every referenced data shard and
companion manifest is represented in its bundle. All **21,336 distinct referenced
objects**, totaling 59.93 GB, exist at the expected sizes. Their current or legacy
content-addressed paths were resolved using the repository's exchange convention.

For 2,062 LFS objects, Hub SHA-256 metadata also matches the bundle hash. The other
19,274 plain Git objects total 5.19 GB; their sizes and presence were checked,
but their contents were not downloaded to recompute SHA-256. Result rows were
not parsed against every record reference. The audit therefore verifies bundle
publication and metadata consistency, not a complete row-level restoration or
the scientific quality of generated answers.

## Archive and GeoAxis checks

The general archive covers **2,863,060 planned file/link entries and
97,996,995,691 original bytes** in 12 units. The compressed parts total
20,000,182,959 bytes; repository metadata and indexes explain the larger 20.20 GB
repository size. Four root indexes and all 12 member lists are present. Their
contents and the tar archives were not restored, and local `COMPLETE.json` or
unit receipts on JUPITER were not inspected.

This archive was filtered. Its plan includes a 50 MB per-file limit, excluded
directories and filename rules, secret/model-weight exclusions, and the
`restricted-local` name exclusion. Other repositories separately hold larger
scientific artifacts. Matching this plan does not prove every arbitrary JUPITER
file was selected, or that a later local modification has been uploaded.

GeoAxis verification checked every payload against its uploaded `MANIFEST.tsv`:

| Export | Payload files | Hub LFS SHA-256 matches | Downloaded and SHA-256 checked | Missing or mismatched |
|---|---:|---:|---:|---:|
| Prompt generation | 269 | 81 | 188 | 0 |
| Readiness axis | 950 | 294 | 656 | 0 |

Small-file downloads also matched the Hub Git blob identity. Identical blobs
were downloaded once across both exports. Large payloads were verified using
the Hub's content hashes rather than downloading the datasets. README,
MANIFEST.tsv and `.gitattributes` account for three additional files per repo.

## Activations and Gemma reviews

The activation inventory totals **551,713,914,048 bytes**. All eight expected
Llama-3.3-70B/Qwen2.5-72B probing model/style roots are present. The
`t7_chunks_full` directories contain **6,488 files and 471,192,891,368 bytes**.
This agrees in scale with the historical full-collection handoff and the
repository's declared `scope: all`. No current local inventory was available for
comparison, and no full activation content hash or restoration check ran. The
activation repository is public; the other four audited repositories are private.

The saved Gemma SI-v3 ZIP was downloaded and independently rehashed:
`31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f`,
1,253,914 bytes. No SI-v4 review export is present under `reviews/` in the audited
paper repository revision. That says nothing about an unreported local HoreKa run.

## Pinned evidence

| Repository | Audited revision |
|---|---|
| [General archive](https://huggingface.co/datasets/ValerianFourel/geodml-jupiter-archive-private/tree/c872bbd48711a54628e57bae8a7fc2f9b1ae8d32) | `c872bbd48711a54628e57bae8a7fc2f9b1ae8d32` |
| [Prompt generation](https://huggingface.co/datasets/ValerianFourel/geoaxis-prompts-generation-26k/tree/74e64ff6ef3907de3fc8f242d51ca840652e2926) | `74e64ff6ef3907de3fc8f242d51ca840652e2926` |
| [Readiness axis](https://huggingface.co/datasets/ValerianFourel/geoaxis-readiness-axis/tree/7869da0f1f0c36128395b52aabe601fd29a27289) | `7869da0f1f0c36128395b52aabe601fd29a27289` |
| [Activations](https://huggingface.co/datasets/ValerianFourel/geodml-emnlp-2026-probing-activations/tree/0503c8f2ff7a9e3754c759e695dec2c27438cf84) | `0503c8f2ff7a9e3754c759e695dec2c27438cf84` |
| [Experiment V2](https://huggingface.co/datasets/ValerianFourel/geodml-experiment-v2-paper-private/tree/098d3f467b4a0f3fec6da2789043421428390770) | `098d3f467b4a0f3fec6da2789043421428390770` |

The audit used fixed-revision Hub file listings, `index/plan-summary.json`, unit
manifests, both GeoAxis `MANIFEST.tsv` files, `coordination/hours.json`,
`coordination/qwen-results.json` and their referenced `exchange/bundles/` manifests.
Canonical bundle/identity hashes and object paths follow
`agentic_hours.py`, `agentic_task_ledger.py` and `agentic_hour_sync.py`.
Private raw registries, downloaded files and inventories remain in the local
temporary audit directory; only aggregate findings are committed here.

## What still requires JUPITER evidence

- Current jobs and transfer processes, including processes on other login nodes.
- Dirty or unpushed local code on JUPITER. The Mac checkout and GitHub branch both
  pointed to `6cb478f89563c4aa982d725c6a9ccbb5feadcd1a` at audit start.
- Any unpublished Qwen outcomes, new files or changed files outside the saved export plans.
- A current local activation inventory comparison, and a restoration check if
  complete byte recovery must be demonstrated.

The [verification page](../../jupiter-verify-ready.html) retains the local checks
for these gaps and now displays the independently verified HF findings. Nothing
in this audit authorizes deleting local data or releasing live allocations.
