# JUPITER remote HF verification, 1 October 2026

Valerian asked Codex to verify Hugging Face independently after preparation of
the JUPITER verification HTML. This turn used the Mac's saved authentication for
read-only Hub calls. No JUPITER SSH, Slurm action, upload, cleanup or registry
mutation was performed. The earlier one-hour HoreKa SI-v4 allocation has no new
returned job ID or result in this conversation; preserve any live shells.

## Verified snapshot

All five heads remained unchanged at the end, 2026-10-01 05:26:47 UTC:

- General archive: `c872bbd48711a54628e57bae8a7fc2f9b1ae8d32`.
- GeoAxis prompt generation: `74e64ff6ef3907de3fc8f242d51ca840652e2926`.
- GeoAxis readiness axis: `7869da0f1f0c36128395b52aabe601fd29a27289`.
- Activations: `0503c8f2ff7a9e3754c759e695dec2c27438cf84`.
- Experiment V2 paper: `098d3f467b4a0f3fec6da2789043421428390770`.

Llama: 310,380 planned, 310,374 completed, six terminal failures, zero remaining
or owned. The six failures record `LLM transport failed`, two attempts each.
1,006 packages are complete and two blocked packages contain the accepted
failures. Registry `failed` corresponds to bundle state `terminal_failed`.

JUPITER Qwen: 24,218 distinct successes, zero failures, 26 bundles. Llama's 60
checkpoint bundles and Qwen's 26 bundles pass canonical hashes, identity hashes
and outcome joins. All 21,336 referenced objects exist at matching sizes; 2,062
LFS SHA-256 values also match. The remaining 19,274 plain Git objects total
5.19 GB and were not downloaded to hash. No complete row-level payload replay.

Archive: all 12 units match the saved remote plan; all parts match LFS SHA-256
and size. 2,863,060 original file/link entries, 97.997 GB before compression,
20.000 GB of compressed parts, 20.201 GB including metadata/indexes. All member
lists and root indexes are present but their entries and tar contents were not
restored. Local COMPLETE/unit receipts were not inspected. Scope remains the
filtered plan, not every file on JUPITER.

GeoAxis: all 269 prompt payloads and all 950 axis payloads match manifest sizes
and hashes. 81/294 hashes use LFS metadata; 188/656 small payloads were downloaded
and rehashed, deduplicating shared Git blobs. Zero missing or mismatched files.

Activations: 8,022 files and 551,713,914,048 bytes visible remotely, including
6,488 full T7 files and 471,192,891,368 bytes. Eight expected probing roots are
present. No live local inventory comparison or full content verification. This
repo is public; the other four are private.

Gemma SI-v3 export: downloaded and SHA-256 verified against
`31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f`,
1,253,914 bytes. No SI-v4 review export under `reviews/` at the audited revision.

## Deliverables and verification

Saved aggregate report and JSON under
`analysis/docs/reviews/jupiter-hf-20261001/`. Updated
`analysis/docs/jupiter-verify-ready.html` and identical workspace-root copy with
the verified snapshot and remaining limitations. The five paste blocks and
JavaScript are unchanged; syntax checks and copy comparison passed. No runtime
code changed and no additional model/tests were needed beyond document checks.

Local audit metadata, inventories and downloads remain in
`/private/tmp/jupiter-hf-audit/`; scratch audit script is
`/private/tmp/check-jupiter-hf.py`. Raw registries/corpus data and credentials
were not committed. The report records revision-pinned evidence and inventory
hashes. Initial audit code state `6cb478f89563c4aa982d725c6a9ccbb5feadcd1a`
matched GitHub `codex/pilot-continuation`; this handoff's commit adds the findings.

Remaining: current JUPITER jobs, transfers, dirty/unpushed local checkouts,
unpublished results and files outside saved plans require local cluster evidence.
The HTML retains those read-only checks. Do not infer deletion readiness or full
activation recovery from remote inventory alone.
