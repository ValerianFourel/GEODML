# Inference counts and Hugging Face update page

Valerian requested a command to count finished Qwen calls, plus a dedicated HTML for auditing inference, updating the private HF dataset and counting total completion. Interpret the quick count as logical generation cells, not internal API calls, Slurm allocations or judge tasks. The current bout division excludes earlier completed/imported cells; its denominator is not the full population. Do not add local and remote counts.

Published executable revision: 7a5d3bb89f2b3349cfdd2c80748f2fa5bf3a3f11 on codex/gemma-selected-cells, pushed with the existing clean-checkout helper and confirmed by ls-remote. Local active checkout also receives the same code; do not push its unrelated history.

Added --qwen-counts-only to audit_horeka_saved_progress.py. It reads CURRENT division identities and observed ledger layout, deduplicates primary/spill identities and reuses the shared-lock nonblocking ledger counter. Reports total, recorded_finished, known_unfinished, unknown, states and observed ledger_stripes. Busy/corrupt evidence is unknown; unsupported division or ambiguous layout is unavailable. It does not inspect scheduler/logs/Gemma/quota or create output files. Full audit CLI remains compatible.

Added report_inference_hub_progress.py. Read-only report at one fixed HF revision reads coordination/qwen-results.json and coordination/hours.json through existing Exchange/HubStore. It verifies bundle manifest hashes, identity fingerprints and reference inventory presence, joins registry checkpoints and deduplicates outcomes across bundles/replans. Conflicting outcomes are excluded from success/failure counts. Missing/malformed evidence is partial; unsupported full-population denominator remains null. generator_totals covers Qwen/Llama; Nemotron stays separate. Counts are manifest-backed operational evidence, not downloaded row verification or scientific acceptance.

Created analysis/docs/horeka-inference-update.html with four copyable, pinned login-shell blocks and matching shell exports:
- horeka-count-qwen.sh: fetch exact code and print CURRENT division cell counts.
- horeka-audit-inference.sh: existing full local audit, unique output directory.
- horeka-update-hf-inference.sh: existing verified Qwen publisher, observed stripe count, local publisher lock, saved HF auth or hidden terminal token, unique report directory, remote receipt checks, then deduplicated published count report.
- horeka-count-published-inference.sh: read-only remote totals without uploading.

Matching HTML and shell files are in workspace root and /Users/valerianfourel/Downloads. New page opened through `open -a Safari`. Copy controls provide manual-selection fallback. Canonical docs include adjacent shell exports, so download links work in each location. The page links the existing Gemma two-map repair guide rather than reimplementing it. Qwen publication does not upload Gemma diagnostics or data available only on JUPITER; this boundary is explicit. No new allocation, claim mutation, inference launch or model-input truncation.

Verification: 43 focused tests passed in the dedicated release checkout, covering local audit/counts, remote unique counts, existing Qwen publisher and hour exchange. Four exported shell blocks pass bash -n; embedded Python compiles; JavaScript passes node --check; anchors and local download links checked in all four exported locations. No live cluster or HF operation executed by Codex. Counts supplied in earlier user logs remain historical: 698 completed Qwen jobs is not a cell count; Gemma selected run 5175156 saved 18/20 cells, two failed maps and ten blocked sources. Pending job 5175229 was not rechecked or changed.

The exact generated update and read-only wrapper programs also passed an offline integration check using the real four-stripe ledger/counts/publisher/Exchange and MemoryHub transport. It proves primary/spill deduplication, separate completed/failed receipts, a repeat update without new writes, read-only reporting without remote writes, missing Llama evidence remaining unavailable, corrupt receipt rejection and credential redaction. Scratch proof: /tmp/verify-horeka-inference-wrapper.py. No tracked test-only production seam was added.

Next: user runs page step 1 and returns qwen_unique_cells; step 3 publishes locally saved terminal Qwen outcomes and produces hub-progress.json with the fixed remote revision. A live whole-population completion percentage is not established by these indexes. Preserve all saved reports and existing jobs.
