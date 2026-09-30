# Fresh Gemma candidate verification fix

Pasted evidence from job 5171481 on hkn0402: running step 3 before successful
preparation produced FileNotFoundError writing quota-before-fresh20.json because
the run directory did not exist. Corrected step 2 then scanned Qwen metadata but
failed verifying generation-bc19eb692fdc97d2bbdd478251c4d595b7e08cfe1e01fc482e430cb0ffd8b857-g1.
No fresh sample was saved and no fresh inference began in this evidence.
We cannot distinguish missing/unsealed shards, corrupt hashes or a mismatched
reference from this traceback alone. No live remote access was performed.

Root implementation defect: sample_refs treated latest completed ledger metadata
as sufficient selection eligibility. iter_cells subsequently and correctly
rejected an unverified record. Code 135d4685f232776d2e8157e93e41657f3b062ee9
checks both candidate references before adding a cell to the frozen sample.
Uses the existing verifier and deterministic method-aware sampler, examining
only enough candidate batches to reach ten verified fresh prompts per corpus.
Unverified candidates are explicitly logged and saved in rejected_candidates;
selection metadata names the verified-reference eligibility policy. If fewer
than ten verified candidates exist or the remaining-time limit is reached,
preparation fails. Later integrity changes and unassessable selected inputs still
fail; no scientific rubric, task schema, caps, seeds or model settings changed.
This changes the eligible sample before freezing, transparently, not frozen data.

Regression reproduced the old sampler selecting a ledger-completed record with
its local shard missing. New sampler excludes it, records the failed reference
and yields ten verified distinct fresh prompts. Existing all-corrupt-evidence
case still fails without writing a bundle. Focused four-file suite: 59 passed.

HTML updated to new code pin, still job 5171481. Step 3 now checks config.json
and run.sh before writing quota evidence and instructs rerunning step 2 on failure.
Do not mkdir the run directory: preparation owns it and refuses partial outputs.
Eight shell blocks/embedded Python parse; copies match; diff check passed.
Next: repeat page steps 1 (separate login shell), 2 and 3 (existing compute shell).
No new allocation, release or queued-job change. Existing twenty-minute remaining
window stays enforced. Actual fresh Gemma outputs are still pending.
