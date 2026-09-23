# One-hour recovery audit

The approved allocation is one full JUPITER booster node, four GPUs, 32 CPUs
requested for the process, all node memory, and 01:00:00 wall-time. Maximum
allocated capacity is four GPU-hours. This job runs no inference. Estimate:
20–60 minutes or longer under filesystem stalls, based on the previously
reported 24 GiB / 437,078 pilot files and 89 GiB of project runs. This is not a
measured collector benchmark. The launcher reserves five minutes before the
actual allocation end and never resubmits or extends the allocation.

`run_agentic_recovery.sh` captures all project run directories, exports,
manifests, the frozen population's final-audit directory, and its own static
provenance. It also runs the existing exact original-500 pilot reporter with a
five-minute limit. The output goes to a fresh fscratch directory. Originals are
read-only. Failed allocations and unfinished manifests are not exclusion rules.

`collect_agentic_forensic_snapshot.py` preserves source bytes in content-addressed
tar packs, checks the copies, and creates an SQLite index. Symlinks, locks,
restricted-local trees, caches and model binaries are excluded and inventoried.
Referenced paths outside the collection roots are listed explicitly. This is
not proof that every GEODML artifact on every filesystem has been found.

`build_agentic_recovery_dataset.py` produces `dataset/hub/data/*.jsonl.gz`:

- prompts and original axis rows;
- cohort memberships and run metadata;
- generations with nested answers, ordered rankings and complete saved traces;
- generation aliases retaining native request and protocol identities;
- trace events, separately from physical HTTP call totals;
- validated native judge outcomes bound to a recovered answer and trace;
- population-bound judge journal records, including valid prefixes before a
  damaged tail, with their validation status;
- factorial coverage for each eligible prompt and both generators;
- allocation accounting, source artifact index and external references.

JSONL preserves nested arrays and dictionaries; each table is independently
compressed. Empty tables are omitted and have zero rows. The summary names all
nonempty tables. This implemented recovery format is
`geodml-recovery-dataset-v1`, separate from the proposed final paper Parquet
contract. It must not be presented as an implemented final-paper contract.

Recovery percentages count slots with one validated saved generator payload,
and linked native Nemotron judgments. Distinct protocol variants and conflicting
payloads are retained. Legacy judge journals are exported as candidates and do
not inflate native judgment totals. Full-population scientific completion stays
null until protocol acceptance and every unresolved artifact/mapping have been
reconciled. Original-500 exact audit results, when successful, are included
separately. Total physical model requests remain unknown if they were not logged.

The Hub directory contains generated-population-bound records and metadata.
Raw packs, unrestricted logs, source corpora and unclassified artifacts stay
in `dataset/local-forensics`; the artifact table points to them. Missing transfer
provenance is excluded from Hub export. Model weights and environments are not
copied to the Hub. This follows the repository restriction on WildChat-derived
and restricted-local material. Such exclusions do not reduce research targets.

The publisher checks every publication checksum, rejects unexpected files,
verifies that the destination is private, and uploads to an immutable snapshot
path. An existing snapshot with a different manifest is rejected. After upload,
it checks the remote file inventory and saves the commit receipt locally.
Upload runs on the login node after the allocation, so network time does not
consume the one-hour GPU allocation. An incomplete build has no publication
manifest and cannot be uploaded. Partial archives remain for recovery; restarting
collection requires a new output directory and a separately approved allocation.

A successful local test is not a claim that the JUPITER job or HF upload ran.
