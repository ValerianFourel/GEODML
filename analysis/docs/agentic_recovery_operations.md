# Recovery audit and interrupted-run continuation

The initial one-hour allocation reached 294,000 captured files within roughly
55 minutes and expired before a verified export was reported. Valerian approved
a fresh four-hour interactive allocation: one full JUPITER booster node, four
GPUs, 32 requested CPUs, all node memory; maximum 16 GPU-hours. The provisional
remaining estimate is 2–4 hours, including verification and export, with unknown
remaining file count. Four hours provides one hour over a three-hour working
estimate; it is not a completion guarantee. No inference runs during this audit.
The launcher reserves five minutes before the actual allocation end and never
resubmits or extends the allocation.

Use `salloc` for the allocation-owning shell, then `srun --pty` inside that
allocation to enter the compute node. Keep both shells open. Supply the runner's
sixth argument as `04:00:00` and seventh argument as the previous
`dataset/local-forensics` directory. The output control directory must be new.

Resume copies the SQLite index and links closed packs into the new snapshot.
It conservatively recaptures the last pack of an interrupted collection. Sources
are rechecked by content hash; changed files are recaptured. Missing sources
remain historical artifacts and lose their validated outcome status. The old
snapshot remains untouched. All packs are verified before publication. Logs
identify collection, archive verification, export and checksum stages.

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
consume the GPU allocation. An incomplete build has no publication
manifest and cannot be uploaded. Partial archives remain for recovery; use `build --resume-from` with a new
output directory and a separately approved allocation.

A successful local test is not a claim that the JUPITER job or HF upload ran.
