# Nemotron sample selection speed, 2026-09-29

User reported slow login setup and requested a fix/new command. The operator
page unnecessarily called completed_generator_refs on both full datasets,
verifying all completed generation/trace references before choosing ten each.
Updated only local testnemotron20.html and /private/tmp/build-nemotron20.py.
No inference pin or production sampling code changed.

Replacement reads sealed task definitions and ledger metadata, orders candidates
by the same seeded hash within each method, and verifies candidates lazily until
ten valid cells per model are selected. Invalid candidates are replaced within
the same method. Existing exclusions and frozen snapshot reuse remain intact.
Progress prints before metadata/ledger reads and every candidate check; Python
runs unbuffered. Metadata scans remain necessary and can still take time on
shared storage. The manifest reports completed candidates and checked candidates,
not a falsely claimed corpus-wide verified count.

Real temporary sealed-dataset/ledger fixture: old and new scripts produce
byte-identical 10+10 selections, preserve method balance/exclusions, replace a
missing highest-priority generation shard, and reuse frozen selections with no
further reference checks. Whole-corpus baseline required 94 checks; updated
script required 42. Local proof is /private/tmp/check-nemotron20-sampling.py.
Saved HTML shell blocks pass bash -n, embedded Python ast.parse, copy JavaScript
node --check. Also exported root testnemotron20-setup.sh for copying.

User asked why not choose randomly: explained that selection is seeded random
within methods; corpus verification was the unnecessary cost. Approval stays
one existing-allocation 20-cell test with 40 minutes left initially. No cluster
execution, new allocation or extension. User may Ctrl-C only the old login setup
before copying corrected step 1; preserve the separate GPU shell. No new live
cluster facts or 20-cell results received. Previous SI-v3 five-cell result remains
structural success only. Next: user runs corrected setup then approved node step
if remaining time permits, and returns full judgments for semantic review.
