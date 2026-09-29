# Nemotron test page, 2026-09-29

Created `/Users/valerianfourel/Hamburg/GEODML_Unified/testnemotron.html` and
opened it in Safari at Valerian's request. The page is a local operator artifact
outside the Git checkout. It targets tested implementation commit
`1722926cf1eda4b2f568491e77c7307e2112cc2b` and provides copy buttons for:

1. Publishing that exact commit from the Mac, without force.
2. Creating its separate HoreKa checkout and running the CPU-only native schema check.
3. Reading current allocations and the historical job 5169925 status.
4. Replaying the original five Qwen cells in an existing allocation.
5. Reading the newest SI-v2 replay's summaries, retry histories and example.

Valerian answered the fresh-allocation question with "i already have a new 1 hour".
The page therefore uses that existing new allocation; it does not request another.
Its Slurm ID and live state remain unknown until the user runs the checks.
Expected replay use is 10–25 minutes including startup, on one exclusive node,
four A100s, 32 requested CPUs, all node memory; one-hour ceiling is 4 GPU-hours.
The replay block refuses a mismatched allocation name/time, fewer than 25 minutes
remaining, a dirty/wrong checkout, or background/stopped jobs in its shell. It
uses `submit --no-submit` to freeze a new run and then executes its run.sh within
the current allocation. The launcher prints a suggested salloc command; the page
explicitly tells the user to ignore that suggestion and use the existing job.
Failure exits only the child shell. No cancellation, extension or resubmission.

Verified the saved HTML's five shell commands with bash syntax checks, both
embedded Python programs with ast.parse, and copy-button JavaScript with node
--check. Safari open returned success. Commands were not run on either cluster;
no commit was pushed here. No native xgrammar or live SI-v2 results received yet.
Next: inspect returned schema/replay outputs before expanding to fresh Qwen/Llama
samples or estimating production bouts. Earlier implementation handoff remains
[here](2026-09-29_nemotron-si-v2-fix_handoff.md).
