# Nemotron v4 results and private evidence export

Valerian returned the job5177346 twenty-cell summary and requested the full
inference here and on the existing private Hugging Face dataset for another
semantic assessment. The pasted summary reports zero complete cells, 93 sources
blocked by map_unusable, seven by map failure, and seven uncertain. There was
one inference failure; no deadline admission stop. Warm SI cost is
0.051949526887490514 node-hours = 3.11697 minutes and 0.20779810755 GPU-hours.
This is not throughput for valid completed cells. No returned source has a
reported numeric scored count. The actual map notes/raw responses are needed
to judge the cause; do not infer them from counters.

Prepared an inline read-only export command for a separate HoreKa login shell,
preserving the live allocation. It requires the terminal attempt receipt and
an idle coordinator lock, dumps all task input/result records from read-only
SQLite including raw responses and retries, frozen cells, run/judge config,
reports, attempt execution/receipt and current Slurm accounting to a unique
JSONL file. Upload only that file into a unique path in the existing private
experiment dataset, requiring private repository metadata, with a hidden
terminal credential prompt. Never reproduce the exposed credential. A replacement
HF token must be provided only in the terminal, not chat.

No cluster access, upload, model rerun or complete semantic analysis was executed
by Codex this turn. Next: receive the pinned upload URL/hash, download that file
with local cached authentication, verify its hash and review all twenty cells
against their exact matched Gemma records. Separate diagnostic review from formal
semantic acceptance and distinguish warm time from allocation occupancy.
