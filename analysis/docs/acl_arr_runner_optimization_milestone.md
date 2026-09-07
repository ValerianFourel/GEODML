# Shared runner and formatting milestone

The shared primary/judge runner now uses a bounded rolling request window.
Completed requests are persisted without waiting for the slowest member of a
four-concurrency-sized chunk. This applies to any frozen model configuration;
it does not verify that a particular model can load on the cluster.

Restart checks validate task identity, request fingerprints, raw output hashes
and parsed outputs. Duplicate or corrupt completions fail closed. Journals are
streamed rather than loaded in full. A directory lock prevents simultaneous
writers. HTTP attempts and invalid raw outputs are retained. Partial execution
returns checkpoint status instead of claiming the complete shard succeeded.
Legacy output directories lacking the new resume identity are rejected without
modification; do not point the new runner at an old pilot output directory.

Use `--scheduler chunked` to benchmark the old admission pattern and
`--scheduler rolling` for the new default. Existing decoding, retry limits,
task inputs, model revisions and validators are unchanged. Use `--pilot-only`
for generic-runner benchmarks. The specialized Llama pilot wrappers retain their
own execution loop and do not automatically acquire the generic scheduler.

## Local measurement

Run from the checkout with a fresh output path:

```bash
python3 analysis/scripts/benchmark_acl_arr_scheduler.py --scheduler both --output /tmp/acl-arr-scheduler-comparison.json
```

Five repetitions, 128 synthetic tasks, concurrency eight, controlled 3 ms
requests with a 120 ms straggler every 32 requests produced median rates of
263.44 validated mock tasks/s for chunked admission and 702.01 for rolling
admission. Median first-result latency was 121.32 ms versus 3.97 ms. Request
payload hashes matched. These measurements exclude GPU inference and durable
storage. They are not GPU speedup or full-study runtime estimates.

## Formatting boundary

`repair_acl_arr_pilot_citation_order.py` accepts an additional explicit
`--approve-pilot-only-formatting` flag. Together with the grouped-citation flag,
this permits whole-response JSON fences and surrounding citation whitespace.
Answer text remains unchanged. Citation membership must match exactly and IDs
must be supplied; missing IDs, unknown IDs, prose wrappers, malformed JSON and
duplicate declared IDs remain rejected. Reports are cumulative pilot-only
sidecars, never production outcomes or generic-runner resume records.

The latest supplied cluster report has 252 originally valid answers, 52
order-only repairs, seven grouped-citation repairs, and 73 unresolved answers.
No evidence establishes how many additional formatting-only cases remain.
Do not infer answer quality from structural validation.

## Remaining execution gates

No new inference or allocation was launched. Four-model GPU performance,
backend compatibility, memory capacity and full-production freezes remain
unverified locally. Keep one model active per four-GPU node until measurements
justify another configuration. Do not claim optimal concurrency for any arm
from this CPU benchmark. Use each frozen model's actual tokenizer and output
allowance for context preflight, not a generic 32768-token limit.
