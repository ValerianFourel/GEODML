# Independent answer-judge readiness

This document defines the judge boundary for the captured-evidence pilot. The
proposed agentic methods use the same principle of independent judgment, but
their retrieval, compaction, workload, and production gates are specified in
[`agentic_search_retrieval_protocol.md`](agentic_search_retrieval_protocol.md).
An agentic compatibility smoke does not establish judge readiness or answer
quality.

Judge compilation requires the exact expected answer task-ID set, not complete
ranking coverage. All saved primary records still pass existing provenance,
request-identity, duplicate, hash and output validation. Missing or invalid
answers block judging. Missing or failed rankings do not. Reports continue to
show incomplete ranking cells; judgment does not promote a failed ranking.

The change preserves judge prompts, schema, seeds, blinding and model checks.
It can read the existing schema-fixed Llama primary directory without rerunning
any of its 17 valid tasks. Keep the primary directory immutable once judgment
starts because judge provenance includes its journal and manifest hashes.

## Streaming boundary

Concurrent reading of an active primary writer is not supported. Compilation
rejects a primary manifest with status `running`. This status guard is not a
cross-process lock: operators must not restart primary inference while a judge
uses its artifacts. Existing hash checks reject subsequent source changes.

No second GPU model is launched by this change. The current generator uses TP=4
and GPU-memory-utilization=0.90. That setting does not establish free capacity
for an independent judge. Idle GPU memory readings do not measure co-residency.
Use sequential serving within the existing allocation until weights, KV cache,
activation peaks and throughput have been measured for both models together.

A future streaming mode needs immutable per-answer handoff records, bounded
backpressure, independently resumable judgment, and measured GPU headroom. Do
not weaken source hashes or share actively changing journals to simulate it.

## Verification

The regression first failed at the old complete-primary gate. It now compiles
nine identical judge requests with one ranking failure, preserves source bytes,
keeps incomplete report cells, rejects incomplete answers, and rejects a running
primary writer. Synthetic fixtures establish behavior, not scientific quality.
