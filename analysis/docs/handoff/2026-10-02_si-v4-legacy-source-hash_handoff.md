# SI-v4 legacy source-hash compatibility

Date: 2026-10-02

Valerian reported an existing HoreKa allocation running pin 9aa5d90e0b310f3eea587a60787727787ab92757. The review failed in Coordinator._import with "source hash does not match its dependency", V4_EXIT=1. No judging pass completed in this invocation. Live allocation ID, remaining time, and actual input mismatch details remain unverified.

History inspection showed SI-v3 source cell records at 961ad0b lacked source_sha256. The v4 bridge copied them without deriving the new field. Extended the existing bridge contract test to legacy records; before fixing, it reproduced the exact coordinator failure. A conflicting-hash test also failed before the fix. freeze_saved now derives absent source hashes from validated frozen task text and rejects existing conflicting hashes. It preserves original bundles, scientific text, task token budgets and the coordinator guard. No existing cluster preparation was edited.

Verification: 32 focused tests passed across test_horeka_si_v4.py and test_horeka_gemma_si.py. Baseline suite was 19 passing before adding the regressions. Test-audit authoring gate applied: existing coverage exercised only modern cells; regression owns the bridge-to-coordinator import contract and needs no production seam. No external OpenClaw/Vitest tools referenced by the generic skill apply to this Python repository.

User asked for the diagnostic in the HTML. Updated analysis/docs/horeka-si-v4.html with a prominent read-only source-hash diagnostic, scheduler status commands, result interpretation and recovery status. Copy button reuses the existing handler. Updated workspace-root copy and added Downloads/horeka-si-v4.html. Verified copied command extraction, bash -n, Python AST parsing and byte-identical exported pages. Original launch commands remain historical; header tells user not to rerun them for this failed preparation.

Next: Valerian runs the HTML diagnostic and returns counts and live allocation status. Confirm all mismatches are absent fields before rebuilding into a new directory from hash-verified original bundles at the corrected commit. Preserve failed preparation and attempts. Do not silently patch frozen manifests or bypass validation. Determine remaining allocation time before any continuation. New or extended allocation requires fresh estimate and explicit approval. No remote operations, inference, allocation, cancellation, push or model download occurred.
