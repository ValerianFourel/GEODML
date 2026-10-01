# Explicit three-hour queue exception

Valerian answered "Allow one new three-hour run alongside Qwen." Added a narrowly scoped exception for reviews/gemma-si-v4-development-20261002-3h with 03:00:00, pinned Gemma model/revision and SI-v4-development trial. Existing one-hour exception remains restricted to its original dated path/pin/duration. Only the five-active and no-pending guards are waived when the explicit check flag is used. Fresh scheduler/quota evidence, ten-minute observed-start spacing, account capacity, duplicate Gemma and attempted-run guards remain enforced.

Extended the existing admission contract table for both dated scopes and mismatched durations, preserving recent-start, full-queue, storage, duplicate and attempted-run negative cases. Focused helper/sibling tests pass. No cluster operation or allocation is performed by this change. Update the HTML's helper pin and both admission invocations to use the approved flag, replace pending-approval wording, publish the code, and verify exports.

Final page updated to pin 1bf21cb0e37374634be30a07bca9079ec18d6f87 and both check commands use the scoped exception. All 43 focused tests passed. All ten decoded HTML command blocks and embedded Python parse; anchors resolve; exactly one 03:00:00 salloc block remains; exports match. The code pin is published to the existing branch. No allocation submitted by Codex.
