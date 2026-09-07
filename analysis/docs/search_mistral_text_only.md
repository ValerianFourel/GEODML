# Mistral text-only startup fix

The reported startup failed in `PixtralProcessor` on dummy text `[IMG]`, before
the primary runner was invoked. The text-only launch had left multimodal
processing enabled. This is distinct from output-schema validation or context
overflow. The underlying processor exception is not fully present in the log
excerpt; this change disables the unused image path rather than patching it.

`slurm/jupiter/run_search_mistral_primary.sh` is now a committed worker. It uses
`--language-model-only`, the native Mistral tokenizer, TP=4, BF16 and
`FLASH_ATTN_MLA`. The pinned revision, text prompts, output contracts and seeds
are unchanged. It checks CLI support before loading weights and fails closed
if that option is unavailable. The setting is recorded in its sibling settings
log. Existing result directories are validated; partial runs stop for review.

Official vLLM documentation describes language-model-only as disabling all
modalities by setting their limits to zero:
https://docs.vllm.ai/en/stable/api/vllm/config/multimodal/

The user verified 18 tasks require 41,192 tokens with the native tokenizer;
the worker retains a 41,472 context. Set `SEARCH_PILOT_ROOT`,
`GEODML_EXECUTION_COMMIT`, `GEODML_EXPECTED_JOB_ID`, and the existing environment
variables before calling it through an existing Slurm GPU step. It does not
allocate or extend resources. Its initial pilot estimate is 10–30 minutes;
the caller must supply the approved step budget.

Local shell syntax and runner tests do not verify GPU initialization. The next
cluster run must establish that text-only startup bypasses the processor error
and that outputs pass validation. No successful Mistral inference is claimed.
