# Search answer schema compatibility fix

Cluster job 1711270 returned eight valid rankings, one invalid ablated ranking,
and nine HTTP 400 answer failures on commit 627ad348. No answers were generated.
The answer schema's nested `uniqueItems` keyword is explicitly rejected by
[vLLM 0.28's xgrammar backend](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/structured_output/backend_xgrammar/).

Remove that keyword from generation only. The existing `_ids` validator still
rejects duplicate and unknown citations. Prompts, evidence, seeds, token limits,
model revisions and ranking validation are unchanged. The invalid ablated C001
ranking remains invalid; no content repair or retry-policy change is included.

The answer schema change changes request hashes and answer task IDs. Do not
resume the old mixed primary directory with this code or modify its manifest.
Preserve its eight valid rankings and all attempt records. Use a separate output
directory for a corrected pilot run; this fix does not implement cross-version
import of saved rankings. Do not add counts across these runs as unique tasks.
The frozen bundle remains usable because the accepted semantic answer contract
and evidence did not change. Generated output is not claimed bitwise equivalent.

Before loading any model, run in the installed cluster environment:

```bash
python3 analysis/scripts/check_search_experience_grammar.py
```

This invokes vLLM's actual request-schema validation path, including grammar
conversion, for ranking, answer and judge schemas. It does not load a tokenizer
or model weights. Failures stop the command; do not proceed to model loading.

Local regression `test_citation_uniqueness_is_validated_outside_generation_schema`
failed before the fix because `uniqueItems` was present. Afterward the focused
suite passed, including duplicate and unknown citation rejection and existing
restart identity checks. The installed-backend test is skipped without vLLM;
cluster grammar acceptance and corrected GPU inference remain to be verified.
