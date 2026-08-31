# Readiness search contracts and high-axis recovery

This document distinguishes the versioned text contracts from the targeted
generation profile used to recover the action-ready end of readiness axis 1.
Prompt embeddings diagnose generated text; they do not define randomized policy
variable `B`.

## `question-v1`: historical Gold pipeline

Generation produces one standalone question of 8–60 words. The exact assigned
keyword phrase must occur verbatim and the text must end in exactly one question
mark. Independent review requires exact-keyword presence, a single question,
topic relevance, search intent, web answerability, standalone interpretation,
natural language, and relevance of at least 4/5.

Strict selection uses the frozen Qwen view and development-aligned Mistral view,
requires both views within `0.017` of the target, assigns at most one candidate to
each keyword–target cell, and enforces global delexicalized-template uniqueness.

## `search-trigger-v2`: relaxed search-form pipeline

Generation produces one 4–60 word online-search trigger. It may be a question,
imperative request, or concise search phrase. The assigned topic is supplied as
metadata, so the exact keyword and standalone interpretation are optional.

Independent review still requires topic relevance, genuine search intent, web
answerability, natural language, and relevance of at least 4/5. It measures but
does not gate on exact-keyword presence, question form, or standalone wording.
Strict dual-view selection, one-to-one assignment, and template uniqueness remain
enabled; the registered v2 distance tolerance is `0.035`.

Relaxing stored v1 candidates recovered no validation rows because those texts
had already passed generator-side v1 form checks. New v2 text must therefore be
generated to test the relaxed form contract.

## `high-axis-action-v1`: targeted v2 generation profile

This is not a third acceptance contract. It is a generation profile layered on
`search-trigger-v2`, and it leaves all v2 validation and selection gates intact.
It processes unresolved targets at or above axis 1 value `0.700`, highest first.

The semantic calibration is:

- `0.70–0.80`: an imminent choice expressed as commitment, preparation, or an
  ordered plan;
- `0.80–0.90`: an already chosen approach expressed as setup, configuration,
  application, or implementation steps;
- `0.90–1.00`: immediate execution or a current blocker expressed as the next
  action, corrective procedure, troubleshooting sequence, and verification.

The search/action boundary is explicit: the trigger requests web-findable
instructions needed for imminent action. It does not claim that the search system
has already acted. The profile must not add cost, safety, brand, quality, or any
other ranking criterion unless it is already part of the assigned topic.

The non-allocating JUPITER harness is
`analysis/scripts/slurm/jupiter/run_readiness_30k_search_trigger_v2_high_axis.sh`.
It must be invoked inside a separately approved allocation. Each generated round
also runs `audit_readiness_high_axis_generation_yield.py`, reporting task,
candidate, acceptance, and recovered-cell yield for the three high-axis bands.
