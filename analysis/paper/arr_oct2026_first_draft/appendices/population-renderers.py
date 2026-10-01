# Verbatim renderer excerpts; not a standalone executable.
def render_generation_request(
    task: ReadinessGenerationTask,
    *,
    candidate_slot: int,
    text_contract: str = "question-v1",
    generation_profile: str = "balanced-v1",
) -> str:
    if text_contract not in TEXT_GENERATION_CONTRACTS:
        raise ValueError(f"unsupported text generation contract: {text_contract}")
    _validate_generation_profile(text_contract, generation_profile)
    support_aware = task.target.target_id.startswith("readiness-support-target:")
    axis_1_only = task.target.target_id.startswith("readiness-axis-1-target:")
    if support_aware or axis_1_only:
        a1 = _continuous_axis_instruction(
            task.target.normalized_axis_1,
            (
                "purely understand or explain the topic",
                "investigate evidence, mechanisms, or implications",
                "evaluate concrete options or trade-offs",
                "prepare a decision, commitment, or practical plan",
                "request an immediate, concrete action or execution step",
            ),
        )
        if axis_1_only:
            a2 = "unconstrained; choose whatever decision mode makes the question natural"
            continuous_control = (
                "Control only readiness stage. Treat its percentage as a graded "
                "semantic mixture, not a category, and preserve differences between "
                "nearby targets through the question's actual information need."
            )
        else:
            a2 = _continuous_axis_instruction(
                task.target.normalized_axis_2,
                (
                    "compare alternatives and decide which approach fits",
                    "select an approach using explicit criteria",
                    "translate a chosen approach into a practical procedure",
                    "implement, configure, troubleshoot, or execute a chosen approach",
                ),
            )
            continuous_control = (
                "Treat each percentage as a graded semantic mixture, not a category. "
                "Preserve the difference between nearby targets through the question's "
                "actual information need."
            )
        surface_control = _surface_realization_instruction(
            task.generation_seed + candidate_slot * 1009
        )
    else:
        a1 = _axis_1_instruction(task.target.normalized_axis_1)
        a2 = _axis_2_instruction(task.target.normalized_axis_2)
        continuous_control = "Use the requested semantic destination."
        surface_control = "Make the wording natural and distinct."
    if text_contract == "search-trigger-v2":
        a2 = a2.replace("question", "search trigger")
        continuous_control = continuous_control.replace(
            "question's", "search trigger's"
        )
        surface_control = _search_trigger_surface_instruction(
            task.generation_seed + candidate_slot * 1009
        )
        high_axis_control = ""
        if generation_profile == "high-axis-action-v1":
            high_axis_control = _high_axis_action_control(
                task.target.normalized_axis_1,
                candidate_slot=candidate_slot,
            )
        return f"""Write one natural-language online-search trigger associated with the assigned topic below.

Assigned search topic (metadata; it need not appear verbatim): {task.keyword}

Semantic destination:
- Readiness stage: {a1}
- Decision mode: {a2}
- Control rule: {continuous_control}
- Surface realization: {surface_control}
{high_axis_control}

Iteration feedback: {task.feedback}
Candidate slot: {candidate_slot}

Contract precedence:
- Any iteration feedback asking for the exact keyword or a question frame comes
  from question-v1 and is superseded by search-trigger-v2.

Hard constraints:
- Express a genuine information need that warrants an online search about the
  assigned topic when the topic metadata is supplied.
- At the high-readiness end, request action-enabling instructions needed to act
  now; do not pretend that the search system has already performed the action.
- A question, imperative request, or concise search phrase is allowed.
- Use 4 to 60 words and one normalized line.
- The exact assigned topic phrase is optional, and hidden topic context is allowed.
- Do not answer the information need.
- Do not mention axes, coordinates, readiness scores, embeddings, reranking,
  publishers, or source-preference policies.
- Make this candidate materially different from obvious generic phrasings.

Return only JSON: {{"search_trigger":"..."}}"""
    return f"""Write one standalone, natural search question about the exact keyword phrase below.

Exact keyword phrase (must appear verbatim): {task.keyword}

Semantic destination:
- Readiness stage: {a1}
- Decision mode: {a2}
- Control rule: {continuous_control}
- Surface realization: {surface_control}

Iteration feedback: {task.feedback}
Candidate slot: {candidate_slot}

Hard constraints:
- Ask one question that a search-capable LLM could research and answer.
- End with exactly one question mark.
- Use 8 to 60 words and one line.
- Include the exact keyword phrase verbatim.
- Do not answer the question.
- Do not mention axes, coordinates, readiness scores, embeddings, reranking,
  publishers, or source-preference policies.
- Make this candidate materially different from obvious generic phrasings.

Return only JSON: {{"question":"..."}}"""


def _continuous_axis_instruction(value: float, anchors: Sequence[str]) -> str:
    if not 0.0 <= value <= 1.0 or len(anchors) < 2:
        raise ValueError("continuous semantic control requires [0, 1] and two anchors")
    scaled = value * (len(anchors) - 1)
    lower = min(int(np.floor(scaled)), len(anchors) - 2)
    upper = lower + 1
    upper_weight = scaled - lower
    lower_weight = 1.0 - upper_weight
    return (
        f"{value:.3f} on a 0-to-1 continuum: "
        f"{lower_weight:.0%} '{anchors[lower]}' and "
        f"{upper_weight:.0%} '{anchors[upper]}'"
    )


def _surface_realization_instruction(seed: int) -> str:
    variants = (
        "use a concise direct interrogative with ordinary wording",
        "place a short context clause before the main interrogative",
        "state the main information need first and its qualifier near the end",
        "use neutral professional wording without unnecessary jargon",
        "use a natural first-person search question without inventing personal facts",
        "use an impersonal search question with a concrete but generic scope",
        "open with a conditional situation and ask what follows from it",
        "frame the question around evidence needed to resolve uncertainty",
        "ask through a contrast between two plausible approaches without naming brands",
        "use a concise how-or-why construction and avoid stock 'what are the best' wording",
        "frame the information need as a diagnostic question about causes or consequences",
        "ask for criteria that would distinguish a suitable approach from an unsuitable one",
        "use a natural scenario-first question with the core topic near the end",
        "ask what a careful reader should verify before proceeding",
        "use an outcome-first question that asks what would help achieve that outcome",
        "frame the question around a concrete obstacle without inventing personal details",
        "ask for a sequence or procedure only when the semantic destination calls for action",
        "use an uncommon but natural interrogative structure rather than a reusable template",
    )
    return variants[seed % len(variants)]


def _search_trigger_surface_instruction(seed: int) -> str:
    variants = (
        "use a concise direct request without forcing interrogative punctuation",
        "use a natural imperative that requests research, setup, or troubleshooting",
        "use a compact search phrase with a concrete intended outcome",
        "state the desired outcome first and the information need second",
        "use a natural first-person request without inventing personal facts",
        "use an impersonal search instruction with a concrete but generic scope",
        "open with a practical obstacle and request the next useful information",
        "request evidence needed to resolve the current uncertainty",
        "contrast two plausible approaches without naming unsupported brands",
        "request a sequence only when the semantic destination calls for action",
        "use a diagnostic search trigger about causes or consequences",
        "request criteria that distinguish a suitable from unsuitable approach",
    )
    return variants[seed % len(variants)]


def _high_axis_action_control(value: float, *, candidate_slot: int) -> str:
    """Make high readiness action-enabling without changing topic semantics."""

    if not 0.70 <= value <= 1.0:
        raise ValueError("high-axis-action-v1 requires an axis-1 target >= 0.70")
    if value < 0.80:
        stage = (
            "The choice is imminent. Request a concrete commitment, preparation "
            "plan, prerequisite check, or ordered checklist, but do not ask for "
            "background explanation or another comparison of options."
        )
    elif value < 0.90:
        stage = (
            "Treat the approach as already chosen. Request exact ordered steps to "
            "set up, configure, apply, or implement it, including the first action "
            "and a practical completion check."
        )
    else:
        stage = (
            "Treat execution as immediate or already blocked. Request the precise "
            "next action, command-like procedure, corrective step, or troubleshooting "
            "sequence and how to verify that it worked."
        )
    realizations = (
        "Use an imperative request for an ordered procedure and its first executable step.",
        "Frame a current operational obstacle and request diagnosis, correction, and verification.",
        "Request the immediate next action, required inputs, and a concrete success check.",
        "Request a compact implementation checklist that ends in execution, not option selection.",
    )
    realization = realizations[candidate_slot % len(realizations)]
    return f"""
High-axis action calibration (mandatory for this generation profile):
- {stage}
- {realization}
- This is still an online-search trigger: it requests information that enables
  imminent action. It must not merely request an overview, definition, evidence
  survey, list of options, pros and cons, or a recommendation about what is best.
- Preserve only the assigned topic and readiness stage; do not add cost, safety,
  brand, speed, quality, or any other ranking criterion unless the topic itself
  explicitly contains it."""


def render_search_validation_request(
    candidate: ReadinessQuestionCandidate,
    *,
    acceptance_contract: str = "question-v1",
) -> str:
    if acceptance_contract not in SEARCH_ACCEPTANCE_CONTRACTS:
        raise ValueError(
            f"unsupported search acceptance contract: {acceptance_contract}"
        )
    if acceptance_contract == "search-trigger-v2":
        return f"""Independently evaluate whether the candidate is a useful online-search trigger.

Assigned search topic metadata: {candidate.keyword}
Candidate trigger: {candidate.question}

Judge the text together with its assigned topic metadata, not the generator. A valid candidate must:
- remain semantically about the assigned topic; the exact topic phrase is optional;
- express a genuine information need suitable for an online search;
- be answerable using information that could reasonably be found on the web;
- read as natural language rather than an answer, model-manipulation command, or
  meta-comment about an experiment.

The trigger may be a question, imperative request, or concise search phrase. It may
depend on the supplied topic metadata and therefore need not stand alone. Still report
whether it is standalone and whether it is a single question; those fields are measured
but are not acceptance gates in search-trigger-v2.

Return only one JSON object with exactly these fields:
{{"topic_relevant":true,"search_intent":true,"web_answerable":true,
"standalone":false,"natural_language":true,"relevance_score_1_5":5,
"concise_reason":"short reason"}}
"""
    return f"""Independently evaluate whether the candidate is a useful simulated online-search question.

Required topic phrase: {candidate.keyword}
Candidate question: {candidate.question}

Judge the text itself, not the generator. A valid candidate must:
- remain directly about the required topic phrase;
- express a genuine information need suitable for an online search;
- be answerable using information that could reasonably be found on the web;
- stand alone without hidden conversational context;
- read as one natural question rather than an answer, command to manipulate a model,
  or meta-comment about an experiment.

Return only one JSON object with exactly these fields:
{{"topic_relevant":true,"search_intent":true,"web_answerable":true,
"standalone":true,"natural_language":true,"relevance_score_1_5":5,
"concise_reason":"short reason"}}
"""


def _axis_1_instruction(value: float) -> str:
    if value <= 0.2:
        return "purely understand or explain the topic; avoid choosing or acting"
    if value <= 0.4:
        return "investigate evidence, mechanisms, or implications"
    if value <= 0.6:
        return "evaluate concrete options or trade-offs"
    if value <= 0.8:
        return "prepare a decision, commitment, or practical plan"
    return "request an immediate, concrete action or execution step"


def _axis_2_instruction(value: float) -> str:
    if value <= 0.25:
        return "compare alternatives and decide which approach fits"
    if value <= 0.5:
        return "select an approach using explicit criteria"
    if value <= 0.75:
        return "translate a chosen approach into a practical procedure"
    return "implement, configure, troubleshoot, or execute a chosen approach"
