def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _array(item, minimum=0, maximum=None):
    return {"type": "array", "items": item, "minItems": minimum,
            **({"maxItems": maximum} if maximum is not None else {})}


def _enum(values):
    return {"type": "string", "enum": list(values)}


def span_schema(word_ids):
    return _object({k: {"type": "string", "enum": list(word_ids)} for k in ("first", "last")})


def map_schema(word_ids):
    span = span_schema(word_ids)
    return _object({
        "status": _enum(["ready", "unusable"]),
        "claims": _array(_object({"spans": _array(span, 1), "kind": _enum(KINDS),
                                  "roles": _array(_enum(ROLES), 1, 2)})),
        "excluded": _array(_object({"span": span, "reason": _enum(["non_substantive", "pure_abstention"])})),
        "note": {"type": "string", "maxLength": 400}})


def source_schema(word_ids, claim_ids, evidence_ids):
    return _object({
        "status": _enum(["scored", "uncertain", "map_issue"]),
        "findings": _array(_object({
            "claim_id": {"type": "string", "enum": list(claim_ids)},
            "relation": _enum(["full", "partial", "contradicted", "uncertain"]),
            "spans": _array(span_schema(word_ids), 1),
            "evidence_ids": _array({"type": "string", "enum": list(evidence_ids)}, 1, 3),
            "note": {"type": "string", "maxLength": 320}})),
        "importance": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 5}, {"type": "null"}]},
        "note": {"type": "string", "maxLength": 400}})


def _answer_block(answer):
    units = "\n".join(f"[{u['unit_id']}] {u['text']}" for u in v3.unitize_answer(answer))
    indexed = "\n".join(f"[{w['word_id']}] {w['text']}" for w in words(answer))
    return f"ORIGINAL ANSWER UNITS:\n{units}\n\nANSWER WORD IDS:\n{indexed}"


def _item(task, inputs, max_tokens):
    required = {"request", "answer", "preprocessing"}
    optional = set()
    if task == "source_importance":
        required |= {"answer_map", "source_title", "source_text"}
        optional = {"diagnostic_passage_ids", "diagnostic_seed"}
    if not required <= inputs.keys() or inputs.keys() - required - optional:
        raise ValueError("unexpected or missing semantic input fields")
    diagnostic_seed = inputs.get("diagnostic_seed")
    if diagnostic_seed is not None and (type(diagnostic_seed) is not int or not 0 <= diagnostic_seed < 2**32):
        raise ValueError("invalid diagnostic seed")
    request, answer = inputs["request"], inputs["answer"]
    if not isinstance(request, str) or not request.strip() or not isinstance(answer, str) or not words(answer):
        raise ValueError("nonempty request and answer required")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if inputs["preprocessing"] not in (PREPROCESSING_VERSION, PROSE_MASK_VERSION):
        raise ValueError("unknown preprocessing")
    word_ids = [w["word_id"] for w in words(answer)]
    prefix = f"\n\nUSER REQUEST:\n{request}\n\n{_answer_block(answer)}"
    if task == "answer_map":
        schema = map_schema(word_ids)
        prompt = MAP_INSTRUCTIONS + prefix
        validator = lambda raw: validate_map(raw, answer=answer)
    elif task == "source_importance":
        answer_map = inputs["answer_map"]
        verify_map(answer_map, answer)
        if answer_map["eligibility"] != "eligible":
            raise ValueError("cannot judge a source against an ineligible map")
        evidence = v3.unitize_source(title=inputs["source_title"], text=inputs["source_text"])
        if not evidence:
            raise ValueError("source is unassessable")
        if "diagnostic_passage_ids" in inputs:
            evidence = _rename_passages(evidence, inputs["diagnostic_passage_ids"])
        schema = source_schema(word_ids, [c["claim_id"] for c in answer_map["claims"]
                                         if c["kind"] != "global_absence"], [e["unit_id"] for e in evidence])
        prompt = (SOURCE_INSTRUCTIONS + prefix + "\n\nFIXED ANSWER MAP:\n" +
                  json.dumps(answer_map, ensure_ascii=False, sort_keys=True) + "\n\nSOURCE PASSAGES:\n" +
                  "\n".join(f"[{e['unit_id']}] {e['text']}" for e in evidence))
        validator = lambda raw: validate_source(raw, answer=answer, answer_map=answer_map, evidence_units=evidence)
    else:
        raise ValueError("unknown v4 task")
    record = {"protocol": PROTOCOL, "task_version": TASK_VERSION, "task": task, "inputs": copy.deepcopy(inputs),
              "max_tokens": max_tokens, "temperature": 0.0, "retry_contract": RETRY_CONTRACT,
              "eligibility_version": ELIGIBILITY_VERSION,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "schema_sha256": _digest(schema)}
    tid = TASK_VERSION + "-" + _digest(record)[:24]
    record.update(judge_task_id=tid, seed=diagnostic_seed if diagnostic_seed is not None else
                  int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16))
    return {"base": {"judge_task_id": tid, "task": task, "protocol": PROTOCOL},
            "record": record, "prompt": prompt, "schema": schema, "schema_name": task + "_v4",
            "validator": validator, "max_tokens": max_tokens, "temperature": 0.0,
            "seed": record["seed"], "maximum_validation_attempts": 2,
            "validation_feedback_contract": RETRY_CONTRACT,
            "prompt_sha256": record["prompt_sha256"], "schema_sha256": record["schema_sha256"]}
