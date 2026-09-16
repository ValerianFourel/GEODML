"""Check frozen judge requests with a supplied, locally loaded chat tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeTask,
    render_agentic_judge_prompt,
)
from analysis.scripts.check_acl_arr_context_budget import _input_token_count


def check_agentic_judge_context_budget(
    tasks: Sequence[AgenticJudgeTask],
    *,
    tokenizer: Any,
    max_model_len: int,
    max_output_tokens: int = 2048,
    disable_thinking: bool = True,
) -> dict[str, Any]:
    """Count the runner's one-user-message input and reserve its output budget.

    This neither loads a model/tokenizer nor fetches documents. The caller must
    supply the same pinned tokenizer and context limit as the serving process.
    Oversized inputs fail before inference; no request or evidence is truncated.
    """
    for name, value in (("max_model_len", max_model_len), ("max_output_tokens", max_output_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(disable_thinking) is not bool:
        raise ValueError("disable_thinking must be a boolean")
    if not tasks:
        raise ValueError("judge context preflight requires at least one task")
    template_kwargs = {"enable_thinking": False} if disable_thinking else {}
    rows = []
    overflow = []
    for task in tasks:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": render_agentic_judge_prompt(task)}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=True,
            **template_kwargs,
        )
        prompt_tokens = _input_token_count(encoded)
        required_tokens = prompt_tokens + max_output_tokens
        rows.append({
            "judge_task_id": task.judge_task_id,
            "prompt_tokens": prompt_tokens,
            "required_tokens": required_tokens,
        })
        if required_tokens > max_model_len:
            overflow.append(f"{task.judge_task_id} requires {required_tokens} tokens")
    if overflow:
        raise ValueError(
            f"judge context exceeds max_model_len={max_model_len}, including "
            f"{max_output_tokens} reserved output tokens: " + "; ".join(overflow)
        )
    return {
        "format_version": "agentic-judge-context-budget-v1",
        "status": "PASS",
        "scientific_result": False,
        "task_count": len(rows),
        "max_model_len": max_model_len,
        "max_output_tokens": max_output_tokens,
        "disable_thinking": disable_thinking,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "max_required_tokens": max(row["required_tokens"] for row in rows),
        "tasks": rows,
    }
