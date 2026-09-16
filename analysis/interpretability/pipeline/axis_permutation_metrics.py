"""Descriptive, held-out fits of a recorded prompt axis to frozen-pool rankings.

Only pairs that both appear in an observed ranking contribute labels. An omitted
candidate is never treated as losing. The objective sums each task's mean pair
log loss, plus ``ridge / 2 * sum(coefficients**2)``. Thus each task has equal
weight, regardless of ranking length. This is a predictive fit, not a causal
effect estimate or an inferential significance test.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from numbers import Real


def _ids(value: object, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate IDs")
    return list(value)


def ranking_agreement(left: list[str], right: list[str]) -> dict[str, object]:
    """Compare observed order and overlap without assigning ranks to absent IDs."""

    left, right = _ids(left, "left ranking"), _ids(right, "right ranking")
    shared = set(left) & set(right)
    left_positions = {item: index for index, item in enumerate(left)}
    right_positions = {item: index for index, item in enumerate(right)}
    concordant = discordant = 0
    for first, second in combinations(sorted(shared), 2):
        if (left_positions[first] < left_positions[second]) == (
            right_positions[first] < right_positions[second]
        ):
            concordant += 1
        else:
            discordant += 1
    pair_count = concordant + discordant
    k = min(3, len(left), len(right))
    overlap = len(set(left[:k]) & set(right[:k]))
    return {
        "left_count": len(left),
        "right_count": len(right),
        "common_count": len(shared),
        "exact_match": left == right if left and right else None,
        "top1_match": left[0] == right[0] if left and right else None,
        "topk_k": k,
        "topk_intersection_count": overlap,
        "topk_overlap": overlap / k if k else None,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "kendall_tau_common": (concordant - discordant) / pair_count
        if pair_count
        else None,
    }


def _split(question_sha256: str, seed: int, modulus: int) -> str:
    payload = json.dumps([seed, question_sha256], separators=(",", ":")).encode()
    return (
        "test"
        if int(hashlib.sha256(payload).hexdigest(), 16) % modulus == 0
        else "train"
    )


def fit_axis_rankings(
    rows: Sequence[Mapping[str, object]],
    *,
    axis_field: str,
    seed: int = 20260916,
    holdout_modulus: int = 5,
    ridge: float = 1.0,
    min_train: int = 6,
    min_test: int = 2,
) -> dict[str, object]:
    """Fit one axis within one exact candidate-pool/configuration stratum.

    The caller must separate configurations, evidence, models, and scientific
    protocols, and exclude fake outputs. This function also checks pool identity.
    Splits use the question-content hash, so alternate IDs for the same question
    cannot leak across train and test. Minimum sizes count distinct questions.
    A growing dataset never moves an existing question to another split.
    """

    if not isinstance(axis_field, str) or not axis_field.strip():
        raise ValueError("axis_field must be a non-empty string")
    for name, value, minimum in (
        ("seed", seed, None),
        ("holdout_modulus", holdout_modulus, 2),
        ("min_train", min_train, 1),
        ("min_test", min_test, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (minimum is not None and value < minimum)
        ):
            raise ValueError(f"invalid {name}")
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, Real)
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("ridge must be finite and positive")

    prepared = []
    task_keys: set[str] = set()
    pool: set[str] | None = None
    missing_axis = unrankable = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each fit row must be an object")  # noqa: TRY004
        for key in ("task_key", "prompt_id", "question_sha256"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        if re.fullmatch(r"[0-9a-f]{64}", row["question_sha256"]) is None:
            raise ValueError("question_sha256 must be a lowercase SHA-256 digest")
        if row["task_key"] in task_keys:
            raise ValueError("duplicate task_key")
        task_keys.add(row["task_key"])
        candidates = set(_ids(row.get("candidate_ids"), "candidate_ids"))
        if not candidates:
            raise ValueError("candidate_ids must not be empty")
        if pool is not None and candidates != pool:
            raise ValueError("fit rows must have the same candidate pool")
        pool = candidates
        ranking = _ids(row.get("ranking"), "ranking")
        if not set(ranking) <= candidates:
            raise ValueError("ranking contains unknown candidate IDs")
        axes = row.get("axes", {})
        if not isinstance(axes, Mapping):
            raise ValueError("axes must be an object")  # noqa: TRY004
        x = axes.get(axis_field)
        missing = axis_field not in axes
        if not missing and (
            isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
        ):
            raise ValueError(
                f"{axis_field} must be a finite numeric axis, not a boolean"
            )
        missing_axis += missing
        unrankable += len(ranking) < 2
        prepared.append(
            {
                "task_key": row["task_key"],
                "prompt_id": row["prompt_id"],
                "question_sha256": row["question_sha256"],
                "axis_value": None if missing else float(x),
                "ranking": ranking,
                "split": _split(row["question_sha256"], seed, holdout_modulus),
                "eligible": not missing and len(ranking) >= 2,
            }
        )
    prepared.sort(key=lambda row: row["task_key"])
    eligible = [row for row in prepared if row["eligible"]]
    train = [row for row in eligible if row["split"] == "train"]
    test = [row for row in eligible if row["split"] == "test"]
    train_questions = len({row["question_sha256"] for row in train})
    test_questions = len({row["question_sha256"] for row in test})
    support = {}
    for label, subset in (("all", eligible), ("train", train), ("test", test)):
        values = [row["axis_value"] for row in subset]
        support[label] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "unique_values": len(set(values)),
        }
    report = {
        "status": "insufficient_data",
        "axis_field": axis_field,
        "split": {
            "seed": seed,
            "holdout_modulus": holdout_modulus,
            "group_by": "question_sha256",
            "minimum_train_questions": min_train,
            "minimum_test_questions": min_test,
        },
        "counts": {
            "rows": len(prepared),
            "eligible": len(eligible),
            "missing_axis": missing_axis,
            "unrankable": unrankable,
            "questions": len({row["question_sha256"] for row in prepared}),
            "train": len(train),
            "test": len(test),
            "train_questions": train_questions,
            "test_questions": test_questions,
        },
        "support": support,
        "candidate_ids": sorted(pool or []),
        "ridge": float(ridge),
        "split_assignments": {row["task_key"]: row["split"] for row in prepared},
        "axis_model": None,
        "baseline_model": None,
        "predictions": [],
        "interpretation": "Descriptive prediction, not a causal effect or significance test.",
        "weighting": "Equal task weight; within each task only observed pairs share its weight.",
        "pairwise_accuracy_ties": "Predicted probability ties receive half credit.",
    }
    if train_questions < min_train or test_questions < min_test:
        report["reason"] = (
            "Fixed question holdout has too few eligible train or test questions."
        )
        return report
    if support["train"]["unique_values"] < 2:
        report.update(
            status="constant_axis", reason="The training axis has no variation."
        )
        return report

    # NumPy/SciPy are CPU dependencies. Ranking agreement itself needs neither.
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    candidate_ids = report["candidate_ids"]
    index = {candidate: offset for offset, candidate in enumerate(candidate_ids)}
    candidate_count = len(index)
    values = np.asarray([row["axis_value"] for row in train], dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            center, scale = float(values.mean()), float(values.std())
            if not math.isfinite(scale) or scale == 0:
                raise ValueError("training axis cannot be stably standardized")
            for row in eligible:
                row["z"] = (row["axis_value"] - center) / scale
                if not math.isfinite(row["z"]):
                    raise ValueError("axis cannot be stably standardized")
        except FloatingPointError as error:
            raise ValueError("axis cannot be stably standardized") from error
    report["standardization"] = {"training_mean": center, "training_scale": scale}
    left, right, axes, weights = [], [], [], []
    candidate_pair_counts = {candidate: 0 for candidate in candidate_ids}
    for row in train:
        pairs = list(combinations(row["ranking"], 2))
        for first, second in pairs:
            left.append(index[first])
            right.append(index[second])
            axes.append(row["z"])
            weights.append(1.0 / len(pairs))
            candidate_pair_counts[first] += 1
            candidate_pair_counts[second] += 1
    left, right = np.asarray(left), np.asarray(right)
    axes, weights = np.asarray(axes), np.asarray(weights)
    report["training_candidate_pair_counts"] = candidate_pair_counts

    def optimize(with_axis):
        def objective(parameters):
            offsets = parameters[:candidate_count]
            logits = offsets[left] - offsets[right]
            if with_axis:
                slopes = parameters[candidate_count:]
                logits = logits + (slopes[left] - slopes[right]) * axes
            loss = np.dot(weights, np.logaddexp(0.0, -logits))
            loss += 0.5 * ridge * np.dot(parameters, parameters)
            residual = -weights * expit(-logits)
            gradient = ridge * parameters.copy()
            np.add.at(gradient, left, residual)
            np.add.at(gradient, right, -residual)
            if with_axis:
                np.add.at(gradient, candidate_count + left, residual * axes)
                np.add.at(gradient, candidate_count + right, -residual * axes)
            return float(loss), gradient

        result = minimize(
            objective,
            np.zeros(candidate_count * (2 if with_axis else 1)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        offsets = result.x[:candidate_count]
        slopes = result.x[candidate_count:] if with_axis else np.zeros(candidate_count)
        coefficients = []
        for candidate, offset, slope in zip(candidate_ids, offsets, slopes):
            original_slope = float(slope / scale)
            original_intercept = float(offset - original_slope * center)
            if not math.isfinite(original_slope) or not math.isfinite(
                original_intercept
            ):
                raise ValueError(
                    "coefficients cannot be represented on the original axis"
                )
            coefficients.append(
                {
                    "candidate_id": candidate,
                    "intercept": original_intercept,
                    "axis_slope": original_slope,
                }
            )
        return (
            {
                "convergence": {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "objective": float(result.fun),
                },
                "coefficients": coefficients,
            },
            offsets,
            slopes,
        )

    models = {
        name: optimize(with_axis)
        for name, with_axis in (
            ("axis_model", True),
            ("baseline_model", False),
        )
    }
    predictions = []
    for row in eligible:
        prediction = {
            key: row[key]
            for key in (
                "task_key",
                "prompt_id",
                "question_sha256",
                "axis_value",
                "split",
            )
        }
        prediction["observed_ranking"] = row["ranking"]
        for name, (_, offsets, slopes) in models.items():
            utility = offsets + slopes * row["z"]
            if not np.isfinite(utility).all():
                raise ValueError("prediction contains non-finite utilities")
            ordered = sorted(
                candidate_ids,
                key=lambda candidate: (-utility[index[candidate]], candidate),
            )
            logits = np.asarray(
                [
                    utility[index[a]] - utility[index[b]]
                    for a, b in combinations(row["ranking"], 2)
                ]
            )
            probabilities = expit(logits)
            ranking = ordered[: len(row["ranking"])]
            prediction[name] = {
                "predicted_ranking": ranking,
                "predicted_full_ranking": ordered,
                "agreement": ranking_agreement(row["ranking"], ranking),
                "observed_pair_count": len(logits),
                "pairwise_log_loss": float(np.mean(np.logaddexp(0.0, -logits))),
                "pairwise_brier": float(np.mean((1.0 - probabilities) ** 2)),
                "pairwise_accuracy": float(
                    np.mean(
                        np.where(
                            np.abs(probabilities - 0.5) <= 1e-12,
                            0.5,
                            probabilities > 0.5,
                        )
                    )
                ),
            }
        predictions.append(prediction)
    report["predictions"] = predictions
    for name, (model, _, _) in models.items():
        for split, label in (("train", "train"), ("test", "heldout")):
            selected = [row[name] for row in predictions if row["split"] == split]
            summary = {
                "task_count": len(selected),
                "observed_pair_count": sum(
                    row["observed_pair_count"] for row in selected
                ),
            }
            for metric in ("pairwise_log_loss", "pairwise_brier", "pairwise_accuracy"):
                summary[metric] = float(np.mean([row[metric] for row in selected]))
            for metric in (
                "exact_match",
                "top1_match",
                "topk_overlap",
                "kendall_tau_common",
            ):
                comparable = [
                    row["agreement"][metric]
                    for row in selected
                    if row["agreement"][metric] is not None
                ]
                summary[metric] = float(np.mean(comparable)) if comparable else None
                summary[metric + "_count"] = len(comparable)
            model[label] = summary
        report[name] = model
    report["status"] = (
        "fit"
        if all(model[0]["convergence"]["success"] for model in models.values())
        else "optimization_failed"
    )
    report["heldout_improvement"] = {
        "log_loss_reduction": report["baseline_model"]["heldout"]["pairwise_log_loss"]
        - report["axis_model"]["heldout"]["pairwise_log_loss"],
        "brier_reduction": report["baseline_model"]["heldout"]["pairwise_brier"]
        - report["axis_model"]["heldout"]["pairwise_brier"],
    }
    # Fail rather than write NaN or Infinity into a progress report.
    json.dumps(report, allow_nan=False)
    return report
