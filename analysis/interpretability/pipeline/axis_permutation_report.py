"""Incremental, descriptive axis-to-permutation reports over saved observations.

This module never runs inference. Missing rankings stay missing; judge agreement
is not correctness, and observed embedding coordinates are not treatments.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .axis_permutation_metrics import fit_axis_rankings, ranking_agreement

FORMAT_VERSION = "axis-permutation-report-v1"
CONFIG_VERSION = "axis-permutation-study-config-v1"
FIT_DEFAULTS = {"seed": 20260916, "holdout_modulus": 5, "ridge": 1.0,
                "min_train": 6, "min_test": 2}
STRATUM = ("cohort_id", "protocol", "configuration_id", "model_id", "model_revision", "method", "engine", "condition")
WARNINGS = [
    "Exploratory saved-artifact report, not a causal or confirmatory result.",
    "Assigned policy targets and observed embedding coordinates are separate predictors; embeddings do not define the treatment.",
    "Fixed-candidate direct reranking and agentic retrieval are separate protocols. Fits require identical evidence pools.",
    "Judge agreement does not establish ranking correctness. Ranking-visible conversation judgments are not an independent ranking check.",
    "Only jointly ranked candidates define pairwise agreement. Missing candidates are never assigned invented ranks.",
    "Repeated updates and incomplete, selectively completed prompts do not provide sequentially valid significance tests.",
    "Fits use one coordinate at a time. Correlated latent coordinates and question semantics are not controlled experiments.",
    "Full-conversation means the recorded messages and supplied evidence, not fetched web pages or hidden reasoning.",
]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def analysis_settings(config):
    if config.get("format_version") != CONFIG_VERSION:
        raise ValueError("unsupported study config format")
    if not isinstance(config.get("study_id"), str) or not config["study_id"].strip():
        raise ValueError("study_id must be non-empty text")
    axes = config.get("axis_fields")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("axis_fields must explicitly distinguish assigned and observed coordinates")
    for field, spec in axes.items():
        if not isinstance(field, str) or not isinstance(spec, dict):
            raise ValueError("invalid axis specification")  # noqa: TRY004 -- invalid serialized configuration
        if spec.get("role") not in {"assigned", "observed"} or not spec.get("construct"):
            raise ValueError("each axis requires a role and a named construct")
        domain = spec.get("domain", [0.0, 1.0])
        if (not isinstance(domain, list) or len(domain) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in domain)
                or domain[0] >= domain[1]):
            raise ValueError("axis domain must contain two increasing finite numbers")
    fit = {**FIT_DEFAULTS, **config.get("fit", {})}
    if set(fit) != set(FIT_DEFAULTS):
        raise ValueError("unknown fit settings")
    for field in ("seed", "holdout_modulus", "min_train", "min_test"):
        if type(fit[field]) is not int:
            raise ValueError(f"{field} must be an integer")
    if fit["holdout_modulus"] < 2 or fit["min_train"] < 2 or fit["min_test"] < 1:
        raise ValueError("invalid fixed-split sample thresholds")
    if type(fit["ridge"]) not in (int, float) or not math.isfinite(fit["ridge"]) or fit["ridge"] <= 0:
        raise ValueError("ridge must be positive and finite")
    return {"format_version": CONFIG_VERSION, "study_id": config["study_id"],
            "axis_fields": axes, "fit": fit}


def _unique(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result and canonical(result[identity]) != canonical(row):
            raise ValueError(f"conflicting {key}: {identity}")
        result[identity] = row
    return result


def _summary(rows):
    completed = sum(row.get("ranking") is not None and not row.get("fake", False) for row in rows)
    return {"expected": len(rows), "completed": completed, "remaining": len(rows) - completed,
            "percent": round(100 * completed / len(rows), 3) if rows else None,
            "fake_excluded": sum(bool(row.get("fake", False)) for row in rows),
            "empty_rankings": sum(row.get("ranking") == [] and not row.get("fake", False) for row in rows)}


def _identity(row):
    return {field: row.get(field) for field in STRATUM}


def _group_key(row):
    return canonical(_identity(row))


def _aggregate_agreement(rows):
    fields = ("exact_match", "top1_match", "kendall_tau_common", "topk_overlap")
    result = {"comparisons": len(rows), "questions": len({r["question_sha256"] for r in rows}),
              "equal_length": sum(r["metrics"]["left_count"] == r["metrics"]["right_count"] for r in rows),
              "at_least_two_shared_candidates": sum(r["metrics"]["common_count"] >= 2 for r in rows)}
    for field in fields:
        values = [row["metrics"].get(field) for row in rows]
        valid = [float(value) for value in values if type(value) in (bool, int, float)]
        result[field] = {"n": len(valid), "mean": sum(valid) / len(valid) if valid else None}
    return result


def _axis_coverage(tasks, field, spec):
    low, high = spec.get("domain", [0.0, 1.0])
    bins = [{"index": i, "lower": low + (high - low) * i / 10,
             "upper": low + (high - low) * (i + 1) / 10, "expected": 0, "completed": 0} for i in range(10)]
    missing = outside = 0
    values = []
    for row in tasks:
        value = row.get("axes", {}).get(field)
        if value is None:
            missing += 1
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"invalid axis coordinate {field}: {row['task_key']}")
        values.append(value)
        if value < low or value > high:
            outside += 1
            continue
        bucket = bins[min(9, int(10 * (value - low) / (high - low)))]
        bucket["expected"] += 1
        bucket["completed"] += row.get("ranking") is not None and not row.get("fake", False)
    return {"field": field, **spec, "missing": missing, "outside_domain": outside,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "bins": bins}


def _ledger(row):
    # Evidence and optional coordinates can arrive after a task is planned.
    # Once an observation arrives, its evidence and provenance become immutable.
    identity = {key: row.get(key) for key in (*STRATUM, "source_kind", "prompt_id", "question",
                "question_sha256", "keyword", "source_cell_id", "top_n")}
    if row.get("source_kind") == "direct":
        identity["planned_pool"] = row.get("pool_id")
        identity["planned_candidates"] = sorted(row["candidate_ids"])
    observed = row.get("ranking") is not None and not row.get("fake", False)
    return {"identity": digest(identity), "axes": row.get("axes", {}),
            "observation": digest({key: row.get(key) for key in ("ranking", "pool_id", "fit_pool_id", "candidate_ids",
                                   "source_result_sha256", "source_trace_sha256", "execution_configuration_id",
                                   "configuration_mixed")}) if observed else None}


def _pairing_key(task):
    return tuple(task.get(key) for key in ("question_sha256", "engine", "pool_id", "model_id", "model_revision"))


def build_report(study, config):
    """Build deterministic report contents; source readers own provenance checks."""
    settings = analysis_settings(config)
    tasks = sorted(_unique(study.get("tasks", []), "task_key").values(), key=lambda r: r["task_key"])
    by_task = {row["task_key"]: row for row in tasks}
    judgments = sorted(study.get("judgments", []), key=lambda r: tuple(r[k] for k in
                       ("judge_task_id", "task_key", "judge_model_id", "judge_model_revision", "protocol")))
    completed = [row for row in tasks if row.get("ranking") is not None and not row.get("fake", False)]
    strata = defaultdict(list)
    for row in tasks:
        strata[_group_key(row)].append(row)
    coverage = [{**_identity(rows[0]), **_summary(rows),
                 "axis_coverage": [_axis_coverage(rows, name, spec) for name, spec in sorted(settings["axis_fields"].items())]}
                for _, rows in sorted(strata.items())]

    fit_groups = defaultdict(list)
    fit_exclusions = Counter()
    for row in completed:
        if row.get("configuration_mixed"):
            fit_exclusions["mixed_execution_configuration"] += 1
            continue
        if row.get("source_kind") == "agentic" and not row.get("execution_configuration_id"):
            fit_exclusions["unknown_execution_configuration"] += 1
            continue
        if len(row["candidate_ids"]) < 2:
            fit_exclusions["fewer_than_two_candidates"] += 1
            continue
        if row.get("pool_id") is not None and len(row["candidate_ids"]) >= 2:
            # Different top-k contracts and retrieved evidence never share a fit.
            fit_groups[(_group_key(row), row.get("execution_configuration_id", row["configuration_id"]),
                        row.get("fit_pool_id", row["pool_id"]),
                        row.get("keyword", ""), len(row["ranking"]))].append(row)
    fits = []
    for (identity, execution_config, pool_id, keyword, top_n), rows in sorted(fit_groups.items()):
        for field, axis in sorted(settings["axis_fields"].items()):
            fit_id = digest([json.loads(identity), execution_config, pool_id, keyword, top_n, field])[:24]
            fitted = fit_axis_rankings(rows, axis_field=field, **settings["fit"])
            fits.append({"fit_id": fit_id, **json.loads(identity), "pool_id": pool_id,
                         "execution_configuration_id": execution_config,
                         "keyword": keyword, "top_n": top_n, "axis_role": axis["role"], "axis_construct": axis["construct"],
                         **fitted})

    comparisons = []
    seen_judgments = set()
    direct_index = defaultdict(list)
    for row in completed:
        if row.get("source_kind") == "direct" and row.get("pool_id") is not None:
            direct_index[_pairing_key(row)].append(row)
    unmatched_direct_judgments = 0
    for judge in judgments:
        if judge.get("fake", False):
            continue
        task = by_task.get(judge["task_key"])
        if task is None:
            raise ValueError("judgment references an unknown generator task")
        if task.get("ranking") is None or task.get("fake", False):
            continue
        identity = (judge["judge_task_id"], judge["judge_model_id"], judge["judge_model_revision"], judge["protocol"])
        if identity in seen_judgments:
            raise ValueError("duplicate normalized judge identity")
        seen_judgments.add(identity)
        allowed = set(task["candidate_ids"])
        alternatives = direct_index.get(_pairing_key(task), []) if task.get("source_kind") != "direct" else []
        unmatched_direct_judgments += task.get("source_kind") != "direct" and not alternatives
        for purpose, key in (("ideal_relevance", "ideal_ranking"), ("realized_support", "support_ranking")):
            ranking = judge[key]
            if not set(ranking).issubset(allowed):
                raise ValueError("judge ranking contains a candidate outside the verified pool")
            for compared in [task, *alternatives]:
                comparisons.append({"task_key": compared["task_key"], "prompt_id": compared["prompt_id"],
                                "question_sha256": compared["question_sha256"], **_identity(compared),
                                "family": "source_generator_vs_judge" if compared is task else "fixed_candidate_direct_vs_judge",
                                "judge_source_task_key": task["task_key"], "judge_source_configuration": _identity(task),
                                "judge_task_id": judge["judge_task_id"], "judge_model_id": judge["judge_model_id"],
                                "judge_model_revision": judge["judge_model_revision"], "judge_protocol": judge["protocol"],
                                "ranking_visible": judge["ranking_visible"], "purpose": purpose,
                                "generator_ranking": compared["ranking"], "judge_ranking": ranking,
                                "scores": judge.get("scores", {}),
                                "metrics": ranking_agreement(compared["ranking"], ranking)})
    agreement_groups = defaultdict(list)
    for row in comparisons:
        key = {**_identity(row), **{field: row[field] for field in (
            "family", "judge_source_configuration", "judge_model_id", "judge_model_revision", "judge_protocol", "ranking_visible", "purpose")}}
        agreement_groups[canonical(key)].append(row)
    agreements = [{**json.loads(key), **_aggregate_agreement(rows)} for key, rows in sorted(agreement_groups.items())]

    question_groups = defaultdict(list)
    for row in tasks:
        question_groups[(row["cohort_id"], row["question_sha256"])].append(row)
    fit_by_task = defaultdict(list)
    fit_diagnostics = defaultdict(list)
    for fit in fits:
        for task_key, split in fit.get("split_assignments", {}).items():
            fit_diagnostics[task_key].append({"fit_id": fit["fit_id"], "axis_field": fit["axis_field"],
                "status": fit["status"], "reason": fit.get("reason"), "split": split})
        for prediction in fit.get("predictions", []):
            fit_by_task[prediction["task_key"]].append({"fit_id": fit["fit_id"], **prediction})
    compare_by_task = defaultdict(list)
    for comparison in comparisons:
        compare_by_task[comparison["task_key"]].append(comparison)
    questions = []
    for (cohort, question_hash), rows in sorted(question_groups.items()):
        if len({r["question"] for r in rows}) != 1:
            raise ValueError("inconsistent text for one question hash")
        questions.append({"cohort_id": cohort, "question_sha256": question_hash,
                          "prompt_ids": sorted({r["prompt_id"] for r in rows}),
                          "question": rows[0]["question"], **_summary(rows),
                          "tasks": [{**row, "fits": fit_by_task[row["task_key"]],
                                     "fit_diagnostics": fit_diagnostics[row["task_key"]],
                                     "judge_comparisons": compare_by_task[row["task_key"]]} for row in rows]})
    planned_judges = study.get("judge_tasks", [])
    observed_judge_ids = {(j["judge_task_id"], j["judge_model_id"], j["judge_model_revision"], j["protocol"])
                          for j in judgments if not j.get("fake", False)}
    pending = [{"stage": "generation", "task_key": r["task_key"], "prompt_id": r["prompt_id"],
                **_identity(r)} for r in tasks if r.get("ranking") is None or r.get("fake", False)]
    pending_judge_ids = set()
    for judge in planned_judges:
        identity = (judge["judge_task_id"], judge["judge_model_id"], judge["judge_model_revision"], judge["protocol"])
        if identity not in observed_judge_ids and identity not in pending_judge_ids:
            pending.append({"stage": "judging", **judge})
            pending_judge_ids.add(identity)
    expected_judgments = len({(j["judge_task_id"], j["judge_model_id"], j["judge_model_revision"], j["protocol"])
                             for j in planned_judges})
    planned_judgment_ledger = {}
    for judge in planned_judges:
        key = digest([judge[k] for k in ("judge_task_id", "judge_model_id", "judge_model_revision", "protocol")])
        linked_task = judge.get("task_key")
        previous_link = planned_judgment_ledger.get(key)
        if previous_link is not None and linked_task is not None and previous_link != linked_task:
            raise ValueError("judge plan identity links conflicting source tasks")
        planned_judgment_ledger[key] = linked_task if linked_task is not None else previous_link
    if any(digest(list(identity)) not in planned_judgment_ledger for identity in observed_judge_ids):
        raise ValueError("completed judgment has no materialized plan")
    return {
        "format_version": FORMAT_VERSION, "study_id": settings["study_id"], "scientific_result": False,
        "interpretation": "exploratory", "settings": settings, "warnings": WARNINGS,
        "study_availability": {kind: "planned" if any(row.get("source_kind") == kind for row in tasks)
                               else "unavailable" for kind in ("direct", "agentic")},
        "generation": _summary(tasks), "questions_count": len(questions),
        "judging": {"expected_materialized": expected_judgments, "completed": len(observed_judge_ids),
                    "remaining_materialized": len(pending_judge_ids),
                    "without_exact_fixed_candidate_direct_match": unmatched_direct_judgments,
                    "unplanned_generator_tasks": len(set(by_task) - {j["task_key"] for j in planned_judges})},
        "coverage": coverage, "fits": fits, "fit_exclusions": dict(fit_exclusions),
        "fit_status_counts": dict(Counter(f["status"] for f in fits)),
        "agreement": agreements, "comparisons": comparisons, "questions": questions, "pending": pending,
        "sources": study.get("sources", []), "issues": study.get("issues", []),
        "task_ledger": {r["task_key"]: _ledger(r) for r in tasks},
        "planned_judgment_ledger": planned_judgment_ledger,
        "judgment_ledger": {digest([j["judge_task_id"], j["judge_model_id"], j["judge_model_revision"], j["protocol"]]): digest(j)
                            for j in judgments if not j.get("fake", False)},
    }


def render_markdown(report):
    generation = report["generation"]
    lines = [f"# {report['study_id']}: axis-to-permutation report", "",
             "Exploratory artifact report. No causal or judge-quality claim.", "",
             f"Fixed-candidate direct study: {report['study_availability']['direct']}. Agentic study: {report['study_availability']['agentic']}.",
             f"Generation: {generation['completed']}/{generation['expected']} saved; {generation['remaining']} missing.",
             f"Questions: {report['questions_count']}. Empty rankings: {generation['empty_rankings']}.",
             f"Judgments: {report['judging']['completed']}/{report['judging']['expected_materialized']} materialized tasks.",
             f"Generator tasks without a judge plan: {report['judging']['unplanned_generator_tasks']}.", "",
             "## Coverage by protocol and model", "",
             "| Cohort | Protocol | Model | Method | Engine | Condition | Saved | Planned |",
             "| --- | --- | --- | --- | --- | --- | ---: | ---: |"]
    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for row in report["coverage"]:
        values = [row.get(k) for k in ("cohort_id", "protocol", "model_id", "method", "engine", "condition", "completed", "expected")]
        lines.append("| " + " | ".join(map(text, values)) + " |")
    lines.extend(["", "## Axis fits", "", "Each fit uses one exact evidence pool and model/protocol configuration.",
                  "Question-hash holdouts remain fixed as observations arrive. Missing ranks are not imputed.", "",
                  f"Fit status counts: `{json.dumps(report['fit_status_counts'], sort_keys=True)}`.",
                  f"Completed observations excluded from fits: `{json.dumps(report['fit_exclusions'], sort_keys=True)}`.",
                  "Coefficients, training and held-out losses, baseline comparisons, and predictions are in `fits.jsonl`.",
                  "Per-question coverage, observed rankings, predictions, and disagreements are in `questions.jsonl`.", "",
                  "| Fit ID | Coordinate | Status | Train questions | Held-out questions | Held-out loss improvement |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for fit in report["fits"]:
        gain = fit.get("heldout_improvement", {}).get("log_loss_reduction")
        values = [fit["fit_id"], fit["axis_field"], fit["status"], fit["counts"]["train_questions"],
                  fit["counts"]["test_questions"], "unavailable" if gain is None else f"{gain:.4f}"]
        lines.append("| " + " | ".join(map(text, values)) + " |")
    lines.extend(["", "Positive loss improvement means the axis model beats the axis-free baseline on held-out questions.", "",
                  "## Judge agreement", "", "Ideal relevance and realized support are different judgments and remain separate.",
                  "Exact cross-study matches require the same question, engine, model revision, URLs, titles, and evidence text.",
                  f"Judgments without a compatible fixed-candidate direct ranking: {report['judging']['without_exact_fixed_candidate_direct_match']}.", "",
                  "| Family | Generator | Method | Engine | Condition | Judge | Protocol | Ranking visible | Purpose | N | Full-list exact (n) | Top 1 (n) | Common-item tau (n) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in report["agreement"]:
        values = [row[k] for k in ("family", "model_id", "method", "engine", "condition", "judge_model_id", "judge_protocol", "ranking_visible", "purpose", "comparisons")]
        for key in ("exact_match", "top1_match", "kendall_tau_common"):
            metric = row[key]
            values.append("not comparable (0)" if metric["mean"] is None else f"{metric['mean']:.3f} ({metric['n']})")
        lines.append("| " + " | ".join(map(text, values)) + " |")
    lines.extend(["", "## Limits", "", *[f"- {warning}" for warning in report["warnings"]], "",
                  f"Input issues: {len(report['issues'])}. Inspect `report.json` before interpreting results.", ""])
    return "\n".join(lines)


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="wb", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_report(report, output_dir):
    """Publish immutable snapshots and one atomic latest pointer, under a lock."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".report.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest_path = output / "latest.json"
        previous = None
        latest = None
        if latest_path.exists():
            latest = json.loads(latest_path.read_text())
            if latest.get("format_version") != FORMAT_VERSION:
                raise ValueError("output belongs to a different report format")
            if not re.fullmatch(r"[0-9a-f]{64}", latest.get("snapshot_id", "")):
                raise ValueError("invalid snapshot identity")
            previous_path = output / "snapshots" / latest["snapshot_id"] / "report.json"
            payload = previous_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != latest["report_sha256"]:
                raise ValueError("previous report changed on disk")
            previous = json.loads(payload)
            if previous["settings"] != report["settings"]:
                raise ValueError("analysis settings changed; choose a new report output directory")
            for key, old in previous["task_ledger"].items():
                new = report["task_ledger"].get(key)
                if new is None or old["identity"] != new["identity"]:
                    raise ValueError("planned task changed or disappeared; inspect input provenance")
                if any(new["axes"].get(axis) != value for axis, value in old["axes"].items()):
                    raise ValueError("previously recorded axis coordinate changed or disappeared")
                if old["observation"] is not None and old["observation"] != new["observation"]:
                    raise ValueError("completed ranking changed or disappeared; do not rewrite history")
            for key, old in previous["judgment_ledger"].items():
                if report["judgment_ledger"].get(key) != old:
                    raise ValueError("completed judgment changed or disappeared; do not rewrite history")
            for key, old_link in previous["planned_judgment_ledger"].items():
                new_plans = report["planned_judgment_ledger"]
                if key not in new_plans or (old_link is not None and new_plans[key] != old_link):
                    raise ValueError("planned judgment changed or disappeared; inspect input provenance")
        snapshot_id = digest(report)
        snapshots = output / "snapshots"
        snapshots.mkdir(exist_ok=True)
        destination = snapshots / snapshot_id
        if destination.exists():
            if json.loads((destination / "report.json").read_text()) != report:
                raise ValueError("snapshot directory contains unexpected content")
            if latest is not None and latest["snapshot_id"] == snapshot_id:
                return destination
        else:
            with tempfile.TemporaryDirectory(prefix=".report-staging-", dir=output) as temporary:
                staging = Path(temporary) / snapshot_id
                staging.mkdir()
                _atomic_json(staging / "report.json", report)
                (staging / "report.md").write_text(render_markdown(report), encoding="utf-8")
                for key in ("questions", "fits", "comparisons", "pending"):
                    with (staging / f"{key}.jsonl").open("wb") as stream:
                        for row in report[key]:
                            stream.write(canonical(row) + b"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                os.rename(staging, destination)
        _atomic_json(latest_path, {"format_version": FORMAT_VERSION, "snapshot_id": snapshot_id,
                     "report_sha256": hashlib.sha256((destination / "report.json").read_bytes()).hexdigest(),
                     "updated_at": datetime.now(timezone.utc).isoformat(),
                     "previous_snapshot_id": digest(previous) if previous is not None else None,
                     "new_completed": report["generation"]["completed"] - (previous["generation"]["completed"] if previous else 0),
                     "new_judgments": report["judging"]["completed"] - (previous["judging"]["completed"] if previous else 0)})
        return destination
