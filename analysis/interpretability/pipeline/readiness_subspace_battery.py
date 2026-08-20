"""Nonlinear and cross-representation robustness battery for readiness maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .readiness_embedding_map import _ranks
from .readiness_hf_dataset import (
    atomic_json,
    atomic_text,
    read_json,
    read_jsonl,
    sha256_file,
)
from .readiness_hf_subspace_comparison import (
    _load_subspace_manifest,
    _validate_frozen_design,
)


READINESS_ROBUSTNESS_BATTERY_VERSION = "readiness-robustness-battery-v1"
TARGET_NAMES = (
    "overall_readiness",
    "information_seeking_reverse",
    "evaluation",
    "selection_commitment",
    "action_implementation",
)
MODEL_KINDS = ("axis_1_linear", "two_axis_linear", "two_axis_polynomial", "additive_cubic_spline")


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    item_count: int
    spearman: float
    pearson: float
    r_squared: float
    mean_absolute_error: float
    baseline_mean_absolute_error: float
    relative_mae_improvement: float


def run_readiness_subspace_robustness_battery(
    *,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    git_commit_sha: str,
    bootstrap_replicates: int = 1000,
    permutation_replicates: int = 200,
    random_seed: int = 20260820,
    minimum_source_items_per_split: int = 50,
) -> dict[str, object]:
    """Evaluate two frozen maps without refitting their 4096-D directions."""

    if bootstrap_replicates < 100 or permutation_replicates < 50:
        raise ValueError("battery requires at least 100 bootstraps and 50 permutations")
    if minimum_source_items_per_split < 10:
        raise ValueError("minimum source size must be at least ten")
    if not git_commit_sha.strip():
        raise ValueError("git_commit_sha must be nonempty")
    reference = Path(reference_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    output = Path(output_dir).resolve()
    if reference == candidate:
        raise ValueError("reference and candidate directories must differ")
    if output.exists():
        raise ValueError(f"refusing to overwrite robustness battery: {output}")

    reference_manifest = _load_subspace_manifest(reference)
    candidate_manifest = _load_subspace_manifest(candidate)
    frozen_design = _validate_frozen_design(reference_manifest, candidate_manifest)
    _validate_consensus_artifact(reference, reference_manifest)
    _validate_consensus_artifact(candidate, candidate_manifest)
    reference_consensus = read_jsonl(reference / "readiness_consensus.jsonl")
    candidate_consensus = read_jsonl(candidate / "readiness_consensus.jsonl")
    if reference_consensus != candidate_consensus:
        raise ValueError("reference and candidate consensus artifacts differ")

    data = _aligned_data(reference, candidate, reference_consensus)
    development = data["split"] == "development"
    confirmation = data["split"] == "confirmation"
    if development.sum() < 100 or confirmation.sum() < 100:
        raise ValueError("battery requires at least 100 usable items in each split")
    y_dev = data["targets"][development]
    y_confirm = data["targets"][confirmation]
    ids_dev = data["item_ids"][development]
    source_dev = data["sources"][development]
    source_confirm = data["sources"][confirmation]

    representation_results = {}
    models = {}
    predictions = {}
    for name, coordinates in (
        ("reference", data["reference_axes"]),
        ("candidate", data["candidate_axes"]),
    ):
        x_dev = coordinates[development]
        x_confirm = coordinates[confirmation]
        representation_results[name] = {}
        models[name] = {}
        predictions[name] = {}
        for kind in MODEL_KINDS:
            model = _fit_model(x_dev, y_dev, ids_dev, kind=kind)
            predicted = _predict_model(model, x_confirm)
            models[name][kind] = model
            predictions[name][kind] = predicted
            representation_results[name][kind] = {
                "selected_ridge_penalty": model["penalty"],
                "feature_count": int(model["coefficient"].shape[0]),
                "targets": _target_metrics(y_confirm, predicted, model["label_mean"]),
                "macro_r_squared": _macro_metric(y_confirm, predicted, model["label_mean"], "r_squared"),
                "macro_mean_absolute_error": float(np.mean(np.abs(y_confirm - predicted))),
            }
        linear_1d = representation_results[name]["axis_1_linear"]["macro_r_squared"]
        linear_2d = representation_results[name]["two_axis_linear"]["macro_r_squared"]
        spline = representation_results[name]["additive_cubic_spline"]["macro_r_squared"]
        polynomial = representation_results[name]["two_axis_polynomial"]["macro_r_squared"]
        representation_results[name]["incremental_tests"] = {
            "axis_2_macro_r_squared_gain": float(linear_2d - linear_1d),
            "spline_macro_r_squared_gain_over_linear": float(spline - linear_2d),
            "polynomial_macro_r_squared_gain_over_linear": float(polynomial - linear_2d),
            "axis_1_spline_monotonic_fraction": _axis_1_monotonic_fraction(
                models[name]["additive_cubic_spline"]
            ),
        }

    alignment = _cross_embedding_alignment(
        data["reference_axes"],
        data["candidate_axes"],
        development,
        confirmation,
    )
    scalar = _scalar_predictions(reference, candidate, data["item_ids"], data["split"])
    scalar_confirmation = confirmation
    direct_scalar = {
        "reference": asdict(
            _metrics(
                y_confirm[:, 0],
                scalar["reference"][scalar_confirmation],
                float(np.mean(y_dev[:, 0])),
            )
        ),
        "candidate": asdict(
            _metrics(
                y_confirm[:, 0],
                scalar["candidate"][scalar_confirmation],
                float(np.mean(y_dev[:, 0])),
            )
        ),
        "cross_embedding_confirmation": _agreement(
            scalar["reference"][confirmation],
            scalar["candidate"][confirmation],
        ),
    }

    source_transfer = {
        name: _leave_one_source_out(
            data[coordinates_name][development],
            y_dev,
            ids_dev,
            source_dev,
            data[coordinates_name][confirmation],
            y_confirm,
            source_confirm,
            penalty=float(models[name]["two_axis_linear"]["penalty"]),
            minimum_items=minimum_source_items_per_split,
        )
        for name, coordinates_name in (
            ("reference", "reference_axes"),
            ("candidate", "candidate_axes"),
        )
    }
    bootstrap = _bootstrap_evidence(
        y_confirm=y_confirm,
        scalar_reference=scalar["reference"][confirmation],
        scalar_candidate=scalar["candidate"][confirmation],
        reference_predictions=predictions["reference"],
        candidate_predictions=predictions["candidate"],
        reference_label_mean=models["reference"]["two_axis_linear"]["label_mean"],
        candidate_label_mean=models["candidate"]["two_axis_linear"]["label_mean"],
        aligned_reference=alignment["reference_confirmation"],
        aligned_candidate=alignment["candidate_confirmation_aligned"],
        replicates=bootstrap_replicates,
        random_seed=random_seed,
    )
    permutation = {
        "reference": _permutation_control(
            data["reference_axes"][development],
            y_dev,
            ids_dev,
            data["reference_axes"][confirmation],
            y_confirm,
            observed=predictions["reference"]["two_axis_linear"][:, 0],
            penalty=float(models["reference"]["two_axis_linear"]["penalty"]),
            replicates=permutation_replicates,
            random_seed=random_seed + 101,
        ),
        "candidate": _permutation_control(
            data["candidate_axes"][development],
            y_dev,
            ids_dev,
            data["candidate_axes"][confirmation],
            y_confirm,
            observed=predictions["candidate"]["two_axis_linear"][:, 0],
            penalty=float(models["candidate"]["two_axis_linear"]["penalty"]),
            replicates=permutation_replicates,
            random_seed=random_seed + 202,
        ),
    }
    checks = _evidence_checks(
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
        representation_results=representation_results,
        alignment=alignment,
        bootstrap=bootstrap,
        source_transfer=source_transfer,
        permutation=permutation,
    )
    assessment = _assessment(checks)
    payload = {
        "format_version": READINESS_ROBUSTNESS_BATTERY_VERSION,
        "git_commit_sha": git_commit_sha,
        "reference": _identity(reference, reference_manifest),
        "candidate": _identity(candidate, candidate_manifest),
        "frozen_design": frozen_design,
        "sample": {
            "development_usable_items": int(development.sum()),
            "confirmation_usable_items": int(confirmation.sum()),
            "target_names": list(TARGET_NAMES),
        },
        "direct_scalar_map": direct_scalar,
        "representation_models": representation_results,
        "cross_embedding_alignment": {
            key: value
            for key, value in alignment.items()
            if not isinstance(value, np.ndarray)
        },
        "leave_one_source_out": source_transfer,
        "bootstrap": bootstrap,
        "permutation_controls": permutation,
        "evidence_checks": checks,
        "assessment": assessment,
        "interpretation_guard": (
            "This exploratory robustness battery tests a frozen semantic map. It is "
            "not causal evidence, does not redefine B, and does not validate direct "
            "decoding of readiness coordinates into text."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_json(temporary / "readiness_robustness_battery.json", payload)
        atomic_text(
            temporary / "readiness_robustness_battery_report.md",
            _render_report(payload),
        )
        artifacts = {
            name: {
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in (
                "readiness_robustness_battery.json",
                "readiness_robustness_battery_report.md",
            )
        }
        manifest = {
            "format_version": READINESS_ROBUSTNESS_BATTERY_VERSION,
            "git_commit_sha": git_commit_sha,
            "reference_map_id": reference_manifest["map_id"],
            "candidate_map_id": candidate_manifest["map_id"],
            "bootstrap_replicates": bootstrap_replicates,
            "permutation_replicates": permutation_replicates,
            "random_seed": random_seed,
            "assessment": assessment,
            "artifacts": artifacts,
        }
        atomic_json(temporary / "battery_manifest.json", manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _aligned_data(reference: Path, candidate: Path, consensus_rows):
    labels = {
        str(row["item_id"]): row
        for row in consensus_rows
        if bool(row["usable_for_axis"])
    }
    left = _axis_index(reference)
    right = _axis_index(candidate)
    if set(left) != set(right):
        raise ValueError("supervised coordinate identities differ")
    keys = sorted(key for key in left if key[1] in labels)
    targets = []
    for _, item_id in keys:
        row = labels[item_id]
        targets.append(
            (
                float(row["overall_readiness_0_100"]) / 100.0,
                (7.0 - float(row["information_seeking_1_7"])) / 6.0,
                (float(row["evaluation_1_7"]) - 1.0) / 6.0,
                (float(row["selection_commitment_1_7"]) - 1.0) / 6.0,
                (float(row["action_implementation_1_7"]) - 1.0) / 6.0,
            )
        )
    return {
        "split": np.asarray([key[0] for key in keys]),
        "item_ids": np.asarray([key[1] for key in keys]),
        "sources": np.asarray([left[key][2] for key in keys]),
        "reference_axes": np.asarray([left[key][:2] for key in keys], dtype=np.float64),
        "candidate_axes": np.asarray([right[key][:2] for key in keys], dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
    }


def _axis_index(root: Path):
    indexed = {}
    for row in read_jsonl(root / "readiness_supervised_subspace_coordinates.jsonl"):
        key = (str(row["split"]), str(row["item_id"]))
        if key in indexed:
            raise ValueError(f"duplicate coordinate identity: {key}")
        values = (float(row["axis_1"]), float(row["axis_2"]), str(row["source_name"]))
        if not np.isfinite(values[:2]).all():
            raise ValueError(f"nonfinite coordinates: {key}")
        indexed[key] = values
    return indexed


def _scalar_predictions(reference, candidate, item_ids, splits):
    result = {}
    expected_keys = list(zip(splits.tolist(), item_ids.tolist()))
    for name, root in (("reference", reference), ("candidate", candidate)):
        indexed = {
            (str(row["evaluation_split"]), str(row["item_id"])): float(
                row["observed_readiness_0_1"]
            )
            for row in read_jsonl(root / "readiness_embedding_coordinates.jsonl")
        }
        if any(key not in indexed for key in expected_keys):
            raise ValueError(f"{name} scalar coordinates omit usable items")
        result[name] = np.asarray([indexed[key] for key in expected_keys])
    return result


def _fit_model(x, y, item_ids, *, kind, fixed_penalty=None):
    transform = _fit_transform(x, kind)
    features = _apply_transform(transform, x)
    penalties = (0.01, 0.1, 1.0, 10.0, 100.0)
    if fixed_penalty is None:
        fold = np.asarray([int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5 for value in item_ids])
        scores = []
        for penalty in penalties:
            errors = []
            for held in range(5):
                train = fold != held
                test = ~train
                if train.sum() < 10 or not test.any():
                    continue
                label_mean, coefficient = _ridge(features[train], y[train], penalty)
                errors.append(float(np.mean(np.abs(y[test] - (label_mean + features[test] @ coefficient)))))
            scores.append((float(np.mean(errors)), penalty))
        penalty = min(scores)[1]
    else:
        penalty = fixed_penalty
    label_mean, coefficient = _ridge(features, y, penalty)
    return {
        "kind": kind,
        "transform": transform,
        "penalty": float(penalty),
        "label_mean": label_mean,
        "coefficient": coefficient,
    }


def _predict_model(model, x):
    return model["label_mean"] + _apply_transform(model["transform"], x) @ model["coefficient"]


def _fit_transform(x, kind):
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("coordinate variance is degenerate")
    z = (x - center) / scale
    knots = tuple(tuple(float(value) for value in np.quantile(z[:, axis], (0.2, 0.4, 0.6, 0.8))) for axis in range(2))
    raw = _raw_features(z, kind, knots)
    feature_center = np.mean(raw, axis=0)
    feature_scale = np.std(raw, axis=0)
    keep = feature_scale > 1e-12
    return {
        "kind": kind,
        "coordinate_center": center,
        "coordinate_scale": scale,
        "knots": knots,
        "feature_center": feature_center[keep],
        "feature_scale": feature_scale[keep],
        "keep": keep,
    }


def _apply_transform(transform, x):
    z = (x - transform["coordinate_center"]) / transform["coordinate_scale"]
    raw = _raw_features(z, transform["kind"], transform["knots"])
    raw = raw[:, transform["keep"]]
    return (raw - transform["feature_center"]) / transform["feature_scale"]


def _raw_features(z, kind, knots):
    x1, x2 = z[:, 0], z[:, 1]
    if kind == "axis_1_linear":
        return x1[:, None]
    if kind == "two_axis_linear":
        return z
    if kind == "two_axis_polynomial":
        return np.column_stack((x1, x2, x1**2, x1 * x2, x2**2, x1**3, x2**3))
    if kind == "additive_cubic_spline":
        columns = []
        for values, axis_knots in ((x1, knots[0]), (x2, knots[1])):
            columns.extend((values, values**2, values**3))
            columns.extend(np.maximum(values - knot, 0.0) ** 3 for knot in axis_knots)
        return np.column_stack(columns)
    raise ValueError(f"unknown feature model: {kind}")


def _ridge(x, y, penalty):
    label_mean = np.mean(y, axis=0)
    coefficient = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ (y - label_mean))
    return label_mean, coefficient


def _target_metrics(observed, predicted, baseline_mean):
    return {
        name: asdict(_metrics(observed[:, index], predicted[:, index], float(baseline_mean[index])))
        for index, name in enumerate(TARGET_NAMES)
    }


def _metrics(observed, predicted, baseline_mean):
    baseline = np.full_like(observed, baseline_mean)
    baseline_mae = float(np.mean(np.abs(observed - baseline)))
    denominator = float(np.sum((observed - baseline_mean) ** 2))
    return PredictionMetrics(
        item_count=len(observed),
        spearman=_spearman(observed, predicted),
        pearson=float(np.corrcoef(observed, predicted)[0, 1]),
        r_squared=float(1.0 - np.sum((observed - predicted) ** 2) / denominator),
        mean_absolute_error=float(np.mean(np.abs(observed - predicted))),
        baseline_mean_absolute_error=baseline_mae,
        relative_mae_improvement=float(1.0 - np.mean(np.abs(observed - predicted)) / baseline_mae),
    )


def _macro_metric(observed, predicted, baseline_mean, metric):
    values = [asdict(_metrics(observed[:, index], predicted[:, index], float(baseline_mean[index])))[metric] for index in range(observed.shape[1])]
    return float(np.mean(values))


def _cross_embedding_alignment(reference, candidate, development, confirmation):
    ref_mean = reference[development].mean(axis=0)
    cand_mean = candidate[development].mean(axis=0)
    ref_scale = reference[development].std(axis=0)
    cand_scale = candidate[development].std(axis=0)
    ref_dev = (reference[development] - ref_mean) / ref_scale
    cand_dev = (candidate[development] - cand_mean) / cand_scale
    left, _, right = np.linalg.svd(cand_dev.T @ ref_dev, full_matrices=False)
    rotation = left @ right
    ref_confirm = (reference[confirmation] - ref_mean) / ref_scale
    cand_confirm = (candidate[confirmation] - cand_mean) / cand_scale @ rotation
    return {
        "fit_split": "development",
        "evaluation_split": "confirmation",
        "reference_development_mean": ref_mean.tolist(),
        "reference_development_scale": ref_scale.tolist(),
        "candidate_development_mean": cand_mean.tolist(),
        "candidate_development_scale": cand_scale.tolist(),
        "orthogonal_rotation": rotation.tolist(),
        "confirmation_axis_1": _agreement(ref_confirm[:, 0], cand_confirm[:, 0]),
        "confirmation_axis_2": _agreement(ref_confirm[:, 1], cand_confirm[:, 1]),
        "confirmation_flattened": _agreement(ref_confirm.ravel(), cand_confirm.ravel()),
        "reference_confirmation": ref_confirm,
        "candidate_confirmation_aligned": cand_confirm,
    }


def _agreement(left, right):
    return {
        "item_count": len(left),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": _spearman(left, right),
        "mean_absolute_difference": float(np.mean(np.abs(left - right))),
    }


def _spearman(left, right):
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(_ranks(np.asarray(left)), _ranks(np.asarray(right)))[0, 1])


def _axis_1_monotonic_fraction(model):
    transform = model["transform"]
    axis_1_z = np.linspace(-2.0, 2.0, 41)
    axis_1 = (
        transform["coordinate_center"][0]
        + axis_1_z * transform["coordinate_scale"][0]
    )
    fractions = []
    for axis_2_z in np.linspace(-1.5, 1.5, 7):
        axis_2 = (
            transform["coordinate_center"][1]
            + axis_2_z * transform["coordinate_scale"][1]
        )
        predicted = _predict_model(
            model,
            np.column_stack((axis_1, np.full_like(axis_1, axis_2))),
        )[:, 0]
        differences = np.diff(predicted)
        fractions.append(
            max(
                float(np.mean(differences >= -1e-6)),
                float(np.mean(differences <= 1e-6)),
            )
        )
    return float(np.mean(fractions))


def _leave_one_source_out(x_dev, y_dev, ids_dev, sources_dev, x_confirm, y_confirm, sources_confirm, *, penalty, minimum_items):
    rows = {}
    for source in sorted(set(sources_dev) & set(sources_confirm)):
        train = sources_dev != source
        test = sources_confirm == source
        if (~train).sum() < minimum_items or test.sum() < minimum_items:
            continue
        model = _fit_model(x_dev[train], y_dev[train], ids_dev[train], kind="two_axis_linear", fixed_penalty=penalty)
        predicted = _predict_model(model, x_confirm[test])
        rows[str(source)] = asdict(_metrics(y_confirm[test, 0], predicted[:, 0], float(model["label_mean"][0])))
    positive_fraction = float(np.mean([row["spearman"] > 0 for row in rows.values()])) if rows else 0.0
    return {"minimum_items_per_split": minimum_items, "source_count": len(rows), "positive_spearman_fraction": positive_fraction, "sources": rows}


def _bootstrap_evidence(*, y_confirm, scalar_reference, scalar_candidate, reference_predictions, candidate_predictions, reference_label_mean, candidate_label_mean, aligned_reference, aligned_candidate, replicates, random_seed):
    rng = np.random.default_rng(random_seed)
    values = {name: [] for name in ("reference_scalar_spearman", "candidate_scalar_spearman", "cross_scalar_spearman", "aligned_axis_1_spearman", "aligned_axis_2_spearman", "reference_axis_2_r2_gain", "candidate_axis_2_r2_gain", "reference_spline_r2_gain", "candidate_spline_r2_gain")}
    for _ in range(replicates):
        index = rng.integers(0, len(y_confirm), len(y_confirm))
        values["reference_scalar_spearman"].append(_spearman(y_confirm[index, 0], scalar_reference[index]))
        values["candidate_scalar_spearman"].append(_spearman(y_confirm[index, 0], scalar_candidate[index]))
        values["cross_scalar_spearman"].append(_spearman(scalar_reference[index], scalar_candidate[index]))
        values["aligned_axis_1_spearman"].append(_spearman(aligned_reference[index, 0], aligned_candidate[index, 0]))
        values["aligned_axis_2_spearman"].append(_spearman(aligned_reference[index, 1], aligned_candidate[index, 1]))
        for prefix, prediction, label_mean in (("reference", reference_predictions, reference_label_mean), ("candidate", candidate_predictions, candidate_label_mean)):
            one = _macro_metric(y_confirm[index], prediction["axis_1_linear"][index], label_mean, "r_squared")
            two = _macro_metric(y_confirm[index], prediction["two_axis_linear"][index], label_mean, "r_squared")
            spline = _macro_metric(y_confirm[index], prediction["additive_cubic_spline"][index], label_mean, "r_squared")
            values[f"{prefix}_axis_2_r2_gain"].append(two - one)
            values[f"{prefix}_spline_r2_gain"].append(spline - two)
    return {name: _interval(rows) for name, rows in values.items()}


def _interval(values):
    array = np.asarray(values)
    return {"median": float(np.median(array)), "lower_95": float(np.quantile(array, 0.025)), "upper_95": float(np.quantile(array, 0.975))}


def _permutation_control(x_dev, y_dev, ids_dev, x_confirm, y_confirm, *, observed, penalty, replicates, random_seed):
    observed_spearman = _spearman(y_confirm[:, 0], observed)
    rng = np.random.default_rng(random_seed)
    null = []
    for _ in range(replicates):
        permuted = y_dev[rng.permutation(len(y_dev))]
        model = _fit_model(x_dev, permuted, ids_dev, kind="two_axis_linear", fixed_penalty=penalty)
        null.append(_spearman(y_confirm[:, 0], _predict_model(model, x_confirm)[:, 0]))
    return {"replicates": replicates, "observed_spearman": observed_spearman, "null_95th_percentile": float(np.quantile(null, 0.95)), "one_sided_p_value": float((1 + sum(value >= observed_spearman for value in null)) / (replicates + 1))}


def _evidence_checks(*, reference_manifest, candidate_manifest, representation_results, alignment, bootstrap, source_transfer, permutation):
    checks = {
        "both_frozen_maps_supportive": reference_manifest["evidence_assessment"]["status"] == "supportive" and candidate_manifest["evidence_assessment"]["status"] == "supportive",
        "reference_scalar_bootstrap_lower_at_least_0_60": bootstrap["reference_scalar_spearman"]["lower_95"] >= 0.60,
        "candidate_scalar_bootstrap_lower_at_least_0_60": bootstrap["candidate_scalar_spearman"]["lower_95"] >= 0.60,
        "cross_scalar_bootstrap_lower_at_least_0_75": bootstrap["cross_scalar_spearman"]["lower_95"] >= 0.75,
        "aligned_axis_1_bootstrap_lower_at_least_0_70": bootstrap["aligned_axis_1_spearman"]["lower_95"] >= 0.70,
        "aligned_axis_2_bootstrap_lower_at_least_0_50": bootstrap["aligned_axis_2_spearman"]["lower_95"] >= 0.50,
        "axis_2_positive_macro_r2_gain_both_views": bootstrap["reference_axis_2_r2_gain"]["lower_95"] > 0 and bootstrap["candidate_axis_2_r2_gain"]["lower_95"] > 0,
        "spline_not_materially_worse_both_views": bootstrap["reference_spline_r2_gain"]["lower_95"] >= -0.03 and bootstrap["candidate_spline_r2_gain"]["lower_95"] >= -0.03,
        "axis_1_spline_monotonic_at_least_0_75_both_views": representation_results["reference"]["incremental_tests"]["axis_1_spline_monotonic_fraction"] >= 0.75 and representation_results["candidate"]["incremental_tests"]["axis_1_spline_monotonic_fraction"] >= 0.75,
        "leave_one_source_out_positive_in_70_percent_both_views": source_transfer["reference"]["positive_spearman_fraction"] >= 0.70 and source_transfer["candidate"]["positive_spearman_fraction"] >= 0.70,
        "permutation_control_p_at_most_0_05_both_views": permutation["reference"]["one_sided_p_value"] <= 0.05 and permutation["candidate"]["one_sided_p_value"] <= 0.05,
    }
    return checks


def _assessment(checks):
    passed = sum(checks.values())
    core = all(value for name, value in checks.items() if name not in {"axis_2_positive_macro_r2_gain_both_views", "spline_not_materially_worse_both_views", "axis_1_spline_monotonic_at_least_0_75_both_views"})
    status = "strongly-supportive" if core and passed == len(checks) else "supportive" if core and passed >= len(checks) - 2 else "inconclusive"
    return {"status": status, "passed_check_count": passed, "total_check_count": len(checks), "all_checks_passed": passed == len(checks)}


def _validate_consensus_artifact(root, manifest):
    name = "readiness_consensus.jsonl"
    identity = manifest.get("artifacts", {}).get(name)
    if not isinstance(identity, Mapping) or sha256_file(root / name) != identity.get("sha256"):
        raise ValueError(f"consensus artifact hash mismatch: {root / name}")


def _identity(root, manifest):
    return {"path": str(root), "map_id": manifest["map_id"], "embedding_model": manifest["embedding_model"], "embedding_dimension": manifest["embedding_dimension"]}


def _render_report(payload):
    lines = [
        "# Two-embedding readiness robustness battery",
        "",
        f"Assessment: **{payload['assessment']['status'].upper()}** "
        f"({payload['assessment']['passed_check_count']}/{payload['assessment']['total_check_count']} checks)",
        "",
        "## Frozen confirmation sample",
        "",
        f"- Development usable items: {payload['sample']['development_usable_items']}",
        f"- Confirmation usable items: {payload['sample']['confirmation_usable_items']}",
        "",
        "## Direct frozen-map performance",
        "",
    ]
    for name in ("reference", "candidate"):
        metrics = payload["direct_scalar_map"][name]
        lines.append(f"- {name}: Spearman={metrics['spearman']:.4f}, R²={metrics['r_squared']:.4f}, MAE={metrics['mean_absolute_error']:.4f}")
    cross = payload["direct_scalar_map"]["cross_embedding_confirmation"]
    lines.extend(["", f"- Cross-embedding scalar Spearman: {cross['spearman']:.4f}", "", "## Nonlinear and two-axis tests", ""])
    for name in ("reference", "candidate"):
        tests = payload["representation_models"][name]["incremental_tests"]
        lines.append(f"- {name}: axis-2 macro R² gain={tests['axis_2_macro_r_squared_gain']:+.4f}; spline gain={tests['spline_macro_r_squared_gain_over_linear']:+.4f}; polynomial gain={tests['polynomial_macro_r_squared_gain_over_linear']:+.4f}; spline monotonicity={tests['axis_1_spline_monotonic_fraction']:.2%}")
    lines.extend(["", "## Cross-embedding aligned axes", ""])
    for axis in ("confirmation_axis_1", "confirmation_axis_2"):
        metrics = payload["cross_embedding_alignment"][axis]
        lines.append(f"- {axis}: Spearman={metrics['spearman']:.4f}, Pearson={metrics['pearson']:.4f}")
    lines.extend(["", "## Frozen checks", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in payload["evidence_checks"].items())
    lines.extend(["", payload["interpretation_guard"], ""])
    return "\n".join(lines)
