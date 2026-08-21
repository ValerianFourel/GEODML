"""Development-fit factor stress test for frozen readiness rubrics and axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

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


READINESS_FACTOR_STRESS_VERSION = "readiness-factor-stress-v1"
RUBRIC_NAMES = (
    "information_seeking_reverse",
    "evaluation",
    "selection_commitment",
    "action_implementation",
)


@dataclass(frozen=True, slots=True)
class AxisMetrics:
    item_count: int
    pearson: float
    spearman: float
    r_squared: float
    mean_absolute_error: float


def run_readiness_factor_stress(
    *,
    dataset_dir: str | Path,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    git_commit_sha: str,
    parallel_replicates: int = 1000,
    bootstrap_replicates: int = 1000,
    random_seed: int = 20260822,
) -> dict[str, object]:
    """Stress-test one- versus two-factor structure on publication-safe rows.

    Factor extraction uses only the four ordinal rubric consensus dimensions.
    Overall readiness and frozen embedding axes are external validation targets.
    All fitted transforms use development rows; confirmation rows remain held out.
    """

    if parallel_replicates < 100 or bootstrap_replicates < 100:
        raise ValueError("factor stress requires at least 100 replicates")
    if not git_commit_sha.strip():
        raise ValueError("git_commit_sha must be nonempty")

    dataset = Path(dataset_dir).resolve()
    reference = Path(reference_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    output = Path(output_dir).resolve()
    if reference == candidate:
        raise ValueError("reference and candidate directories must differ")
    if output.exists():
        raise ValueError(f"refusing to overwrite factor stress output: {output}")

    safe_ids, dataset_manifest = _publication_safe_item_ids(dataset)
    reference_manifest = _load_subspace_manifest(reference)
    candidate_manifest = _load_subspace_manifest(candidate)
    frozen_design = _validate_frozen_design(reference_manifest, candidate_manifest)
    _validate_consensus(reference, reference_manifest)
    _validate_consensus(candidate, candidate_manifest)
    reference_consensus = read_jsonl(reference / "readiness_consensus.jsonl")
    candidate_consensus = read_jsonl(candidate / "readiness_consensus.jsonl")
    if reference_consensus != candidate_consensus:
        raise ValueError("reference and candidate consensus artifacts differ")

    data = _aligned_safe_data(
        safe_ids=safe_ids,
        consensus_rows=reference_consensus,
        reference_dir=reference,
        candidate_dir=candidate,
    )
    development = data["split"] == "development"
    confirmation = data["split"] == "confirmation"
    if development.sum() < 50 or confirmation.sum() < 50:
        raise ValueError(
            "factor stress requires at least 50 usable safe rows per split"
        )

    rubric_dev = data["rubrics"][development]
    rubric_confirm = data["rubrics"][confirmation]
    center = rubric_dev.mean(axis=0)
    scale = rubric_dev.std(axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("development rubric variance is degenerate")
    z_dev = (rubric_dev - center) / scale
    z_confirm = (rubric_confirm - center) / scale
    correlation_dev = _spearman_correlation(rubric_dev)
    correlation_confirm = _spearman_correlation(rubric_confirm)

    parallel = _parallel_analysis(
        rubric_dev,
        replicates=parallel_replicates,
        random_seed=random_seed,
    )
    one_factor = _fit_factor_model(correlation_dev, factor_count=1)
    two_factor = _fit_factor_model(correlation_dev, factor_count=2)
    scores = {
        "one": _factor_scores(
            z_dev, z_confirm, correlation_dev, one_factor["loadings"]
        ),
        "two": _factor_scores(
            z_dev, z_confirm, correlation_dev, two_factor["loadings"]
        ),
    }
    factor_models = {
        "one_factor": _factor_model_evidence(
            one_factor,
            correlation_dev=correlation_dev,
            correlation_confirm=correlation_confirm,
        ),
        "two_factor": _factor_model_evidence(
            two_factor,
            correlation_dev=correlation_dev,
            correlation_confirm=correlation_confirm,
        ),
    }
    confirmation_replication = _confirmation_replication(
        correlation_confirm,
        one_factor=one_factor,
        two_factor=two_factor,
    )
    bootstrap = _bootstrap_loadings(
        rubric_dev,
        reference_loadings=two_factor["loadings"],
        replicates=bootstrap_replicates,
        random_seed=random_seed + 17,
    )
    overall_validation = {
        name: _score_target_associations(
            value["confirmation"],
            data["overall_readiness"][confirmation],
        )
        for name, value in scores.items()
    }
    axis_association = {
        name: {
            factor_name: _factor_to_axes(
                factor_scores["development"],
                factor_scores["confirmation"],
                axes[development],
                axes[confirmation],
            )
            for factor_name, factor_scores in scores.items()
        }
        for name, axes in (
            ("reference", data["reference_axes"]),
            ("candidate", data["candidate_axes"]),
        )
    }
    checks = _checks(
        parallel=parallel,
        factor_models=factor_models,
        confirmation_replication=confirmation_replication,
        bootstrap=bootstrap,
        axis_association=axis_association,
    )
    assessment = _assessment(checks)
    payload = {
        "format_version": READINESS_FACTOR_STRESS_VERSION,
        "git_commit_sha": git_commit_sha,
        "dataset": {
            "path": str(dataset),
            "included_prompt_count": int(dataset_manifest["included_prompt_count"]),
            "manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
            "publication_safe": True,
        },
        "reference": _map_identity(reference, reference_manifest),
        "candidate": _map_identity(candidate, candidate_manifest),
        "frozen_design": frozen_design,
        "sample": {
            "publication_safe_item_count": len(safe_ids),
            "usable_development_items": int(development.sum()),
            "usable_confirmation_items": int(confirmation.sum()),
            "rubric_names": list(RUBRIC_NAMES),
        },
        "method": {
            "fit_split": "development",
            "evaluation_split": "confirmation",
            "correlation": "spearman",
            "extraction": "iterated-principal-axis",
            "two_factor_rotation": "varimax",
            "overall_readiness_role": "external-validation-only",
            "identification_limit": (
                "With four indicators, an unconstrained two-factor common-factor "
                "model is not a confirmatory identified measurement model. The "
                "two-factor loadings are descriptive stress-test diagnostics."
            ),
            "parallel_replicates": parallel_replicates,
            "bootstrap_replicates": bootstrap_replicates,
            "random_seed": random_seed,
        },
        "rubric_correlation": {
            "development": correlation_dev.tolist(),
            "confirmation": correlation_confirm.tolist(),
        },
        "parallel_analysis": parallel,
        "factor_models": factor_models,
        "confirmation_replication": confirmation_replication,
        "bootstrap_loading_stability": bootstrap,
        "overall_readiness_external_validation": overall_validation,
        "frozen_axis_association": axis_association,
        "evidence_checks": checks,
        "assessment": assessment,
        "interpretation_guard": (
            "This exploratory factor analysis probes the dimensionality of frozen "
            "rubric labels on publication-safe rows. Four indicators cannot identify "
            "an unconstrained confirmatory two-factor measurement model, so the "
            "loadings are descriptive. This analysis does not define the frozen "
            "embedding axes, does not define or replace policy variable B, and is "
            "not causal evidence."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_json(temporary / "readiness_factor_stress.json", payload)
        atomic_text(
            temporary / "readiness_factor_stress_report.md",
            _render_report(payload),
        )
        artifacts = {
            name: {
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in (
                "readiness_factor_stress.json",
                "readiness_factor_stress_report.md",
            )
        }
        manifest = {
            "format_version": READINESS_FACTOR_STRESS_VERSION,
            "git_commit_sha": git_commit_sha,
            "reference_map_id": reference_manifest["map_id"],
            "candidate_map_id": candidate_manifest["map_id"],
            "parallel_replicates": parallel_replicates,
            "bootstrap_replicates": bootstrap_replicates,
            "random_seed": random_seed,
            "assessment": assessment,
            "artifacts": artifacts,
        }
        atomic_json(temporary / "factor_stress_manifest.json", manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _publication_safe_item_ids(root: Path):
    manifest = read_json(root / "dataset_manifest.json")
    if not manifest.get("publication_safe"):
        raise ValueError("dataset is not marked publication-safe")
    expected_checksums = read_json(root / "checksums.json")
    actual_checksums = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    if expected_checksums != actual_checksums:
        raise ValueError("finalized dataset checksum verification failed")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "factor stress requires pyarrow to read the finalized HF dataset"
        ) from exc
    paths = sorted((root / "data" / "prompts").glob("*.parquet"))
    if not paths:
        raise ValueError("finalized dataset has no prompt Parquet shards")
    item_ids = []
    for path in paths:
        table = pq.read_table(path, columns=["item_id"])
        item_ids.extend(str(value) for value in table["item_id"].to_pylist())
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("finalized dataset contains duplicate prompt item IDs")
    if len(item_ids) != int(manifest["included_prompt_count"]):
        raise ValueError("finalized dataset prompt count differs from manifest")
    return frozenset(item_ids), manifest


def _validate_consensus(root: Path, manifest: Mapping[str, object]) -> None:
    name = "readiness_consensus.jsonl"
    identity = manifest.get("artifacts", {}).get(name)
    if not isinstance(identity, Mapping) or sha256_file(root / name) != identity.get(
        "sha256"
    ):
        raise ValueError(f"consensus artifact hash mismatch: {root / name}")


def _aligned_safe_data(*, safe_ids, consensus_rows, reference_dir, candidate_dir):
    labels = {
        str(row["item_id"]): row
        for row in consensus_rows
        if bool(row["usable_for_axis"]) and str(row["item_id"]) in safe_ids
    }
    reference = _axis_index(reference_dir)
    candidate = _axis_index(candidate_dir)
    if set(reference) != set(candidate):
        raise ValueError("supervised coordinate identities differ")
    keys = sorted(key for key in reference if key[1] in labels)
    rubrics = []
    overall = []
    for _, item_id in keys:
        row = labels[item_id]
        rubrics.append(
            (
                (7.0 - float(row["information_seeking_1_7"])) / 6.0,
                (float(row["evaluation_1_7"]) - 1.0) / 6.0,
                (float(row["selection_commitment_1_7"]) - 1.0) / 6.0,
                (float(row["action_implementation_1_7"]) - 1.0) / 6.0,
            )
        )
        overall.append(float(row["overall_readiness_0_100"]) / 100.0)
    return {
        "split": np.asarray([key[0] for key in keys]),
        "item_ids": np.asarray([key[1] for key in keys]),
        "source": np.asarray([reference[key][2] for key in keys]),
        "rubrics": np.asarray(rubrics, dtype=np.float64),
        "overall_readiness": np.asarray(overall, dtype=np.float64),
        "reference_axes": np.asarray([reference[key][:2] for key in keys]),
        "candidate_axes": np.asarray([candidate[key][:2] for key in keys]),
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


def _spearman_correlation(matrix):
    ranked = np.column_stack(
        [_ranks(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    return _nearest_correlation(np.corrcoef(ranked, rowvar=False))


def _nearest_correlation(matrix):
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    positive = vectors @ np.diag(np.maximum(values, 1e-8)) @ vectors.T
    scale = np.sqrt(np.diag(positive))
    result = positive / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)
    return result


def _parallel_analysis(matrix, *, replicates, random_seed):
    observed = np.linalg.eigvalsh(_spearman_correlation(matrix))[::-1]
    rng = np.random.default_rng(random_seed)
    null = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    for replicate in range(replicates):
        permuted = np.column_stack(
            [rng.permutation(matrix[:, column]) for column in range(matrix.shape[1])]
        )
        null[replicate] = np.linalg.eigvalsh(_spearman_correlation(permuted))[::-1]
    threshold = np.quantile(null, 0.95, axis=0)
    return {
        "observed_eigenvalues": observed.tolist(),
        "null_95th_percentile": threshold.tolist(),
        "retained_factor_count": int(np.sum(observed > threshold)),
    }


def _fit_factor_model(correlation, *, factor_count):
    inverse = np.linalg.pinv(correlation)
    communalities = np.clip(1.0 - 1.0 / np.diag(inverse), 0.05, 0.99)
    converged = False
    iterations = 0
    for iterations in range(1, 501):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communalities)
        values, vectors = np.linalg.eigh(reduced)
        order = np.argsort(values)[::-1][:factor_count]
        selected = np.maximum(values[order], 0.0)
        loadings = vectors[:, order] * np.sqrt(selected)
        updated = np.clip(np.sum(loadings**2, axis=1), 0.0, 0.999)
        if np.max(np.abs(updated - communalities)) < 1e-8:
            communalities = updated
            converged = True
            break
        communalities = updated
    if factor_count > 1:
        loadings = _varimax(loadings)
    loadings = _canonical_loadings(loadings)
    communalities = np.sum(loadings**2, axis=1)
    return {
        "factor_count": factor_count,
        "loadings": loadings,
        "communalities": communalities,
        "uniqueness": 1.0 - communalities,
        "iterations": iterations,
        "converged": converged,
    }


def _varimax(loadings, *, maximum_iterations=500, tolerance=1e-8):
    row_count, factor_count = loadings.shape
    rotation = np.eye(factor_count)
    previous = 0.0
    for _ in range(maximum_iterations):
        rotated = loadings @ rotation
        left, singular, right = np.linalg.svd(
            loadings.T
            @ (
                rotated**3
                - (rotated @ np.diag(np.sum(rotated**2, axis=0))) / row_count
            ),
            full_matrices=False,
        )
        rotation = left @ right
        objective = float(np.sum(singular))
        if previous and objective - previous < tolerance:
            break
        previous = objective
    return loadings @ rotation


def _canonical_loadings(loadings):
    result = np.asarray(loadings, dtype=np.float64).copy()
    order = np.argsort(np.sum(result**2, axis=0))[::-1]
    result = result[:, order]
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def _factor_scores(z_dev, z_confirm, correlation, loadings):
    weights = np.linalg.pinv(correlation) @ loadings
    raw_dev = z_dev @ weights
    raw_confirm = z_confirm @ weights
    center = raw_dev.mean(axis=0)
    scale = raw_dev.std(axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("factor score variance is degenerate")
    return {
        "development": (raw_dev - center) / scale,
        "confirmation": (raw_confirm - center) / scale,
    }


def _factor_model_evidence(model, *, correlation_dev, correlation_confirm):
    implied = model["loadings"] @ model["loadings"].T
    return {
        "loadings": {
            rubric: [float(value) for value in model["loadings"][index]]
            for index, rubric in enumerate(RUBRIC_NAMES)
        },
        "communalities": {
            rubric: float(model["communalities"][index])
            for index, rubric in enumerate(RUBRIC_NAMES)
        },
        "uniqueness": {
            rubric: float(model["uniqueness"][index])
            for index, rubric in enumerate(RUBRIC_NAMES)
        },
        "converged": bool(model["converged"]),
        "iterations": int(model["iterations"]),
        "development_off_diagonal_rmse": _off_diagonal_rmse(correlation_dev, implied),
        "confirmation_off_diagonal_rmse": _off_diagonal_rmse(
            correlation_confirm, implied
        ),
    }


def _off_diagonal_rmse(observed, implied):
    mask = ~np.eye(observed.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((observed[mask] - implied[mask]) ** 2)))


def _confirmation_replication(correlation, *, one_factor, two_factor):
    independent_one = _fit_factor_model(correlation, factor_count=1)
    independent_two = _fit_factor_model(correlation, factor_count=2)
    aligned_one = _align_loadings(independent_one["loadings"], one_factor["loadings"])
    aligned_two = _align_loadings(independent_two["loadings"], two_factor["loadings"])
    return {
        "one_factor_congruence": _column_congruence(
            one_factor["loadings"], aligned_one
        ),
        "two_factor_congruence": _column_congruence(
            two_factor["loadings"], aligned_two
        ),
        "aligned_confirmation_two_factor_loadings": {
            rubric: [float(value) for value in aligned_two[index]]
            for index, rubric in enumerate(RUBRIC_NAMES)
        },
    }


def _align_loadings(candidate, reference):
    left, _, right = np.linalg.svd(candidate.T @ reference, full_matrices=False)
    return candidate @ (left @ right)


def _column_congruence(reference, candidate):
    return [
        float(
            np.dot(reference[:, column], candidate[:, column])
            / (
                np.linalg.norm(reference[:, column])
                * np.linalg.norm(candidate[:, column])
            )
        )
        for column in range(reference.shape[1])
    ]


def _bootstrap_loadings(matrix, *, reference_loadings, replicates, random_seed):
    rng = np.random.default_rng(random_seed)
    collected = []
    congruence = []
    for _ in range(replicates):
        index = rng.integers(0, len(matrix), len(matrix))
        fitted = _fit_factor_model(
            _spearman_correlation(matrix[index]),
            factor_count=2,
        )
        aligned = _align_loadings(fitted["loadings"], reference_loadings)
        collected.append(aligned)
        congruence.append(_column_congruence(reference_loadings, aligned))
    loadings = np.asarray(collected)
    congruence_array = np.asarray(congruence)
    return {
        "loading_intervals": {
            rubric: {
                f"factor_{factor + 1}": {
                    "lower_95": float(
                        np.quantile(loadings[:, rubric_index, factor], 0.025)
                    ),
                    "median": float(np.median(loadings[:, rubric_index, factor])),
                    "upper_95": float(
                        np.quantile(loadings[:, rubric_index, factor], 0.975)
                    ),
                }
                for factor in range(2)
            }
            for rubric_index, rubric in enumerate(RUBRIC_NAMES)
        },
        "factor_congruence": {
            f"factor_{factor + 1}": {
                "lower_95": float(np.quantile(congruence_array[:, factor], 0.025)),
                "median": float(np.median(congruence_array[:, factor])),
                "upper_95": float(np.quantile(congruence_array[:, factor], 0.975)),
            }
            for factor in range(2)
        },
    }


def _score_target_associations(scores, target):
    return {
        f"factor_{column + 1}": {
            "pearson": float(np.corrcoef(scores[:, column], target)[0, 1]),
            "spearman": _spearman(scores[:, column], target),
        }
        for column in range(scores.shape[1])
    }


def _factor_to_axes(scores_dev, scores_confirm, axes_dev, axes_confirm):
    design_dev = np.column_stack((np.ones(len(scores_dev)), scores_dev))
    coefficient = np.linalg.solve(
        design_dev.T @ design_dev + 1e-6 * np.eye(design_dev.shape[1]),
        design_dev.T @ axes_dev,
    )
    prediction = (
        np.column_stack((np.ones(len(scores_confirm)), scores_confirm)) @ coefficient
    )
    baseline = axes_dev.mean(axis=0)
    metrics = {
        f"axis_{column + 1}": asdict(
            _axis_metrics(
                axes_confirm[:, column], prediction[:, column], baseline[column]
            )
        )
        for column in range(2)
    }
    metrics["macro_r_squared"] = float(
        np.mean([metrics["axis_1"]["r_squared"], metrics["axis_2"]["r_squared"]])
    )
    return metrics


def _axis_metrics(observed, predicted, baseline):
    denominator = float(np.sum((observed - baseline) ** 2))
    return AxisMetrics(
        item_count=len(observed),
        pearson=float(np.corrcoef(observed, predicted)[0, 1]),
        spearman=_spearman(observed, predicted),
        r_squared=float(1.0 - np.sum((observed - predicted) ** 2) / denominator),
        mean_absolute_error=float(np.mean(np.abs(observed - predicted))),
    )


def _spearman(left, right):
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(_ranks(np.asarray(left)), _ranks(np.asarray(right)))[0, 1])


def _checks(
    *, parallel, factor_models, confirmation_replication, bootstrap, axis_association
):
    one_rmse = factor_models["one_factor"]["confirmation_off_diagonal_rmse"]
    two_rmse = factor_models["two_factor"]["confirmation_off_diagonal_rmse"]
    return {
        "parallel_analysis_retains_one_or_two_factors": 1
        <= parallel["retained_factor_count"]
        <= 2,
        "two_factor_confirmation_reconstruction_improves": two_rmse < one_rmse,
        "confirmation_factor_1_congruence_at_least_0_85": confirmation_replication[
            "two_factor_congruence"
        ][0]
        >= 0.85,
        "confirmation_factor_2_congruence_at_least_0_70": confirmation_replication[
            "two_factor_congruence"
        ][1]
        >= 0.70,
        "bootstrap_factor_1_lower_congruence_at_least_0_85": bootstrap[
            "factor_congruence"
        ]["factor_1"]["lower_95"]
        >= 0.85,
        "bootstrap_factor_2_lower_congruence_at_least_0_70": bootstrap[
            "factor_congruence"
        ]["factor_2"]["lower_95"]
        >= 0.70,
        "two_factors_improve_axis_association_both_views": all(
            models["two"]["macro_r_squared"] > models["one"]["macro_r_squared"]
            for models in axis_association.values()
        ),
    }


def _assessment(checks):
    passed = sum(checks.values())
    status = (
        "descriptively-supportive"
        if passed == len(checks)
        else "partially-descriptively-supportive"
        if passed >= len(checks) - 2
        else "inconclusive"
    )
    return {
        "status": status,
        "passed_check_count": passed,
        "total_check_count": len(checks),
        "all_checks_passed": passed == len(checks),
    }


def _map_identity(root, manifest):
    return {
        "path": str(root),
        "map_id": manifest["map_id"],
        "embedding_model": manifest["embedding_model"],
        "embedding_dimension": manifest["embedding_dimension"],
    }


def _render_report(payload):
    one_rmse = payload["factor_models"]["one_factor"][
        "confirmation_off_diagonal_rmse"
    ]
    two_rmse = payload["factor_models"]["two_factor"][
        "confirmation_off_diagonal_rmse"
    ]
    lines = [
        "# Readiness factor stress test",
        "",
        f"Assessment: **{payload['assessment']['status'].upper()}** "
        f"({payload['assessment']['passed_check_count']}/"
        f"{payload['assessment']['total_check_count']} checks)",
        "",
        "## Publication-safe frozen sample",
        "",
        f"- Development usable items: {payload['sample']['usable_development_items']}",
        "- Confirmation usable items: "
        f"{payload['sample']['usable_confirmation_items']}",
        "",
        "## Factor count and confirmation reconstruction",
        "",
        f"- Identification note: {payload['method']['identification_limit']}",
        "- Parallel-analysis retained factors: "
        f"{payload['parallel_analysis']['retained_factor_count']}",
        f"- One-factor confirmation off-diagonal RMSE: {one_rmse:.4f}",
        f"- Two-factor confirmation off-diagonal RMSE: {two_rmse:.4f}",
        "",
        "## Two-factor development loadings",
        "",
    ]
    for rubric, values in payload["factor_models"]["two_factor"]["loadings"].items():
        lines.append(
            f"- {rubric}: factor 1={values[0]:+.4f}; factor 2={values[1]:+.4f}"
        )
    lines.extend(["", "## Confirmation replication", ""])
    congruence = payload["confirmation_replication"]["two_factor_congruence"]
    lines.append(f"- Factor 1 congruence: {congruence[0]:.4f}")
    lines.append(f"- Factor 2 congruence: {congruence[1]:.4f}")
    lines.extend(["", "## Frozen-axis association", ""])
    for name, models in payload["frozen_axis_association"].items():
        gain = models["two"]["macro_r_squared"] - models["one"]["macro_r_squared"]
        lines.append(
            f"- {name}: one-factor macro R²={models['one']['macro_r_squared']:.4f}; "
            f"two-factor macro R²={models['two']['macro_r_squared']:.4f}; "
            f"gain={gain:+.4f}"
        )
    lines.extend(["", "## Exploratory checks", ""])
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in payload["evidence_checks"].items()
    )
    lines.extend(["", payload["interpretation_guard"], ""])
    return "\n".join(lines)
