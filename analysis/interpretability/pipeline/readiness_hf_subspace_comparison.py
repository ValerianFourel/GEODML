"""Cross-embedding confirmation for immutable semantic-readiness subspaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


READINESS_SUBSPACE_COMPARISON_VERSION = "readiness-subspace-comparison-v1"


@dataclass(frozen=True, slots=True)
class CoordinateAgreement:
    item_count: int
    pearson: float
    spearman: float
    mean_absolute_difference: float


def compare_readiness_hf_subspaces(
    *,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    git_commit_sha: str,
) -> dict[str, object]:
    """Compare two frozen maps without assuming shared embedding dimensions."""

    reference = Path(reference_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    output = Path(output_dir).resolve()
    if not git_commit_sha.strip():
        raise ValueError("git_commit_sha must be nonempty")
    if reference == candidate:
        raise ValueError("reference and candidate subspaces must differ")
    if output.exists():
        raise ValueError(f"refusing to overwrite comparison directory: {output}")

    reference_manifest = _load_subspace_manifest(reference)
    candidate_manifest = _load_subspace_manifest(candidate)
    frozen_design = _validate_frozen_design(
        reference_manifest,
        candidate_manifest,
    )
    scalar_agreement = _scalar_prediction_agreement(reference, candidate)
    subspace_alignment = _supervised_subspace_alignment(reference, candidate)
    reference_diagnostics = read_json(
        reference / "readiness_embedding_map_diagnostics.json"
    )
    candidate_diagnostics = read_json(
        candidate / "readiness_embedding_map_diagnostics.json"
    )

    payload = {
        "format_version": READINESS_SUBSPACE_COMPARISON_VERSION,
        "git_commit_sha": git_commit_sha,
        "reference": _subspace_identity(reference, reference_manifest),
        "candidate": _subspace_identity(candidate, candidate_manifest),
        "frozen_design": frozen_design,
        "scalar_prediction_agreement": {
            split: asdict(metrics) for split, metrics in scalar_agreement.items()
        },
        "supervised_subspace_alignment": subspace_alignment,
        "reference_holdout_evidence": reference_diagnostics["holdout_evidence"],
        "candidate_holdout_evidence": candidate_diagnostics["holdout_evidence"],
        "interpretation_guard": (
            "This is a representation-family robustness check on a frozen label "
            "target and split. It is not a causal result, and prompt embeddings do "
            "not define the experimental policy variable."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_json(temporary / "readiness_subspace_comparison.json", payload)
        atomic_text(
            temporary / "readiness_subspace_comparison_report.md",
            _render_report(payload),
        )
        artifacts = {
            name: {
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in (
                "readiness_subspace_comparison.json",
                "readiness_subspace_comparison_report.md",
            )
        }
        manifest = {
            "format_version": READINESS_SUBSPACE_COMPARISON_VERSION,
            "git_commit_sha": git_commit_sha,
            "reference_map_id": reference_manifest["map_id"],
            "candidate_map_id": candidate_manifest["map_id"],
            "prompt_sha256": frozen_design["prompt_sha256"],
            "annotation_sha256": frozen_design["annotation_sha256"],
            "artifacts": artifacts,
        }
        atomic_json(temporary / "comparison_manifest.json", manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _load_subspace_manifest(root: Path) -> Mapping[str, object]:
    path = root / "subspace_manifest.json"
    if not path.is_file():
        raise ValueError(f"missing subspace manifest: {path}")
    manifest = read_json(path)
    for artifact in (
        "readiness_embedding_coordinates.jsonl",
        "readiness_supervised_subspace_coordinates.jsonl",
        "readiness_embedding_map_diagnostics.json",
    ):
        identity = manifest.get("artifacts", {}).get(artifact)
        artifact_path = root / artifact
        if not artifact_path.is_file() or not isinstance(identity, Mapping):
            raise ValueError(f"subspace omits required artifact: {artifact}")
        if sha256_file(artifact_path) != identity.get("sha256"):
            raise ValueError(f"subspace artifact hash mismatch: {artifact_path}")
    return manifest


def _validate_frozen_design(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    reference_inputs = reference["inputs"]
    candidate_inputs = candidate["inputs"]
    prompt_sha = reference_inputs["prompts"]["sha256"]
    annotation_sha = reference_inputs["annotations"]["sha256"]
    checks = {
        "prompt_hash_equal": prompt_sha
        == candidate_inputs["prompts"]["sha256"],
        "annotation_hash_equal": annotation_sha
        == candidate_inputs["annotations"]["sha256"],
        "judge_slots_equal": reference["judge_slots"] == candidate["judge_slots"],
        "label_policy_equal": reference["label_policy"] == candidate["label_policy"],
        "split_policy_equal": reference["split_policy"] == candidate["split_policy"],
        "prompt_count_equal": reference["prompt_count"] == candidate["prompt_count"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"subspaces do not share a frozen design: {failed}")
    return {
        "checks": checks,
        "prompt_sha256": prompt_sha,
        "annotation_sha256": annotation_sha,
        "judge_slots": reference["judge_slots"],
        "label_policy": reference["label_policy"],
        "split_policy": reference["split_policy"],
        "prompt_count": reference["prompt_count"],
    }


def _scalar_prediction_agreement(
    reference: Path,
    candidate: Path,
) -> dict[str, CoordinateAgreement]:
    left = _coordinate_index(
        read_jsonl(reference / "readiness_embedding_coordinates.jsonl"),
        split_key="evaluation_split",
        coordinate_names=("observed_readiness_0_1",),
    )
    right = _coordinate_index(
        read_jsonl(candidate / "readiness_embedding_coordinates.jsonl"),
        split_key="evaluation_split",
        coordinate_names=("observed_readiness_0_1",),
    )
    if set(left) != set(right):
        raise ValueError("scalar prediction item/split identities differ")
    metrics = {}
    for split in ("development", "confirmation"):
        keys = sorted(key for key in left if key[0] == split)
        if len(keys) < 3:
            raise ValueError(f"too few shared scalar predictions for {split}")
        reference_values = np.asarray([left[key][0] for key in keys])
        candidate_values = np.asarray([right[key][0] for key in keys])
        metrics[split] = _agreement(reference_values, candidate_values)
    return metrics


def _supervised_subspace_alignment(
    reference: Path,
    candidate: Path,
) -> dict[str, object]:
    coordinate_names = ("axis_1", "axis_2")
    left = _coordinate_index(
        read_jsonl(reference / "readiness_supervised_subspace_coordinates.jsonl"),
        split_key="split",
        coordinate_names=coordinate_names,
    )
    right = _coordinate_index(
        read_jsonl(candidate / "readiness_supervised_subspace_coordinates.jsonl"),
        split_key="split",
        coordinate_names=coordinate_names,
    )
    if set(left) != set(right):
        raise ValueError("supervised coordinate item/split identities differ")

    development_keys = sorted(key for key in left if key[0] == "development")
    confirmation_keys = sorted(key for key in left if key[0] == "confirmation")
    if len(development_keys) < 3 or len(confirmation_keys) < 3:
        raise ValueError("subspace alignment needs both frozen splits")
    reference_development = np.asarray([left[key] for key in development_keys])
    candidate_development = np.asarray([right[key] for key in development_keys])
    reference_confirmation = np.asarray([left[key] for key in confirmation_keys])
    candidate_confirmation = np.asarray([right[key] for key in confirmation_keys])

    reference_mean = reference_development.mean(axis=0)
    candidate_mean = candidate_development.mean(axis=0)
    reference_scale = float(
        np.sqrt(np.mean((reference_development - reference_mean) ** 2))
    )
    candidate_scale = float(
        np.sqrt(np.mean((candidate_development - candidate_mean) ** 2))
    )
    if reference_scale <= 1e-12 or candidate_scale <= 1e-12:
        raise ValueError("subspace coordinate variance is degenerate")

    reference_dev_z = (reference_development - reference_mean) / reference_scale
    candidate_dev_z = (candidate_development - candidate_mean) / candidate_scale
    left_singular, _, right_singular = np.linalg.svd(
        candidate_dev_z.T @ reference_dev_z,
        full_matrices=False,
    )
    rotation = left_singular @ right_singular
    aligned_dev = candidate_dev_z @ rotation

    reference_confirm_z = (
        reference_confirmation - reference_mean
    ) / reference_scale
    candidate_confirm_z = (
        candidate_confirmation - candidate_mean
    ) / candidate_scale
    aligned_confirm = candidate_confirm_z @ rotation

    return {
        "alignment_fit_split": "development",
        "evaluation_split": "confirmation",
        "development_item_count": len(development_keys),
        "confirmation_item_count": len(confirmation_keys),
        "orthogonal_rotation": rotation.tolist(),
        "development_flattened": asdict(
            _agreement(reference_dev_z.ravel(), aligned_dev.ravel())
        ),
        "confirmation_flattened": asdict(
            _agreement(reference_confirm_z.ravel(), aligned_confirm.ravel())
        ),
        "confirmation_axis_agreement": {
            f"axis_{index + 1}": asdict(
                _agreement(reference_confirm_z[:, index], aligned_confirm[:, index])
            )
            for index in range(reference_confirm_z.shape[1])
        },
    }


def _coordinate_index(
    rows: Sequence[Mapping[str, object]],
    *,
    split_key: str,
    coordinate_names: Sequence[str],
) -> dict[tuple[str, str], tuple[float, ...]]:
    indexed = {}
    for row in rows:
        key = (str(row[split_key]), str(row["item_id"]))
        if key in indexed:
            raise ValueError(f"duplicate coordinate identity: {key}")
        values = tuple(float(row[name]) for name in coordinate_names)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite coordinates: {key}")
        indexed[key] = values
    return indexed


def _agreement(left: np.ndarray, right: np.ndarray) -> CoordinateAgreement:
    if left.shape != right.shape or left.ndim != 1 or len(left) < 3:
        raise ValueError("agreement vectors must be aligned and nontrivial")
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        raise ValueError("agreement vectors must have nonzero variance")
    return CoordinateAgreement(
        item_count=len(left),
        pearson=float(np.corrcoef(left, right)[0, 1]),
        spearman=float(np.corrcoef(_ranks(left), _ranks(right))[0, 1]),
        mean_absolute_difference=float(np.mean(np.abs(left - right))),
    )


def _subspace_identity(
    root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "path": str(root),
        "map_id": manifest["map_id"],
        "embedding_model": manifest["embedding_model"],
        "embedding_dimension": manifest["embedding_dimension"],
        "evidence_assessment": manifest["evidence_assessment"],
    }


def _render_report(payload: Mapping[str, object]) -> str:
    reference = payload["reference"]
    candidate = payload["candidate"]
    scalar = payload["scalar_prediction_agreement"]["confirmation"]
    alignment = payload["supervised_subspace_alignment"]
    candidate_evidence = payload["candidate_holdout_evidence"]
    lines = [
        "# Cross-embedding semantic-readiness confirmation",
        "",
        f"- Reference: `{reference['embedding_model']}`",
        f"- Candidate: `{candidate['embedding_model']}`",
        "- Frozen prompt, annotation, judge-panel, label-policy, and split checks: PASS",
        "",
        "## Candidate confirmation evidence",
        "",
        f"- Assessment: {candidate_evidence['assessment']['status'].upper()}",
        f"- Scalar Spearman: {candidate_evidence['scalar_spearman']:.4f}",
        f"- Scalar R-squared: {candidate_evidence['scalar_r_squared']:.4f}",
        f"- Relative MAE improvement: {candidate_evidence['relative_mae_improvement']:.2%}",
        "",
        "## Cross-embedding agreement",
        "",
        f"- Confirmation scalar-prediction Spearman: {scalar['spearman']:.4f}",
        f"- Confirmation scalar-prediction Pearson: {scalar['pearson']:.4f}",
        "- Two-dimensional alignment was learned on development only and evaluated on confirmation.",
        f"- Aligned confirmation Spearman (flattened): {alignment['confirmation_flattened']['spearman']:.4f}",
    ]
    for axis_name, metrics in alignment["confirmation_axis_agreement"].items():
        lines.append(
            f"- Aligned confirmation {axis_name} Spearman: {metrics['spearman']:.4f}"
        )
    lines.extend(["", payload["interpretation_guard"], ""])
    return "\n".join(lines)
