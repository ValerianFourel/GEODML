#!/usr/bin/env python3
"""Test a randomized ownership-by-intent latent prompt plane."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.llm2vec_gen_axis import (  # noqa: E402
    OWNERSHIP_INTENT_PLANE_VERSION,
    build_ownership_intent_requests,
    build_realization_reconstruction_text,
    clean_decoded_realization,
    stable_array_hash,
)
from scripts.validate_llm2vec_gen_axis import LLM2VecGenBackend  # noqa: E402
from scripts.validate_query_centroid_axis import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _atomic_npz,
    _atomic_text,
    _cosine,
    _duplicate_groups,
    _git_sha,
    _monotonicity_diagnostics,
    _runtime,
    _sha256_file,
    _spearman,
)


DEFAULT_MODEL = "McGill-NLP/LLM2Vec-Gen-Qwen3-8B"
DEFAULT_BANK = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "query_conditioned_ownership_intent_plane_v1.json"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "ownership_intent_plane_feasibility"


class FakePlaneBackend:
    """Deterministic CPU plumbing backend with no scientific meaning."""

    backend_name = "fake-ownership-intent-plane"
    model_name = "fake-ownership-intent-plane-v1"
    reconstruction_dtype = "float32"

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del batch_size, max_length
        pooled: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for index, text in enumerate(texts):
            lowered = text.casefold()
            ownership = 0.0
            intent = 0.0
            if "prefer evidence published independently" in lowered:
                ownership = -1.0
            elif "prefer evidence controlled by vendors" in lowered:
                ownership = 1.0
            elif "seller-independent" in lowered:
                ownership = -1.0
            elif "seller-controlled" in lowered:
                ownership = 1.0
            if "not yet selecting or adopting" in lowered:
                intent = -1.0
            elif "ready to select and adopt" in lowered:
                intent = 1.0
            elif "informational" in lowered:
                intent = -1.0
            elif "transactional" in lowered:
                intent = 1.0
            surface = (index % 6) / 5.0
            state = np.zeros((2, 5), dtype=np.float32)
            state[:, 0] = ownership
            state[:, 1] = intent
            state[0, 2] = surface
            state[1, 3] = surface
            states.append(state)
            pooled.append(
                np.array([ownership, intent, surface, surface], dtype=np.float32)
            )
        return np.stack(pooled), np.stack(states)

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str:
        del max_new_tokens
        value = np.asarray(state)
        ownership = float(value[:, 0].mean())
        intent = float(value[:, 1].mean())
        intent_text = (
            "The B2B evaluator has informational intent and is learning about the category."
            if intent < 0
            else "The B2B evaluator has transactional intent and is selecting a product now."
        )
        ownership_text = (
            "Prefer seller-independent evidence when relevance and quality are comparable."
            if ownership < 0
            else "Prefer seller-controlled evidence when relevance and quality are comparable."
        )
        return f"{intent_text} {ownership_text}"


def _parse_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coordinates must be numbers") from exc
    if len(values) < 2 or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("coordinates need at least two finite values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("coordinates must be strictly increasing")
    return values


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("style seeds must be integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("style seeds must be unique and non-empty")
    return seeds


def _orthogonal_basis(
    ownership_effect: np.ndarray, intent_effect: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    ownership = np.asarray(ownership_effect, dtype=np.float64).reshape(-1)
    intent = np.asarray(intent_effect, dtype=np.float64).reshape(-1)
    ownership_scale = float(np.linalg.norm(ownership))
    intent_scale = float(np.linalg.norm(intent))
    if ownership_scale <= 1e-12 or intent_scale <= 1e-12:
        raise ValueError("semantic main-effect direction has zero norm")
    raw_cosine = float((ownership @ intent) / (ownership_scale * intent_scale))
    matrix = np.column_stack((ownership / ownership_scale, intent / intent_scale))
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    orthogonal = left @ right_t
    return (
        orthogonal[:, 0],
        orthogonal[:, 1],
        ownership_scale,
        intent_scale,
        raw_cosine,
    )


def _plane_projection(
    state: np.ndarray,
    *,
    centroid: np.ndarray,
    ownership_unit: np.ndarray,
    intent_unit: np.ndarray,
    ownership_scale: float,
    intent_scale: float,
) -> dict[str, float]:
    value = np.asarray(state, dtype=np.float64)
    centered = (value - centroid).reshape(-1)
    ownership_distance = float(centered @ ownership_unit)
    intent_distance = float(centered @ intent_unit)
    projected = ownership_distance * ownership_unit + intent_distance * intent_unit
    residual = float(np.linalg.norm(centered - projected))
    reference_scale = (ownership_scale + intent_scale) / 2.0
    return {
        "ownership_coordinate": ownership_distance / ownership_scale,
        "intent_coordinate": intent_distance / intent_scale,
        "off_plane_distance": residual,
        "off_plane_distance_over_mean_axis_scale": residual / reference_scale,
    }


def _policy_checks(text: str) -> dict[str, object]:
    lowered = text.casefold()
    forbidden = {
        "freshness": r"\bfresh(?:ness)?\b|\brecen(?:t|cy)\b",
        "authority": r"\bauthorit(?:y|ative)\b|\bcredib(?:le|ility)\b",
        "popularity": r"\bpopular(?:ity)?\b|\bbrand fame\b",
        "citations": r"\bcitations?\b",
        "hard-exclusion": r"\bexclude\b|\bnever rank\b|\bonly rank\b",
        "competitor-rule": r"\bcompet(?:e|es|itor|itors|ing)\b",
        "platform-history": r"\bgoogle play\b|\bapp store\b|\bremoved from\b",
    }
    found = [label for label, pattern in forbidden.items() if re.search(pattern, lowered)]
    ownership = any(
        term in lowered
        for term in ("seller", "vendor", "publisher", "first-party", "independent")
    )
    informational = any(term in lowered for term in ("informational", "learn", "understand", "research"))
    transactional = any(term in lowered for term in ("transactional", "select", "adopt", "choose", "buy"))
    return {
        "ownership_language_present": ownership,
        "informational_language_present": informational,
        "transactional_language_present": transactional,
        "forbidden_off_plane_criteria": found,
        "passes_lexical_invariant_screen": not found,
        "lexical_screen_is_not_semantic_validation": True,
    }


def _render_prompt(query: str, realization: str) -> str:
    return f"""You are the same B2B software evaluator for the same company and time horizon.

Exact keyword: {query}

Apply only these randomized semantic policies:
{realization.strip()}

Rank the supplied candidate documents using the fixed query-relevance and evidence-quality criteria. Do not add any other ranking criterion.

Query:
{{QUERY}}

Candidates:
{{CANDIDATES}}

Return exactly {{TOP_N}} candidate identifiers only, with no explanation."""


def _slice_monotonicity(
    rows: Sequence[dict[str, object]], *, assigned_key: str, recovered_key: str, fixed_key: str
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    fixed_values = sorted({float(row[fixed_key]) for row in rows})
    for fixed in fixed_values:
        selected = sorted(
            (row for row in rows if float(row[fixed_key]) == fixed),
            key=lambda row: float(row[assigned_key]),
        )
        assigned = [float(row[assigned_key]) for row in selected]
        recovered = [float(row["matched_projection"][recovered_key]) for row in selected]
        results.append(
            {
                "fixed_coordinate": fixed,
                "assigned": assigned,
                "recovered": recovered,
                "spearman": _spearman(assigned, recovered),
                "monotonicity": _monotonicity_diagnostics(assigned, recovered),
            }
        )
    return results


def _report(diagnostics: dict[str, object], *, fake: bool) -> str:
    geometry = diagnostics["plane_geometry"]
    cycle = diagnostics["decode_cycle"]
    banner = "> **Mock output only.** No scientific claim.\n\n" if fake else ""
    return f"""# Ownership-by-intent latent plane feasibility

{banner}This run independently varies ownership preference `O`, search intent `I`,
and an axis-orthogonal surface residual selected by `S`.

- Query: `{diagnostics['query']}`
- Ownership grid: `{diagnostics['ownership_grid']}`
- Intent grid: `{diagnostics['intent_grid']}`
- Style seeds: `{diagnostics['style_seeds']}`
- Decoded prompts: `{diagnostics['latent_point_count']}`
- Raw main-effect cosine before orthogonalization: `{geometry['raw_main_effect_cosine']}`
- Factorial interaction/main-effect norm ratio: `{geometry['interaction_to_mean_main_effect_norm']}`
- Mean corner reconstruction error: `{geometry['mean_corner_residual_over_mean_axis_scale']}`
- Ownership slice Spearman mean: `{cycle['ownership_slice_spearman_mean']}`
- Intent slice Spearman mean: `{cycle['intent_slice_spearman_mean']}`
- Maximum matched off-plane residual: `{cycle['maximum_matched_off_plane_residual']}`
- Duplicate decoded groups: `{diagnostics['duplicates']['group_count']}`
- Lexical invariant passes: `{diagnostics['semantic_screen']['passing_count']}/{diagnostics['latent_point_count']}`

Coordinates outside `[-1,1]` extend each fitted direction and are feasibility
probes, not randomized treatment values. Valid two-factor use requires stable
within-slice ordering and acceptable interaction curvature and off-plane drift.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "local"), default="fake")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query", required=True)
    parser.add_argument("--template-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--ownership-grid", type=_parse_values, default=(-2.0, -1.0, 0.0, 1.0, 2.0))
    parser.add_argument("--intent-grid", type=_parse_values, default=(-2.0, -1.0, 0.0, 1.0, 2.0))
    parser.add_argument("--style-seeds", type=_parse_seeds, default=(0, 1))
    parser.add_argument("--surface-scale", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.surface_scale) or args.surface_scale < 0:
        parser.error("surface-scale must be finite and non-negative")

    output = Path(args.output_dir)
    targets = {
        "diagnostics": output / "ownership_intent_plane_diagnostics.json",
        "grid": output / "decoded_ownership_intent_grid.jsonl",
        "state": output / "ownership_intent_plane_state.npz",
        "report": output / "ownership_intent_plane_report.md",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(map(str, existing)))

    try:
        query = " ".join(args.query.split())
        bank_path = Path(args.template_bank).resolve()
        specification = json.loads(bank_path.read_text(encoding="utf-8"))
        requests = build_ownership_intent_requests(query, specification)
        backend = FakePlaneBackend() if args.backend == "fake" else LLM2VecGenBackend(args.model)
        corner_keys = [f"o{o:+d}_i{i:+d}" for o in (-1, 1) for i in (-1, 1)]
        count = len(requests[corner_keys[0]])
        ordered = [row["request"] for key in corner_keys for row in requests[key]]
        _, states = backend.encode(
            ordered, batch_size=args.encode_batch_size, max_length=args.encode_max_length
        )
        corners = {
            key: states[index * count : (index + 1) * count]
            for index, key in enumerate(corner_keys)
        }
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    signed = {(o, i): corners[f"o{o:+d}_i{i:+d}"] for o in (-1, 1) for i in (-1, 1)}
    stacked = np.stack([signed[(o, i)] for o in (-1, 1) for i in (-1, 1)])
    grand = stacked.mean(axis=(0, 1))
    ownership_effect = np.mean(
        [o * signed[(o, i)] for o in (-1, 1) for i in (-1, 1)], axis=(0, 1)
    )
    intent_effect = np.mean(
        [i * signed[(o, i)] for o in (-1, 1) for i in (-1, 1)], axis=(0, 1)
    )
    interaction = np.mean(
        [o * i * signed[(o, i)] for o in (-1, 1) for i in (-1, 1)], axis=(0, 1)
    )
    ownership_unit, intent_unit, ownership_scale, intent_scale, raw_cosine = (
        _orthogonal_basis(ownership_effect, intent_effect)
    )
    shape = grand.shape
    ownership_unit_state = ownership_unit.reshape(shape)
    intent_unit_state = intent_unit.reshape(shape)

    frame_centroids = stacked.mean(axis=0)
    surface_residuals: list[np.ndarray] = []
    for frame_centroid in frame_centroids:
        residual = (frame_centroid - grand).reshape(-1)
        residual -= float(residual @ ownership_unit) * ownership_unit
        residual -= float(residual @ intent_unit) * intent_unit
        surface_residuals.append(residual.reshape(shape))

    corner_residuals: list[float] = []
    mean_scale = (ownership_scale + intent_scale) / 2.0
    for o in (-1, 1):
        for i in (-1, 1):
            predicted = grand + o * ownership_scale * ownership_unit_state + i * intent_scale * intent_unit_state
            corner_residuals.append(
                float(np.linalg.norm((signed[(o, i)].mean(axis=0) - predicted).reshape(-1))) / mean_scale
            )

    decoded_rows: list[dict[str, object]] = []
    assigned_states: list[np.ndarray] = []
    for seed in args.style_seeds:
        frame_index = seed % count
        surface = args.surface_scale * surface_residuals[frame_index]
        for ownership in args.ownership_grid:
            for intent in args.intent_grid:
                state = (
                    grand
                    + ownership * ownership_scale * ownership_unit_state
                    + intent * intent_scale * intent_unit_state
                    + surface
                )
                assigned_states.append(state)
                raw = backend.decode(state, max_new_tokens=args.max_new_tokens)
                realization = clean_decoded_realization(raw)
                checks = _policy_checks(realization)
                decoded_rows.append(
                    {
                        "path_kind": "query-specific-ownership-intent-plane",
                        "query": query,
                        "assigned_ownership": ownership,
                        "assigned_intent": intent,
                        "experimental_O": ownership if -1 <= ownership <= 1 else None,
                        "experimental_I": intent if -1 <= intent <= 1 else None,
                        "style_seed": seed,
                        "surface_frame_id": requests[corner_keys[0]][frame_index]["frame_id"],
                        "assigned_state_hash": stable_array_hash(state),
                        "assigned_projection": _plane_projection(
                            state,
                            centroid=grand,
                            ownership_unit=ownership_unit,
                            intent_unit=intent_unit,
                            ownership_scale=ownership_scale,
                            intent_scale=intent_scale,
                        ),
                        "decoded_raw": raw,
                        "decoded_realization": realization,
                        "semantic_checks": checks,
                        "rendered_reranking_prompt": _render_prompt(query, realization),
                    }
                )

    _, matched_states = backend.encode(
        [build_realization_reconstruction_text(str(row["decoded_realization"])) for row in decoded_rows],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    for index, row in enumerate(decoded_rows):
        row["matched_projection"] = _plane_projection(
            matched_states[index],
            centroid=grand,
            ownership_unit=ownership_unit,
            intent_unit=intent_unit,
            ownership_scale=ownership_scale,
            intent_scale=intent_scale,
        )
        row["matched_cosine_to_assigned_state"] = _cosine(
            assigned_states[index], matched_states[index]
        )

    ownership_slices: list[dict[str, object]] = []
    intent_slices: list[dict[str, object]] = []
    for seed in args.style_seeds:
        seed_rows = [row for row in decoded_rows if row["style_seed"] == seed]
        ownership_slices.extend(
            {"style_seed": seed, **row}
            for row in _slice_monotonicity(
                seed_rows,
                assigned_key="assigned_ownership",
                recovered_key="ownership_coordinate",
                fixed_key="assigned_intent",
            )
        )
        intent_slices.extend(
            {"style_seed": seed, **row}
            for row in _slice_monotonicity(
                seed_rows,
                assigned_key="assigned_intent",
                recovered_key="intent_coordinate",
                fixed_key="assigned_ownership",
            )
        )
    ownership_spearman = [row["spearman"] for row in ownership_slices if row["spearman"] is not None]
    intent_spearman = [row["spearman"] for row in intent_slices if row["spearman"] is not None]
    off_plane = [
        float(row["matched_projection"]["off_plane_distance_over_mean_axis_scale"])
        for row in decoded_rows
    ]
    duplicate_input = [
        {
            "assigned_coordinate": index,
            "decoded_realization": row["decoded_realization"],
        }
        for index, row in enumerate(decoded_rows)
    ]
    duplicates = _duplicate_groups(duplicate_input)
    interaction_norm = float(np.linalg.norm(interaction.reshape(-1)))
    diagnostics: dict[str, object] = {
        "diagnostic_version": "ownership-intent-plane-feasibility-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": _git_sha(),
        "status": "feasibility-only",
        "scientific_result": False,
        "plane_version": OWNERSHIP_INTENT_PLANE_VERSION,
        "query": query,
        "formula": "Z(q,O,I,S)=C+O*D_O_orth+I*D_I_orth+R_S_orth",
        "ownership_grid": list(args.ownership_grid),
        "intent_grid": list(args.intent_grid),
        "style_seeds": list(args.style_seeds),
        "surface_scale": args.surface_scale,
        "latent_point_count": len(decoded_rows),
        "coordinates_outside_unit_square_are_feasibility_only": True,
        "requests": requests,
        "template_bank": str(bank_path),
        "template_bank_sha256": _sha256_file(bank_path),
        "backend": backend.backend_name,
        "model": backend.model_name,
        "plane_geometry": {
            "raw_main_effect_cosine": raw_cosine,
            "ownership_effect_norm": ownership_scale,
            "intent_effect_norm": intent_scale,
            "interaction_effect_norm": interaction_norm,
            "interaction_to_mean_main_effect_norm": interaction_norm / mean_scale,
            "corner_residuals_over_mean_axis_scale": corner_residuals,
            "mean_corner_residual_over_mean_axis_scale": float(np.mean(corner_residuals)),
            "orthogonal_basis_cosine": float(ownership_unit @ intent_unit),
            "surface_projection_max_abs": max(
                max(
                    abs(float(residual.reshape(-1) @ ownership_unit)),
                    abs(float(residual.reshape(-1) @ intent_unit)),
                )
                for residual in surface_residuals
            ),
        },
        "decode_cycle": {
            "method": "instruction-matched exact two-policy reconstruction",
            "ownership_slices": ownership_slices,
            "intent_slices": intent_slices,
            "ownership_slice_spearman_mean": float(np.mean(ownership_spearman)) if ownership_spearman else None,
            "intent_slice_spearman_mean": float(np.mean(intent_spearman)) if intent_spearman else None,
            "matched_off_plane_residuals": off_plane,
            "maximum_matched_off_plane_residual": max(off_plane),
        },
        "duplicates": {"group_count": len(duplicates), "groups": duplicates},
        "semantic_screen": {
            "passing_count": sum(
                bool(row["semantic_checks"]["passes_lexical_invariant_screen"])
                for row in decoded_rows
            ),
            "ownership_language_count": sum(
                bool(row["semantic_checks"]["ownership_language_present"])
                for row in decoded_rows
            ),
            "informational_language_count": sum(
                bool(row["semantic_checks"]["informational_language_present"])
                for row in decoded_rows
            ),
            "transactional_language_count": sum(
                bool(row["semantic_checks"]["transactional_language_present"])
                for row in decoded_rows
            ),
        },
        "runtime": _runtime(),
        "interpretation": {
            "assigned_coordinates_are_randomization_variables": True,
            "embeddings_are_validation_not_confounders": True,
            "surface_residuals_are_axis_orthogonal_by_construction": True,
            "manual_semantic_review_required": True,
            "mocked_runs_support_scientific_claims": False,
        },
    }

    _atomic_npz(
        targets["state"],
        {
            "grand_centroid": grand.astype(np.float32),
            "ownership_effect_raw": ownership_effect.astype(np.float32),
            "intent_effect_raw": intent_effect.astype(np.float32),
            "interaction_effect": interaction.astype(np.float32),
            "ownership_unit_orthogonal": ownership_unit_state.astype(np.float32),
            "intent_unit_orthogonal": intent_unit_state.astype(np.float32),
            "surface_residuals_orthogonal": np.stack(surface_residuals).astype(np.float32),
            "assigned_states": np.stack(assigned_states).astype(np.float32),
            "matched_states": matched_states.astype(np.float32),
        },
    )
    _atomic_json(targets["diagnostics"], diagnostics)
    _atomic_jsonl(targets["grid"], decoded_rows)
    _atomic_text(targets["report"], _report(diagnostics, fake=args.backend == "fake"))
    for path in targets.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
