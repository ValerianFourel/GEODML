#!/usr/bin/env python3
"""Test a query-specific seller-independent to seller-controlled latent field."""

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
    SOURCE_OWNERSHIP_AXIS_VERSION,
    axis_geometry_diagnostics,
    build_decodable_axis,
    build_realization_reconstruction_text,
    build_source_ownership_requests,
    clean_decoded_realization,
    decode_record_checks,
    projection_residual_diagnostics,
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
    _parse_grid,
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
    / "query_conditioned_source_ownership_axis_v1.json"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "source_ownership_axis_feasibility"


class FakeSourceOwnershipBackend:
    """Deterministic CPU plumbing backend with no scientific meaning."""

    backend_name = "fake-source-ownership"
    model_name = "fake-source-ownership-v1"
    reconstruction_dtype = "float32"

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del batch_size, max_length
        pooled: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for index, text in enumerate(texts):
            lowered = text.casefold()
            if "strongly prefer evidence published by sources independent" in lowered:
                coordinate = -1.0
            elif "strongly prefer evidence published by vendors" in lowered:
                coordinate = 1.0
            elif "do not use whether a publisher" in lowered:
                coordinate = 0.0
            elif "seller-controlled" in lowered or "prefer vendor-controlled" in lowered:
                coordinate = 1.0
            elif "seller-independent" in lowered or "prefer independent" in lowered:
                coordinate = -1.0
            else:
                coordinate = 0.0
            surface = (index % 6) / 5.0
            state = np.zeros((2, 4), dtype=np.float32)
            state[:, 0] = coordinate
            state[0, 1] = surface
            state[1, 2] = surface
            states.append(state)
            pooled.append(np.array([coordinate, surface, surface], dtype=np.float32))
        return np.stack(pooled), np.stack(states)

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str:
        del max_new_tokens
        coordinate = float(np.asarray(state)[:, 0].mean())
        if coordinate < -0.25:
            return "Prefer seller-independent evidence when relevance and quality are comparable."
        if coordinate > 0.25:
            return "Prefer seller-controlled evidence when relevance and quality are comparable."
        return "Apply no preference based on whether evidence is seller controlled or independent."


def _centroid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or not np.isfinite(array).all():
        raise ValueError("endpoint states must be a finite batch")
    return array.mean(axis=0)


def _ownership_projection(
    state: np.ndarray,
    *,
    neutral: np.ndarray,
    direction_unit: np.ndarray,
    half_endpoint_distance: float,
) -> dict[str, float]:
    value = np.asarray(state, dtype=np.float64)
    centered = value - neutral
    projected_distance = float(centered.reshape(-1) @ direction_unit.reshape(-1))
    coordinate = projected_distance / half_endpoint_distance
    projected = neutral + projected_distance * direction_unit
    residual = float(np.linalg.norm((value - projected).reshape(-1)))
    return {
        "lambda_coordinate": coordinate,
        "off_axis_distance": residual,
        "off_axis_distance_over_half_endpoint_distance": (
            residual / half_endpoint_distance
        ),
    }


def _render_prompt(query: str, policy: str) -> str:
    return f"""You are evaluating B2B SaaS software for a company using the exact keyword: {query}

Rank the supplied candidate documents by relevance to that exact keyword and by evidence quality.

Publisher-ownership policy:
{policy.strip()}

Do not change the evaluator identity, keyword, company context, time horizon, candidates, relevance criterion, evidence-quality criterion, ranking task, or output format.

Query:
{{QUERY}}

Candidates:
{{CANDIDATES}}

Return exactly {{TOP_N}} candidate identifiers only, with no explanation."""


def _ownership_policy_checks(policy: str) -> dict[str, object]:
    lowered = policy.casefold()
    forbidden_patterns = {
        "freshness": r"\bfresh(?:ness)?\b|\brecen(?:t|cy)\b",
        "authority": r"\bauthorit(?:y|ative)\b|\bcredib(?:le|ility)\b",
        "popularity": r"\bpopular(?:ity)?\b|\bbrand fame\b",
        "citations": r"\bcitations?\b",
        "page-length": r"\bpage length\b",
        "statistics-density": r"\bstatistics? density\b|\bnumerical density\b",
        "writing-quality": r"\bwriting quality\b",
        "review-scores": r"\breview scores?\b",
        "hard-exclusion": r"\bexclude\b|\bnever rank\b|\bonly rank\b",
    }
    forbidden = [
        label for label, pattern in forbidden_patterns.items() if re.search(pattern, lowered)
    ]
    ownership_terms = [
        term
        for term in ("seller", "vendor", "publisher", "first-party", "independent")
        if term in lowered
    ]
    return {
        "ownership_terms": ownership_terms,
        "ownership_language_present": bool(ownership_terms),
        "relevance_preserved": "relevance" in lowered or "relevant" in lowered,
        "quality_preserved": "quality" in lowered,
        "forbidden_off_axis_criteria": forbidden,
        "passes_lexical_invariant_screen": not forbidden,
        "lexical_screen_is_not_semantic_validation": True,
    }


def _report(diagnostics: dict[str, object], *, fake: bool) -> str:
    geometry = diagnostics["endpoint_geometry"]
    neutral = diagnostics["neutral_location_on_endpoint_axis"]
    cycle = diagnostics["decode_cycle"]
    banner = (
        "> **Mock output only.** This supports no scientific claim.\n\n"
        if fake
        else ""
    )
    return f"""# Seller-independent to seller-controlled evidence axis

{banner}This feasibility run holds the B2B evaluator and ranking task fixed while
varying only publisher-ownership preference along lambda in `[-1, 1]`.

- Query: `{diagnostics['query']}`
- Matched surface frames per region: `{diagnostics['surface_frame_count']}`
- Endpoint pair-direction cosine mean: `{geometry['pair_direction_cosine_mean']}`
- Endpoint leave-one-frame-out positive rate: `{geometry['leave_one_pair_out_positive_rate']}`
- Neutral coordinate on independent-to-controlled endpoint axis (ideal `0`): `{neutral['lambda_coordinate']}`
- Neutral residual from endpoint line: `{neutral['off_axis_distance_over_half_endpoint_distance']}`
- Instruction-matched Spearman: `{cycle['matched_policy_spearman']}`
- Instruction-matched decreases/ties: `{cycle['matched_policy_monotonicity']['decrease_count']}/{cycle['matched_policy_monotonicity']['tie_count']}`
- Maximum instruction-matched residual: `{cycle['maximum_matched_policy_residual']}`
- Duplicate decoded policy groups: `{diagnostics['decoded_policy_duplicates']['group_count']}`
- Decoded policies with ownership language: `{diagnostics['ownership_language']['count']}/{diagnostics['latent_point_count']}`
- Decoded policies passing off-axis lexical screen: `{diagnostics['semantic_invariant_screen']['passing_count']}/{diagnostics['latent_point_count']}`

A coherent field requires aligned endpoint differences, a neutral centroid near
the endpoint midpoint and line, and decoded/re-encoded policies that move
monotonically without introducing other ranking criteria. This is a feasibility
diagnostic, not a reranking result.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "local"), default="fake")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query", required=True)
    parser.add_argument("--template-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--lambda-grid", type=_parse_grid)
    parser.add_argument("--number-points", type=int, default=13)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.number_points < 12:
        parser.error("number-points must be at least 12")
    grid = (
        args.lambda_grid
        if args.lambda_grid is not None
        else tuple(float(value) for value in np.linspace(-1.0, 1.0, args.number_points))
    )
    if min(grid) < -1.0 or max(grid) > 1.0:
        parser.error("lambda coordinates must remain in [-1, 1]")

    output = Path(args.output_dir)
    targets = {
        "diagnostics": output / "source_ownership_diagnostics.json",
        "grid": output / "decoded_source_ownership_grid.jsonl",
        "state": output / "source_ownership_state.npz",
        "report": output / "source_ownership_report.md",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(map(str, existing)))

    try:
        query = " ".join(args.query.split())
        bank_path = Path(args.template_bank).resolve()
        specification = json.loads(bank_path.read_text(encoding="utf-8"))
        requests = build_source_ownership_requests(query, specification)
        backend = (
            FakeSourceOwnershipBackend()
            if args.backend == "fake"
            else LLM2VecGenBackend(args.model)
        )
        ordered = (
            requests["seller-independent"]
            + requests["neutral"]
            + requests["seller-controlled"]
        )
        pooled, reconstruction = backend.encode(
            [row["request"] for row in ordered],
            batch_size=args.encode_batch_size,
            max_length=args.encode_max_length,
        )
        count = len(requests["neutral"])
        independent_recon = reconstruction[:count]
        neutral_recon = reconstruction[count : count * 2]
        controlled_recon = reconstruction[count * 2 :]
        independent_pooled = pooled[:count]
        neutral_pooled = pooled[count : count * 2]
        controlled_pooled = pooled[count * 2 :]
        endpoint_axis = build_decodable_axis(
            independent_recon,
            controlled_recon,
            axis_version=SOURCE_OWNERSHIP_AXIS_VERSION,
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    independent_centroid = _centroid(independent_recon)
    neutral_centroid = _centroid(neutral_recon)
    controlled_centroid = _centroid(controlled_recon)
    endpoint_delta = controlled_centroid - independent_centroid
    endpoint_distance = float(np.linalg.norm(endpoint_delta.reshape(-1)))
    direction_unit = endpoint_delta / endpoint_distance
    half_distance = endpoint_distance / 2.0

    decoded_rows: list[dict[str, object]] = []
    assigned_states: list[np.ndarray] = []
    for coordinate in grid:
        state = neutral_centroid + coordinate * half_distance * direction_unit
        assigned_states.append(state)
        raw = backend.decode(state, max_new_tokens=args.max_new_tokens)
        policy = clean_decoded_realization(raw)
        prompt = _render_prompt(query, policy)
        checks = decode_record_checks(policy)
        ownership_checks = _ownership_policy_checks(policy)
        decoded_rows.append(
            {
                "path_kind": "query-specific-source-ownership-axis",
                "query": query,
                "assigned_lambda": coordinate,
                "assigned_state_hash": stable_array_hash(state),
                "assigned_state_projection": _ownership_projection(
                    state,
                    neutral=neutral_centroid,
                    direction_unit=direction_unit,
                    half_endpoint_distance=half_distance,
                ),
                "decoded_raw": raw,
                "decoded_policy": policy,
                "ownership_language_present": ownership_checks[
                    "ownership_language_present"
                ],
                "ownership_policy_checks": ownership_checks,
                "structural_checks": checks,
                "rendered_reranking_prompt": prompt,
            }
        )

    matched_pooled, matched_recon = backend.encode(
        [
            build_realization_reconstruction_text(str(row["decoded_policy"]))
            for row in decoded_rows
        ],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    prompt_pooled, prompt_recon = backend.encode(
        [str(row["rendered_reranking_prompt"]) for row in decoded_rows],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    del matched_pooled, prompt_pooled
    for index, row in enumerate(decoded_rows):
        row["matched_policy_projection"] = _ownership_projection(
            matched_recon[index],
            neutral=neutral_centroid,
            direction_unit=direction_unit,
            half_endpoint_distance=half_distance,
        )
        row["full_prompt_projection"] = _ownership_projection(
            prompt_recon[index],
            neutral=neutral_centroid,
            direction_unit=direction_unit,
            half_endpoint_distance=half_distance,
        )
        row["matched_policy_cosine_to_assigned_state"] = _cosine(
            assigned_states[index], matched_recon[index]
        )

    assigned = [float(row["assigned_lambda"]) for row in decoded_rows]
    recovered = [
        float(row["matched_policy_projection"]["lambda_coordinate"])
        for row in decoded_rows
    ]
    residuals = [
        float(
            row["matched_policy_projection"][
                "off_axis_distance_over_half_endpoint_distance"
            ]
        )
        for row in decoded_rows
    ]
    neutral_endpoint_projection = projection_residual_diagnostics(
        endpoint_axis, neutral_centroid
    )
    neutral_location = {
        "lambda_coordinate": (
            2.0 * neutral_endpoint_projection["axis_coordinate"] - 1.0
        ),
        "off_axis_distance_over_half_endpoint_distance": (
            2.0
            * neutral_endpoint_projection[
                "off_axis_distance_over_centroid_distance"
            ]
        ),
    }
    duplicates = _duplicate_groups(
        [
            {
                "assigned_coordinate": row["assigned_lambda"],
                "decoded_realization": row["decoded_policy"],
            }
            for row in decoded_rows
        ]
    )
    diagnostics: dict[str, object] = {
        "diagnostic_version": "source-ownership-axis-feasibility-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": _git_sha(),
        "status": "feasibility-only",
        "scientific_result": False,
        "axis_version": SOURCE_OWNERSHIP_AXIS_VERSION,
        "query": query,
        "lambda_definition": {
            "range": [-1.0, 1.0],
            "minus_one": "prefer seller-independent evidence",
            "zero": "ownership-neutral",
            "plus_one": "prefer seller-controlled evidence",
        },
        "axis_formula": (
            "H(query,lambda)=C_neutral(query)+lambda*"
            "0.5*(C_controlled(query)-C_independent(query))"
        ),
        "all_nonownership_ranking_components_fixed": True,
        "surface_frame_count": count,
        "requests": requests,
        "template_bank": str(bank_path),
        "template_bank_sha256": _sha256_file(bank_path),
        "backend": backend.backend_name,
        "model": backend.model_name,
        "latent_point_count": len(grid),
        "lambda_grid": list(grid),
        "endpoint_geometry": axis_geometry_diagnostics(
            independent_recon, controlled_recon
        ),
        "neutral_location_on_endpoint_axis": neutral_location,
        "decoded_policy_duplicates": {
            "group_count": len(duplicates),
            "groups": duplicates,
        },
        "ownership_language": {
            "count": sum(
                bool(row["ownership_language_present"]) for row in decoded_rows
            )
        },
        "semantic_invariant_screen": {
            "passing_count": sum(
                bool(
                    row["ownership_policy_checks"][
                        "passes_lexical_invariant_screen"
                    ]
                )
                for row in decoded_rows
            ),
            "failing_points": [
                {
                    "assigned_lambda": row["assigned_lambda"],
                    "forbidden_off_axis_criteria": row["ownership_policy_checks"][
                        "forbidden_off_axis_criteria"
                    ],
                }
                for row in decoded_rows
                if not row["ownership_policy_checks"][
                    "passes_lexical_invariant_screen"
                ]
            ],
        },
        "decode_cycle": {
            "method": "instruction-matched exact policy reconstruction",
            "matched_policy_coordinates": recovered,
            "matched_policy_spearman": _spearman(assigned, recovered),
            "matched_policy_monotonicity": _monotonicity_diagnostics(
                assigned, recovered
            ),
            "matched_policy_residuals": residuals,
            "maximum_matched_policy_residual": max(residuals),
        },
        "runtime": _runtime(),
        "interpretation": {
            "neutral_location_is_measured_not_assumed": True,
            "assigned_states_lie_on_neutral_origin_direction_by_construction": True,
            "manual_semantic_review_required": True,
            "mocked_runs_support_scientific_claims": False,
        },
    }

    _atomic_npz(
        targets["state"],
        {
            "seller_independent_states": independent_recon.astype(np.float32),
            "neutral_states": neutral_recon.astype(np.float32),
            "seller_controlled_states": controlled_recon.astype(np.float32),
            "seller_independent_pooled": independent_pooled.astype(np.float32),
            "neutral_pooled": neutral_pooled.astype(np.float32),
            "seller_controlled_pooled": controlled_pooled.astype(np.float32),
            "seller_independent_centroid": independent_centroid.astype(np.float32),
            "neutral_centroid": neutral_centroid.astype(np.float32),
            "seller_controlled_centroid": controlled_centroid.astype(np.float32),
            "direction_unit": direction_unit.astype(np.float32),
            "assigned_states": np.stack(assigned_states).astype(np.float32),
            "matched_policy_states": matched_recon.astype(np.float32),
            "full_prompt_states": prompt_recon.astype(np.float32),
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
