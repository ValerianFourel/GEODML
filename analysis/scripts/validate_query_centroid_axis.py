#!/usr/bin/env python3
"""Test a query-specific informational-to-buy centroid path in LLM2Vec-Gen.

For one exact query, matched surface frames instantiate multiple informational
and multiple buy-intent generation requests. Their reconstruction-state means
define query-specific endpoint centroids. The script decodes points on the line
between those centroids and measures decode/re-encode monotonicity and off-axis
residuals. This is a feasibility diagnostic, not a reranking experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Protocol, Sequence

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.llm2vec_gen_axis import (  # noqa: E402
    QUERY_CENTROID_AXIS_VERSION,
    DecodableAxis,
    anchor_query_to_decoded_text,
    axis_geometry_diagnostics,
    build_decodable_axis,
    build_query_centroid_requests,
    clean_decoded_realization,
    decode_record_checks,
    extend_axis_centroids,
    projection_residual_diagnostics,
    stable_array_hash,
)
from scripts.validate_llm2vec_gen_axis import LLM2VecGenBackend  # noqa: E402


DEFAULT_MODEL = "McGill-NLP/LLM2Vec-Gen-Qwen3-8B"
DEFAULT_TEMPLATE_BANK = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "query_conditioned_info_buy_centroid_v1.json"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "query_centroid_axis_feasibility"


class Backend(Protocol):
    backend_name: str
    model_name: str
    reconstruction_dtype: str

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str: ...


class FakeQueryCentroidBackend:
    """Deterministic CPU plumbing backend with no scientific meaning."""

    backend_name = "fake-query-centroid"
    model_name = "fake-query-centroid-v1"
    reconstruction_dtype = "float32"

    def __init__(self, query: str) -> None:
        self._query = query

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del batch_size, max_length
        pooled: list[np.ndarray] = []
        states: list[np.ndarray] = []
        frame_cues = (
            "write one",
            "state the",
            "formulate",
            "express the",
            "produce exactly",
            "compose one",
        )
        for text in texts:
            lowered = text.lower()
            coordinate = (
                1.0 if "intends to choose and buy or adopt" in lowered else 0.0
            )
            frame = next(
                (index for index, cue in enumerate(frame_cues) if cue in lowered),
                0,
            )
            surface = frame / max(len(frame_cues) - 1, 1)
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
        if coordinate < 0.25:
            purpose = "learn and understand neutral explanatory information about"
        elif coordinate < 0.75:
            purpose = "compare information while preparing to choose a solution for"
        else:
            purpose = "choose and buy or adopt a suitable solution now for"
        return f'The user wants to {purpose} "{self._query}".'


def _parse_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must contain numbers") from exc
    if len(grid) < 2 or any(not np.isfinite(item) for item in grid):
        raise argparse.ArgumentTypeError("grid needs at least two finite values")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise argparse.ArgumentTypeError("grid values must be strictly increasing")
    return grid


def _axis_region(coordinate: float) -> str:
    if coordinate < 0.0:
        return "pre-informational-extrapolation"
    if coordinate > 1.0:
        return "post-buy-extrapolation"
    return "centroid-interpolation"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
    )


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return None if denominator <= 1e-12 else float((first @ second) / denominator)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    first = _rankdata(left)
    second = _rankdata(right)
    if float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ANALYSIS_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for distribution in ("llm2vec-gen", "torch", "transformers", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": sys.version.split()[0],
        "dependency_versions": versions,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_gpus": os.getenv("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.getenv("HF_HOME"),
        "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
        "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
    }


def _report(diagnostics: dict[str, object], *, fake: bool) -> str:
    cycle = diagnostics["decode_cycle"]
    retention = diagnostics["query_retention"]
    anchored_retention = diagnostics["query_anchored_retention"]
    geometry = diagnostics["reconstruction_geometry"]
    banner = (
        "> **Mock output only.** This validates plumbing and supports no scientific "
        "claim.\n\n"
        if fake
        else ""
    )
    return f"""# Query-specific informational-to-buy centroid feasibility

{banner}For this query, multiple matched surface frames were encoded at each
semantic endpoint. Their means define query-specific informational and buy-intent
centroids. Every assigned latent point is on the line between those centroids by
construction; decoded language and decode/re-encode diagnostics are the test.

- Query: `{diagnostics['query']}`
- Latent coordinate range: `{diagnostics['latent_coordinate_range']}`
- Decoded latent points: `{diagnostics['latent_point_count']}`
- Extrapolated feasibility points: `{diagnostics['extrapolated_point_count']}`
- Matched endpoint pairs: `{diagnostics['surface_frame_count']}`
- Pair-direction cosine mean: `{geometry['pair_direction_cosine_mean']}`
- Surface-frame leave-one-out positive rate: `{geometry['leave_one_pair_out_positive_rate']}`
- Exact-query retention in raw decodes: `{retention['retained_count']}/{retention['decoded_count']}`
- Exact-query retention after fixed anchoring: `{anchored_retention['retained_count']}/{anchored_retention['decoded_count']}`
- Raw decode/re-encode Spearman: `{cycle['raw_reconstruction_spearman']}`
- Anchored decode/re-encode Spearman: `{cycle['anchored_reconstruction_spearman']}`
- Anchored coordinates strictly increasing: `{cycle['anchored_reconstruction_strictly_increasing']}`
- Maximum anchored normalized off-axis residual: `{cycle['maximum_anchored_off_axis_distance_over_centroid_distance']}`

The leave-one-out geometry here tests stability across surface frames for one
query, not generalization to unseen queries. Inspect every JSONL row for semantic
ordering, unintended criteria, fluency, and query fidelity before using this
construction in any reranking experiment. Coordinates outside `[0, 1]` extend
the fitted direction for feasibility analysis and are not experimental `B`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "local"), default="fake")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query", required=True)
    parser.add_argument("--template-bank", default=str(DEFAULT_TEMPLATE_BANK))
    parser.add_argument("--target-grid", type=_parse_grid)
    parser.add_argument("--axis-min", type=float, default=-1.0)
    parser.add_argument("--axis-max", type=float, default=2.0)
    parser.add_argument("--number-points", type=int, default=13)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.encode_batch_size <= 0 or args.encode_max_length <= 0:
        parser.error("encoding limits must be positive")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    if args.target_grid is None:
        if not np.isfinite(args.axis_min) or not np.isfinite(args.axis_max):
            parser.error("axis bounds must be finite")
        if args.axis_max <= args.axis_min:
            parser.error("axis-max must be greater than axis-min")
        if args.number_points < 12:
            parser.error("number-points must be at least 12")
        target_grid = tuple(
            float(value)
            for value in np.linspace(args.axis_min, args.axis_max, args.number_points)
        )
    else:
        target_grid = args.target_grid

    output = Path(args.output_dir)
    targets = {
        "diagnostics": output / "query_centroid_diagnostics.json",
        "grid": output / "decoded_query_centroid_grid.jsonl",
        "state": output / "query_centroid_state.npz",
        "report": output / "query_centroid_report.md",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(map(str, existing)))

    try:
        bank_path = Path(args.template_bank).resolve()
        specification = json.loads(bank_path.read_text(encoding="utf-8"))
        informational_rows, buy_rows = build_query_centroid_requests(
            args.query, specification
        )
        query = " ".join(args.query.split())
        backend: Backend = (
            FakeQueryCentroidBackend(query)
            if args.backend == "fake"
            else LLM2VecGenBackend(args.model)
        )
        all_requests = [
            *(row["request"] for row in informational_rows),
            *(row["request"] for row in buy_rows),
        ]
        pooled, reconstruction = backend.encode(
            all_requests,
            batch_size=args.encode_batch_size,
            max_length=args.encode_max_length,
        )
        count = len(informational_rows)
        if pooled.shape[0] != count * 2 or reconstruction.shape[0] != count * 2:
            raise ValueError(
                f"unexpected endpoint shapes: {pooled.shape}, {reconstruction.shape}"
            )
        pooled_info, pooled_buy = pooled[:count], pooled[count:]
        recon_info, recon_buy = reconstruction[:count], reconstruction[count:]
        recon_axis = build_decodable_axis(
            recon_info, recon_buy, axis_version=QUERY_CENTROID_AXIS_VERSION
        )
        pooled_axis = build_decodable_axis(
            pooled_info, pooled_buy, axis_version=QUERY_CENTROID_AXIS_VERSION
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    decoded_rows: list[dict[str, object]] = []
    latent_states: list[np.ndarray] = []
    for coordinate in target_grid:
        state = extend_axis_centroids(recon_axis, coordinate)
        latent_states.append(state)
        text = backend.decode(state, max_new_tokens=args.max_new_tokens).strip()
        realization = clean_decoded_realization(text)
        checks = decode_record_checks(realization)
        decoded_rows.append(
            {
                "path_kind": "query-specific-extended-centroid-line",
                "query": query,
                "assigned_coordinate": coordinate,
                "latent_axis_coordinate": coordinate,
                "experimental_B": coordinate if 0.0 <= coordinate <= 1.0 else None,
                "axis_region": _axis_region(coordinate),
                "assigned_state_projection": projection_residual_diagnostics(
                    recon_axis, state
                ),
                "latent_state_hash": stable_array_hash(state),
                "decoded_text": text,
                "decoded_realization": realization,
                "query_present_case_insensitive": (
                    query.casefold() in realization.casefold()
                ),
                "query_anchored_text": anchor_query_to_decoded_text(
                    query, realization
                ),
                "decoded_line_count": len(realization.splitlines()),
                "structural_checks": checks,
            }
        )

    raw_re_pooled, raw_re_reconstruction = backend.encode(
        [str(row["decoded_realization"]) for row in decoded_rows],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    anchored_re_pooled, anchored_re_reconstruction = backend.encode(
        [str(row["query_anchored_text"]) for row in decoded_rows],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    expected_shape = (len(decoded_rows), *recon_axis.state_shape)
    if raw_re_reconstruction.shape != expected_shape:
        parser.error(f"unexpected raw decode-cycle shape: {raw_re_reconstruction.shape}")
    if anchored_re_reconstruction.shape != expected_shape:
        parser.error(
            "unexpected anchored decode-cycle shape: "
            f"{anchored_re_reconstruction.shape}"
        )
    for index, row in enumerate(decoded_rows):
        row["reencoded_reconstruction_projection"] = projection_residual_diagnostics(
            recon_axis, raw_re_reconstruction[index]
        )
        row["reencoded_pooled_projection"] = projection_residual_diagnostics(
            pooled_axis, raw_re_pooled[index]
        )
        row["anchored_reencoded_reconstruction_projection"] = (
            projection_residual_diagnostics(
                recon_axis, anchored_re_reconstruction[index]
            )
        )
        row["anchored_reencoded_pooled_projection"] = projection_residual_diagnostics(
            pooled_axis, anchored_re_pooled[index]
        )
        row["decode_cycle_cosine_to_assigned_state"] = _cosine(
            latent_states[index], raw_re_reconstruction[index]
        )
        row["anchored_decode_cycle_cosine_to_assigned_state"] = _cosine(
            latent_states[index], anchored_re_reconstruction[index]
        )
        row["reencoded_reconstruction_residual_ratio"] = row[
            "reencoded_reconstruction_projection"
        ]["off_axis_distance_over_centroid_distance"]
        row["anchored_reencoded_reconstruction_residual_ratio"] = row[
            "anchored_reencoded_reconstruction_projection"
        ]["off_axis_distance_over_centroid_distance"]

    assigned = [float(row["assigned_coordinate"]) for row in decoded_rows]
    raw_re_coordinates = [
        float(row["reencoded_reconstruction_projection"]["axis_coordinate"])
        for row in decoded_rows
    ]
    raw_pooled_coordinates = [
        float(row["reencoded_pooled_projection"]["axis_coordinate"])
        for row in decoded_rows
    ]
    raw_off_axis = [
        float(
            row["reencoded_reconstruction_projection"][
                "off_axis_distance_over_centroid_distance"
            ]
        )
        for row in decoded_rows
    ]
    anchored_re_coordinates = [
        float(row["anchored_reencoded_reconstruction_projection"]["axis_coordinate"])
        for row in decoded_rows
    ]
    anchored_pooled_coordinates = [
        float(row["anchored_reencoded_pooled_projection"]["axis_coordinate"])
        for row in decoded_rows
    ]
    anchored_off_axis = [
        float(
            row["anchored_reencoded_reconstruction_projection"][
                "off_axis_distance_over_centroid_distance"
            ]
        )
        for row in decoded_rows
    ]
    retained = sum(bool(row["query_present_case_insensitive"]) for row in decoded_rows)
    anchored_retained = sum(
        query.casefold() in str(row["query_anchored_text"]).casefold()
        for row in decoded_rows
    )
    diagnostics: dict[str, object] = {
        "diagnostic_version": "query-centroid-axis-feasibility-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit_sha": _git_sha(),
        "status": "feasibility-only",
        "scientific_result": False,
        "axis_version": QUERY_CENTROID_AXIS_VERSION,
        "axis_hash": recon_axis.axis_hash,
        "query": query,
        "query_included_before_encoding_in_every_endpoint": True,
        "centroids_recomputed_for_each_query": True,
        "surface_frame_count": count,
        "surface_frame_ids": [row["frame_id"] for row in informational_rows],
        "informational_requests": informational_rows,
        "buy_intent_requests": buy_rows,
        "axis_formula": "H(query,B)=C_info(query)+B*(C_buy(query)-C_info(query))",
        "template_bank": str(bank_path),
        "template_bank_sha256": _sha256_file(bank_path),
        "model": backend.model_name,
        "backend": backend.backend_name,
        "reconstruction_dtype": backend.reconstruction_dtype,
        "target_grid": list(target_grid),
        "latent_coordinate_range": [min(target_grid), max(target_grid)],
        "latent_point_count": len(target_grid),
        "extrapolated_point_count": sum(
            coordinate < 0.0 or coordinate > 1.0 for coordinate in target_grid
        ),
        "experimental_B_domain": [0.0, 1.0],
        "coordinates_outside_B_are_feasibility_only": True,
        "reconstruction_geometry": axis_geometry_diagnostics(recon_info, recon_buy),
        "pooled_geometry": axis_geometry_diagnostics(pooled_info, pooled_buy),
        "query_retention": {
            "retained_count": retained,
            "decoded_count": len(decoded_rows),
            "rate": retained / len(decoded_rows),
        },
        "query_anchored_retention": {
            "retained_count": anchored_retained,
            "decoded_count": len(decoded_rows),
            "rate": anchored_retained / len(decoded_rows),
            "method": "fixed structural query prefix after latent decode",
        },
        "decode_cycle": {
            "raw_reencoded_reconstruction_coordinates": raw_re_coordinates,
            "raw_reencoded_pooled_coordinates": raw_pooled_coordinates,
            "raw_reconstruction_spearman": _spearman(assigned, raw_re_coordinates),
            "raw_pooled_spearman": _spearman(assigned, raw_pooled_coordinates),
            "raw_reconstruction_strictly_increasing": all(
                right > left
                for left, right in zip(raw_re_coordinates, raw_re_coordinates[1:])
            ),
            "raw_pooled_strictly_increasing": all(
                right > left
                for left, right in zip(
                    raw_pooled_coordinates, raw_pooled_coordinates[1:]
                )
            ),
            "raw_normalized_off_axis_residuals": raw_off_axis,
            "maximum_raw_off_axis_distance_over_centroid_distance": max(
                raw_off_axis
            ),
            "anchored_reencoded_reconstruction_coordinates": anchored_re_coordinates,
            "anchored_reencoded_pooled_coordinates": anchored_pooled_coordinates,
            "anchored_reconstruction_spearman": _spearman(
                assigned, anchored_re_coordinates
            ),
            "anchored_pooled_spearman": _spearman(
                assigned, anchored_pooled_coordinates
            ),
            "anchored_reconstruction_strictly_increasing": all(
                right > left
                for left, right in zip(
                    anchored_re_coordinates, anchored_re_coordinates[1:]
                )
            ),
            "anchored_pooled_strictly_increasing": all(
                right > left
                for left, right in zip(
                    anchored_pooled_coordinates, anchored_pooled_coordinates[1:]
                )
            ),
            "anchored_normalized_off_axis_residuals": anchored_off_axis,
            "maximum_anchored_off_axis_distance_over_centroid_distance": max(
                anchored_off_axis
            ),
            # Backward-compatible raw-decode aliases from feasibility v1.
            "reencoded_reconstruction_coordinates": raw_re_coordinates,
            "reencoded_pooled_coordinates": raw_pooled_coordinates,
            "reconstruction_spearman": _spearman(assigned, raw_re_coordinates),
            "pooled_spearman": _spearman(assigned, raw_pooled_coordinates),
            "reconstruction_strictly_increasing": all(
                right > left
                for left, right in zip(raw_re_coordinates, raw_re_coordinates[1:])
            ),
            "pooled_strictly_increasing": all(
                right > left
                for left, right in zip(
                    raw_pooled_coordinates, raw_pooled_coordinates[1:]
                )
            ),
            "normalized_off_axis_residuals": raw_off_axis,
            "maximum_off_axis_distance_over_centroid_distance": max(raw_off_axis),
            "same_model_diagnostic_only": True,
        },
        "interpretation": {
            "assigned_states_lie_on_axis_by_construction": True,
            "surface_frame_leave_one_out_is_not_topic_generalization": True,
            "decoded_semantic_monotonicity_requires_manual_review": True,
            "literal_query_is_a_fixed_invariant_not_latent_randomness": True,
            "mocked_runs_support_scientific_claims": False,
        },
        "runtime": _runtime(),
    }

    _atomic_npz(
        targets["state"],
        {
            "informational_endpoint_states": recon_info.astype(np.float32),
            "buy_intent_endpoint_states": recon_buy.astype(np.float32),
            "informational_centroid": recon_axis.informational_centroid.astype(
                np.float32
            ),
            "buy_intent_centroid": recon_axis.transactional_centroid.astype(np.float32),
            "direction_unit": recon_axis.direction_unit.astype(np.float32),
            "assigned_grid_states": np.stack(latent_states).astype(np.float32),
            "raw_reencoded_grid_states": raw_re_reconstruction.astype(np.float32),
            "reencoded_grid_states": raw_re_reconstruction.astype(np.float32),
            "query_anchored_reencoded_grid_states": anchored_re_reconstruction.astype(
                np.float32
            ),
        },
    )
    _atomic_json(targets["diagnostics"], diagnostics)
    _atomic_jsonl(targets["grid"], decoded_rows)
    _atomic_text(
        targets["report"], _report(diagnostics, fake=args.backend == "fake")
    )
    for path in targets.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
