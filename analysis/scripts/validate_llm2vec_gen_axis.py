#!/usr/bin/env python3
"""Locate and decode an informational-to-transactional LLM2Vec-Gen axis.

This is a feasibility diagnostic, not a reranking experiment.  It encodes a
versioned bank of paired endpoint templates, estimates a direction in both the
pooled and decodable reconstruction representations, performs
leave-one-topic-out geometry checks, and greedily decodes interpolation points.
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
    ENCODING_INSTRUCTION_VERSION,
    LLM2VEC_GEN_AXIS_VERSION,
    DecodableAxis,
    axis_geometry_diagnostics,
    build_decodable_axis,
    build_encoding_text,
    decode_record_checks,
    inject_query_after_decode,
    interpolate_axis_centroids,
    interpolate_endpoint_pair,
    project_onto_axis,
    stable_array_hash,
)


DEFAULT_MODEL = "McGill-NLP/LLM2Vec-Gen-Qwen3-8B"
DEFAULT_ENDPOINTS = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "search_purpose_endpoint_pairs_v1.json"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "llm2vec_gen_axis_feasibility"


class ReconstructionBackend(Protocol):
    backend_name: str
    model_name: str
    reconstruction_dtype: str

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pooled embeddings and reconstruction states."""

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str:
        """Greedily decode one reconstruction state."""


class LLM2VecGenBackend:
    """Lazy single-GPU adapter around the official llm2vec-gen package."""

    backend_name = "llm2vec-gen"

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from llm2vec_gen import LLM2VecGenModel
        except ImportError as exc:  # pragma: no cover - cluster dependency
            raise ImportError(
                "local backend requires llm2vec-gen and its GPU dependencies; "
                "install analysis/requirements-horeka-llm2vec-gen.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("the local LLM2Vec-Gen feasibility run requires CUDA")
        visible = torch.cuda.device_count()
        if visible != 1:
            raise RuntimeError(
                "LLM2Vec-Gen's current high-level loader moves the full model to one "
                f"CUDA device; expose exactly one GPU, found {visible}"
            )
        self._torch = torch
        self.model_name = model_name
        self._model = LLM2VecGenModel.from_pretrained(model_name)
        self._device = self._model.device
        try:
            parameter = next(self._model.model.decoder_model.parameters())
            self._decode_dtype = parameter.dtype
        except (AttributeError, StopIteration):  # pragma: no cover - defensive
            self._decode_dtype = torch.float32
        self.reconstruction_dtype = str(self._decode_dtype).replace("torch.", "")

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        pooled_batches: list[np.ndarray] = []
        reconstruction_batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            pooled, reconstruction = self._model.encode(
                list(texts[start : start + batch_size]),
                max_length=max_length,
                get_recon_hidden_states=True,
            )
            pooled_batches.append(pooled.detach().float().cpu().numpy())
            reconstruction_batches.append(
                reconstruction.detach().float().cpu().numpy()
            )
        return (
            np.concatenate(pooled_batches, axis=0),
            np.concatenate(reconstruction_batches, axis=0),
        )

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str:
        reconstruction = self._torch.as_tensor(
            np.asarray(state), device=self._device, dtype=self._decode_dtype
        )
        return str(
            self._model.generate(
                recon_hidden_states=reconstruction,
                max_new_tokens=max_new_tokens,
            )
        ).strip()


class FakeReconstructionBackend:
    """Deterministic CPU backend for plumbing tests only."""

    backend_name = "fake-llm2vec-gen"
    model_name = "fake-llm2vec-gen-v1"
    reconstruction_dtype = "float32"

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del batch_size, max_length
        if not texts:
            raise ValueError("fake backend requires at least one text")
        pooled: list[np.ndarray] = []
        states: list[np.ndarray] = []
        topics = (
            "password manager",
            "crm software",
            "cloud backup",
            "project management",
            "website analytics",
            "video conferenc",
        )
        for text in texts:
            lowered = text.lower()
            index = next(
                (position for position, topic in enumerate(topics) if topic in lowered),
                0,
            )
            if "compare approaches and prepare" in lowered:
                coordinate = 0.5
            elif any(
                cue in lowered
                for cue in (
                    "ready to select",
                    "select and start",
                    "select and deploy",
                    "select, configure",
                    "choose, configure",
                    "begin using",
                    "complete the relevant action now",
                )
            ):
                coordinate = 1.0
            else:
                coordinate = 0.0
            base = np.zeros((2, 4), dtype=np.float32)
            base[:, 0] = coordinate
            base[0, 1] = index / max(len(topics), 1)
            base[1, 2] = (index + 1) / (len(topics) + 1)
            pooled.append(
                np.array([coordinate, base[0, 1], base[1, 2]], dtype=np.float32)
            )
            states.append(base)
        return np.stack(pooled), np.stack(states)

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str:
        del max_new_tokens
        coordinate = float(np.asarray(state)[:, 0].mean())
        if coordinate < 0.25:
            purpose = "learn and understand the topic"
        elif coordinate < 0.75:
            purpose = "compare approaches and prepare for a possible action"
        else:
            purpose = "select and complete the relevant action now"
        return (
            f"Rank {{CANDIDATES}} for {{QUERY}} for a user who wants to {purpose}. "
            "Return exactly {TOP_N} candidate identifiers only, with no explanation."
        )


def _parse_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must contain numbers") from exc
    if not grid or any(not 0.0 <= item <= 1.0 for item in grid):
        raise argparse.ArgumentTypeError("grid values must lie in [0, 1]")
    return grid


def _load_endpoint_bank(path: Path) -> tuple[str, list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("endpoint_bank_version")
    pairs = payload.get("pairs")
    if not isinstance(version, str) or not version:
        raise ValueError("endpoint bank has no version")
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError("endpoint bank requires at least two pairs")
    normalized: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"endpoint pair {index} is not an object")
        row: dict[str, str] = {}
        for field in ("query", "informational_prompt", "transactional_prompt"):
            value = pair.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"endpoint pair {index} has invalid {field}")
            row[field] = value.strip()
        for field in ("informational_prompt", "transactional_prompt"):
            missing = [
                token
                for token in ("{QUERY}", "{CANDIDATES}", "{TOP_N}")
                if token not in row[field]
            ]
            if missing:
                raise ValueError(
                    f"endpoint pair {index} {field} lacks {', '.join(missing)}"
                )
        normalized.append(row)
    return version, normalized


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _decode_row(
    backend: ReconstructionBackend,
    *,
    path_kind: str,
    coordinate: float,
    state: np.ndarray,
    global_axis: DecodableAxis,
    max_new_tokens: int,
    pair_index: int | None = None,
    source_query: str | None = None,
    source_endpoint: str | None = None,
    source_prompt_template: str | None = None,
    probe_query: str | None = None,
) -> dict[str, object]:
    text = backend.decode(state, max_new_tokens=max_new_tokens)
    checks = decode_record_checks(text)
    injected = None
    injection_error = None
    if probe_query:
        try:
            injected = inject_query_after_decode(text, probe_query)
        except ValueError as exc:
            injection_error = str(exc)
    return {
        "path_kind": path_kind,
        "pair_index": pair_index,
        "source_query": source_query,
        "assigned_coordinate": coordinate,
        "observed_global_axis_coordinate": float(project_onto_axis(global_axis, state)),
        "latent_state_hash": stable_array_hash(state),
        "decoded_template": text,
        "structural_checks": checks,
        "probe_query": probe_query,
        "query_injected_after_decode": injected,
        "query_injection_error": injection_error,
        "source_endpoint": source_endpoint,
        "source_prompt_template": source_prompt_template,
    }


def _runtime_metadata() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for distribution in ("llm2vec-gen", "torch", "transformers", "peft", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": sys.version.split()[0],
        "dependency_versions": versions,
        "slurm": {
            "job_id": os.getenv("SLURM_JOB_ID"),
            "job_name": os.getenv("SLURM_JOB_NAME"),
            "partition": os.getenv("SLURM_JOB_PARTITION"),
            "account": os.getenv("SLURM_JOB_ACCOUNT"),
            "node_list": os.getenv("SLURM_JOB_NODELIST"),
            "cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.getenv("SLURM_JOB_GPUS"),
        },
        "offline_environment": {
            "hf_home": os.getenv("HF_HOME"),
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
            "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    return float((first @ second) / denominator)


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
    if len(left) != len(right) or len(left) < 2:
        return None
    first = _rankdata(left)
    second = _rankdata(right)
    if float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _path_cycle_diagnostic(
    rows: Sequence[dict[str, object]], *, path_id: str
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: float(row["assigned_coordinate"]))
    assigned = [float(row["assigned_coordinate"]) for row in ordered]
    reconstruction = [
        float(row["reencoded_reconstruction_axis_coordinate"]) for row in ordered
    ]
    pooled = [float(row["reencoded_pooled_axis_coordinate"]) for row in ordered]
    return {
        "path_id": path_id,
        "assigned_coordinates": assigned,
        "reencoded_reconstruction_coordinates": reconstruction,
        "reencoded_pooled_coordinates": pooled,
        "reconstruction_strictly_increasing": bool(
            all(right > left for left, right in zip(reconstruction, reconstruction[1:]))
        ),
        "pooled_strictly_increasing": bool(
            all(right > left for left, right in zip(pooled, pooled[1:]))
        ),
        "reconstruction_spearman": _spearman(assigned, reconstruction),
        "pooled_spearman": _spearman(assigned, pooled),
    }


def _report_markdown(
    *,
    diagnostics: dict[str, object],
    decoded_rows: Sequence[dict[str, object]],
    fake: bool,
) -> str:
    recon = diagnostics["reconstruction_geometry"]
    pooled = diagnostics["pooled_geometry"]
    cycle = diagnostics["decode_cycle"]
    monotonic_paths = sum(
        bool(path["reconstruction_strictly_increasing"])
        for path in cycle["path_monotonicity"]
    )
    path_count = len(cycle["path_monotonicity"])
    preserved = sum(
        bool(row["structural_checks"]["all_placeholders_preserved"])
        for row in decoded_rows
    )
    banner = (
        "> **Mock output only.** This run validates plumbing and supports no scientific "
        "claim.\n\n"
        if fake
        else ""
    )
    return f"""# LLM2Vec-Gen axis feasibility report

{banner}This run tests whether paired informational/transactional templates define a
topic-general direction in LLM2Vec-Gen and whether points in its reconstruction
state can still be decoded. It does not run reranking or estimate an effect.

## Geometry

- Reconstruction state shape: `{tuple(recon['state_shape'])}`
- Reconstruction leave-one-pair-out positive rate: `{recon['leave_one_pair_out_positive_rate']}`
- Reconstruction leave-one-pair-out mean cosine: `{recon['leave_one_pair_out_cosine_mean']}`
- Pooled leave-one-pair-out positive rate: `{pooled['leave_one_pair_out_positive_rate']}`
- Pooled leave-one-pair-out mean cosine: `{pooled['leave_one_pair_out_cosine_mean']}`

The leave-one-pair-out test is the main geometric check: every topic is tested
against a direction estimated without that topic. A positive result is necessary
but not sufficient for a valid semantic intervention.

## Decoding

- Decoded points: `{len(decoded_rows)}`
- Points preserving all three literal placeholders: `{preserved}/{len(decoded_rows)}`
- Query strategy: post-decode substitution only
- Decode-cycle paths increasing in reconstruction space: `{monotonic_paths}/{path_count}`
- Mean endpoint decode-cycle cosine: `{cycle['endpoint_reconstruction_cosine_mean']}`

Inspect `decoded_latent_grid.jsonl` for semantic monotonicity, fluency, unintended
criteria, and endpoint fidelity. Interpolation of reconstruction states is not a
validated capability of LLM2Vec-Gen, so geometry alone must not be reported as a
successful informational-to-transactional continuum.
"""


def main() -> int:
    started_at = _utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "local"), default="fake")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint-bank", default=str(DEFAULT_ENDPOINTS))
    parser.add_argument("--target-grid", type=_parse_grid, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument(
        "--decode-pairs",
        type=int,
        default=2,
        help="Number of topic-matched paths to decode in addition to the centroid path",
    )
    parser.add_argument("--probe-query", default=None)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.decode_pairs < 0:
        parser.error("decode-pairs must be nonnegative")
    if args.encode_batch_size <= 0 or args.encode_max_length <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and token limits must be positive")

    output_dir = Path(args.output_dir)
    targets = {
        "diagnostics": output_dir / "axis_diagnostics.json",
        "grid": output_dir / "decoded_latent_grid.jsonl",
        "state": output_dir / "axis_state.npz",
        "report": output_dir / "axis_feasibility_report.md",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(str(path) for path in existing))

    try:
        endpoint_path = Path(args.endpoint_bank).resolve()
        endpoint_version, pairs = _load_endpoint_bank(endpoint_path)
        backend: ReconstructionBackend
        if args.backend == "fake":
            backend = FakeReconstructionBackend()
        else:
            backend = LLM2VecGenBackend(args.model)

        informational_texts = [
            build_encoding_text(pair["informational_prompt"]) for pair in pairs
        ]
        transactional_texts = [
            build_encoding_text(pair["transactional_prompt"]) for pair in pairs
        ]
        pooled, reconstruction = backend.encode(
            [*informational_texts, *transactional_texts],
            batch_size=args.encode_batch_size,
            max_length=args.encode_max_length,
        )
        expected = len(pairs) * 2
        if pooled.ndim != 2 or pooled.shape[0] != expected:
            raise ValueError(f"unexpected pooled embedding shape: {pooled.shape}")
        if reconstruction.ndim != 3 or reconstruction.shape[0] != expected:
            raise ValueError(
                f"unexpected reconstruction state shape: {reconstruction.shape}"
            )
        if not np.isfinite(pooled).all() or not np.isfinite(reconstruction).all():
            raise ValueError("model returned non-finite representations")
    except (FileNotFoundError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    count = len(pairs)
    pooled_info, pooled_trans = pooled[:count], pooled[count:]
    recon_info, recon_trans = reconstruction[:count], reconstruction[count:]
    recon_axis = build_decodable_axis(recon_info, recon_trans)
    pooled_axis = build_decodable_axis(pooled_info, pooled_trans)

    decoded_rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        for source_endpoint, coordinate, state, source_prompt in (
            (
                "informational",
                0.0,
                recon_info[pair_index],
                pair["informational_prompt"],
            ),
            (
                "transactional",
                1.0,
                recon_trans[pair_index],
                pair["transactional_prompt"],
            ),
        ):
            decoded_rows.append(
                _decode_row(
                    backend,
                    path_kind="endpoint-reconstruction-control",
                    coordinate=coordinate,
                    state=state,
                    global_axis=recon_axis,
                    max_new_tokens=args.max_new_tokens,
                    pair_index=pair_index,
                    source_query=pair["query"],
                    source_endpoint=source_endpoint,
                    source_prompt_template=source_prompt,
                    probe_query=args.probe_query,
                )
            )
    for coordinate in args.target_grid:
        decoded_rows.append(
            _decode_row(
                backend,
                path_kind="global-centroid-stress-test",
                coordinate=coordinate,
                state=interpolate_axis_centroids(recon_axis, coordinate),
                global_axis=recon_axis,
                max_new_tokens=args.max_new_tokens,
                probe_query=args.probe_query,
            )
        )
    for pair_index, pair in enumerate(pairs[: args.decode_pairs]):
        for coordinate in args.target_grid:
            if coordinate in (0.0, 1.0):
                continue
            decoded_rows.append(
                _decode_row(
                    backend,
                    path_kind="topic-matched-pair",
                    coordinate=coordinate,
                    pair_index=pair_index,
                    source_query=pair["query"],
                    state=interpolate_endpoint_pair(
                        recon_info[pair_index], recon_trans[pair_index], coordinate
                    ),
                    global_axis=recon_axis,
                    max_new_tokens=args.max_new_tokens,
                    probe_query=args.probe_query,
                )
            )

    reencoded_pooled, reencoded_reconstruction = backend.encode(
        [build_encoding_text(str(row["decoded_template"])) for row in decoded_rows],
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    if (
        reencoded_pooled.shape[0] != len(decoded_rows)
        or reencoded_reconstruction.shape != (len(decoded_rows), *recon_axis.state_shape)
    ):
        parser.error(
            "decode-cycle encoding returned unexpected shapes: "
            f"pooled={reencoded_pooled.shape}, reconstruction={reencoded_reconstruction.shape}"
        )
    reencoded_recon_coordinates = project_onto_axis(
        recon_axis, reencoded_reconstruction
    )
    reencoded_pooled_coordinates = project_onto_axis(pooled_axis, reencoded_pooled)
    for index, row in enumerate(decoded_rows):
        row["reencoded_reconstruction_axis_coordinate"] = float(
            reencoded_recon_coordinates[index]
        )
        row["reencoded_pooled_axis_coordinate"] = float(
            reencoded_pooled_coordinates[index]
        )
        if row["path_kind"] == "endpoint-reconstruction-control":
            pair_index = int(row["pair_index"])
            source = (
                recon_info[pair_index]
                if row["source_endpoint"] == "informational"
                else recon_trans[pair_index]
            )
            row["decode_cycle_cosine_to_source_state"] = _cosine(
                source, reencoded_reconstruction[index]
            )
        else:
            row["decode_cycle_cosine_to_source_state"] = None

    path_cycles = [
        _path_cycle_diagnostic(
            [
                row
                for row in decoded_rows
                if row["path_kind"] == "global-centroid-stress-test"
            ],
            path_id="global-centroid-stress-test",
        )
    ]
    for pair_index in range(min(count, args.decode_pairs)):
        path_rows = [
            row
            for row in decoded_rows
            if (
                row["pair_index"] == pair_index
                and row["path_kind"]
                in ("endpoint-reconstruction-control", "topic-matched-pair")
            )
        ]
        path_cycles.append(
            _path_cycle_diagnostic(path_rows, path_id=f"topic-matched-pair-{pair_index}")
        )

    endpoint_cycle_cosines = [
        float(row["decode_cycle_cosine_to_source_state"])
        for row in decoded_rows
        if row["decode_cycle_cosine_to_source_state"] is not None
    ]

    diagnostics: dict[str, object] = {
        "diagnostic_version": "llm2vec-gen-axis-feasibility-v1",
        "generated_at": _utc_now(),
        "started_at": started_at,
        "git_commit_sha": _git_sha(),
        "status": "feasibility-only",
        "scientific_result": False,
        "axis_version": LLM2VEC_GEN_AXIS_VERSION,
        "axis_hash": recon_axis.axis_hash,
        "backend": backend.backend_name,
        "model": backend.model_name,
        "model_revision": (
            "not-applicable"
            if args.backend == "fake"
            else "unresolved library default; pin a local snapshot for confirmatory work"
        ),
        "reconstruction_dtype_for_decode": backend.reconstruction_dtype,
        "endpoint_bank": str(endpoint_path),
        "endpoint_bank_sha256": _sha256_file(endpoint_path),
        "endpoint_bank_version": endpoint_version,
        "endpoint_pair_count": count,
        "encoding_instruction_version": ENCODING_INSTRUCTION_VERSION,
        "runtime": _runtime_metadata(),
        "encoding_configuration": {
            "batch_size": args.encode_batch_size,
            "max_length": args.encode_max_length,
        },
        "decoding_configuration": {
            "method": "greedy (library default)",
            "max_new_tokens": args.max_new_tokens,
            "target_grid": list(args.target_grid),
            "decoded_topic_pair_count": min(count, args.decode_pairs),
        },
        "query_strategy": {
            "method": "post-decode literal placeholder substitution",
            "probe_query": args.probe_query,
            "probe_query_used_during_axis_estimation": False,
            "paired_endpoint_topics_used_during_axis_estimation": True,
            "paired_differences_are_used_to_reduce_topic_offsets": True,
            "vector_addition_tested": False,
        },
        "representation_hashes": {
            "pooled_informational": stable_array_hash(pooled_info),
            "pooled_transactional": stable_array_hash(pooled_trans),
            "reconstruction_informational": stable_array_hash(recon_info),
            "reconstruction_transactional": stable_array_hash(recon_trans),
        },
        "reconstruction_geometry": axis_geometry_diagnostics(recon_info, recon_trans),
        "pooled_geometry": axis_geometry_diagnostics(pooled_info, pooled_trans),
        "decode_cycle": {
            "method": (
                "decode, wrap generated template with the same encoding "
                "instruction, re-encode"
            ),
            "same_model_diagnostic_only": True,
            "path_monotonicity": path_cycles,
            "endpoint_reconstruction_cosine_mean": float(
                np.mean(endpoint_cycle_cosines)
            ),
            "endpoint_reconstruction_cosine_min": float(
                np.min(endpoint_cycle_cosines)
            ),
        },
        "interpretation": {
            "primary_geometry_check": "leave-one-pair-out signed gaps",
            "semantic_monotonicity_requires_manual_review": True,
            "centroid_path_is_a_manifold_stress_test": True,
            "mocked_runs_support_scientific_claims": False,
        },
    }

    _atomic_npz(
        targets["state"],
        {
            "informational_centroid": recon_axis.informational_centroid.astype(np.float32),
            "transactional_centroid": recon_axis.transactional_centroid.astype(np.float32),
            "direction_unit": recon_axis.direction_unit.astype(np.float32),
            "informational_endpoint_states": recon_info.astype(np.float32),
            "transactional_endpoint_states": recon_trans.astype(np.float32),
            "pooled_informational_endpoints": pooled_info.astype(np.float32),
            "pooled_transactional_endpoints": pooled_trans.astype(np.float32),
        },
    )
    _atomic_json(targets["diagnostics"], diagnostics)
    _atomic_jsonl(targets["grid"], decoded_rows)
    _atomic_text(
        targets["report"],
        _report_markdown(
            diagnostics=diagnostics,
            decoded_rows=decoded_rows,
            fake=args.backend == "fake",
        ),
    )
    print(f"wrote {targets['diagnostics']}")
    print(f"wrote {targets['grid']}")
    print(f"wrote {targets['state']}")
    print(f"wrote {targets['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
