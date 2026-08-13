#!/usr/bin/env python3
"""Build a query-bound randomized complete-block panel from a frozen A1 manifold."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.a1_prompt_manifold import SelectedA1Prompt  # noqa: E402
from interpretability.pipeline.a1_query_panel import (  # noqa: E402
    A1_QUERY_PANEL_VERSION,
    build_query_conditioned_a1_panel,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifold", required=True)
    parser.add_argument(
        "--source-run-manifest",
        help="Optional run_manifest.json from the selected A1 manifold run.",
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--serp-parquet",
        help="Canonical SERP parquet; distinct values from --keyword-column are blocks.",
    )
    sources.add_argument(
        "--keywords-file",
        help="UTF-8 text file containing one search term per non-empty line.",
    )
    parser.add_argument("--keyword-column", default="keyword")
    parser.add_argument(
        "--expected-keywords",
        type=int,
        default=None,
        help="Fail unless this exact number of distinct normalized terms is loaded.",
    )
    parser.add_argument("--master-seed", type=int, default=20260817)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        output = Path(args.output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        selected_path = Path(args.selected_manifold).resolve()
        prompts = tuple(
            SelectedA1Prompt(**row) for row in _read_jsonl(selected_path)
        )
        search_terms, source = _load_search_terms(args)
        if args.expected_keywords is not None and len(search_terms) != args.expected_keywords:
            raise ValueError(
                f"expected {args.expected_keywords} search terms, loaded {len(search_terms)}"
            )
        panel, diagnostics = build_query_conditioned_a1_panel(
            search_terms=search_terms,
            selected_prompts=prompts,
            master_seed=args.master_seed,
        )
        source_manifest = None
        if args.source_run_manifest:
            source_manifest_path = Path(args.source_run_manifest).resolve()
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source["source_run_manifest"] = str(source_manifest_path)
            source["source_run_manifest_sha256"] = _sha256_file(source_manifest_path)

        manifest = {
            "artifact_version": A1_QUERY_PANEL_VERSION,
            "scientific_result": False,
            "status": "scheduled-unrun",
            "design": "randomized-complete-block",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "git_commit_sha": _git_sha(),
            "master_seed": args.master_seed,
            "source_selected_manifold": str(selected_path),
            "source_selected_manifold_sha256": _sha256_file(selected_path),
            "source_manifold_git_commit_sha": (
                source_manifest.get("git_commit_sha") if source_manifest else None
            ),
            "source_manifold_status": (
                source_manifest.get("status") if source_manifest else None
            ),
            "source": source,
            "query_count": diagnostics.query_count,
            "prompts_per_query": diagnostics.prompts_per_query,
            "assignment_count": diagnostics.assignment_count,
            "a1_levels": diagnostics.a1_levels,
            "style_seeds": diagnostics.style_seeds,
            "treatment": "assigned_a1",
            "surface_factor": "style_seed",
            "blocking_variable": "search_term",
            "randomized_fields": ["keyword_order", "within_keyword_order"],
            "candidate_sets_bound": False,
            "outcomes_observed": False,
        }
        report = _report(manifest, asdict(diagnostics))

        output.mkdir(parents=True)
        _atomic_jsonl(
            output / "a1_query_prompt_panel.jsonl",
            (asdict(row) for row in panel),
        )
        _atomic_json(output / "a1_query_panel_diagnostics.json", asdict(diagnostics))
        _atomic_json(output / "run_manifest.json", manifest)
        _atomic_text(output / "a1_query_panel_report.md", report)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"scheduled {diagnostics.assignment_count} query-bound A1 prompts: "
        f"{diagnostics.prompts_per_query} prompts x {diagnostics.query_count} search terms"
    )
    print(f"output: {output}")
    return 0


def _load_search_terms(args) -> tuple[tuple[str, ...], dict[str, object]]:
    if args.keywords_file:
        path = Path(args.keywords_file).resolve()
        terms = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return terms, {
            "type": "keywords-text",
            "path": str(path),
            "sha256": _sha256_file(path),
        }

    import pandas as pd

    path = Path(args.serp_parquet).resolve()
    frame = pd.read_parquet(path, columns=[args.keyword_column])
    if args.keyword_column not in frame.columns:
        raise ValueError(f"SERP table has no {args.keyword_column!r} column")
    terms = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in frame[args.keyword_column].tolist()
            if value is not None and str(value).strip()
        )
    )
    return terms, {
        "type": "serp-parquet",
        "path": str(path),
        "sha256": _sha256_file(path),
        "keyword_column": args.keyword_column,
        "row_count": len(frame),
    }


def _report(manifest: dict[str, object], diagnostics: dict[str, object]) -> str:
    return "\n".join(
        (
            "# Query-conditioned semantic A1 randomized panel",
            "",
            f"- Design: `{manifest['design']}`",
            f"- Search-term blocks: `{diagnostics['query_count']}`",
            f"- Prompts per search term: `{diagnostics['prompts_per_query']}`",
            f"- Total scheduled assignments: `{diagnostics['assignment_count']}`",
            f"- Assigned A1 levels: `{diagnostics['a1_levels']}`",
            f"- Surface seeds: `{diagnostics['style_seeds']}`",
            f"- Exact query binding rate: `{diagnostics['exact_query_binding_rate']}`",
            f"- Complete block rate: `{diagnostics['complete_block_rate']}`",
            "",
            "`assigned_a1` is the semantic treatment. `style_seed` is a surface-realization",
            "factor, and `search_term` is the blocking variable. Every search term receives",
            "the same complete 7 x 4 manifold. Seeded keyword and within-keyword execution",
            "orders are randomized; treatment membership is not thinned or regenerated.",
            "",
            "This artifact is a pre-outcome schedule. Candidate sets and ranking outcomes",
            "have not been bound, so it is not a scientific result.",
            "",
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
