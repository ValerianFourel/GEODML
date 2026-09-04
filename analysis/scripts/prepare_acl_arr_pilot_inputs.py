#!/usr/bin/env python3
"""Select a deterministic pilot and extract page text from a cached HTML run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from analysis.interpretability.pipeline.features import _pick_html_run_id  # noqa: E402
from analysis.interpretability.utils import HTMLLoader, extract_passage  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_serp(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(_read_jsonl(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--serp", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pilot-size", type=int, default=128)
    parser.add_argument("--master-seed", type=int, default=20260904)
    parser.add_argument("--engine", default="searxng")
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--minimum-documents", type=int, default=11)
    parser.add_argument("--max-document-characters", type=int, default=12000)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.pilot_size <= 0 or args.pool <= 0 or args.minimum_documents <= 0:
        raise SystemExit("pilot-size, pool, and minimum-documents must be positive")
    if args.minimum_documents > args.pool or args.max_document_characters <= 0:
        raise SystemExit("document limits are invalid")

    audit_root = Path(args.audit_root).resolve()
    serp_path = Path(args.serp).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompt_output = output / "pilot-prompts.jsonl"
    axis_output = output / "pilot-axis.jsonl"
    serp_output = output / "pilot-serp.jsonl"
    page_output = output / "pilot-page-text.jsonl"
    manifest_output = output / "pilot-input-manifest.json"
    artifacts = (prompt_output, axis_output, serp_output, page_output, manifest_output)
    if all(path.exists() and path.stat().st_size > 0 for path in artifacts):
        print("PILOT_INPUTS=ALREADY_COMPLETE")
        print(f"MANIFEST={manifest_output}")
        return 0
    if any(path.exists() for path in artifacts):
        raise SystemExit("partial prepared inputs exist; inspect them before retrying")

    try:
        serp = _read_serp(serp_path)
        required = {"keyword", "position", "title", "url", "snippet"}
        missing = required.difference(serp.columns)
        if missing:
            raise ValueError(f"SERP is missing columns: {sorted(missing)}")
        serp = serp.sort_values(["keyword", "position"])
        serp = serp.groupby("keyword", sort=False).head(args.pool).copy()
        html_run = _pick_html_run_id(data_root, args.engine, args.pool)
        if html_run is None:
            raise ValueError(
                f"no cached {args.engine} top-{args.pool} HTML run is available"
            )

        page_rows: list[dict[str, object]] = []
        with HTMLLoader(html_run, root=data_root) as loader:
            urls = sorted({str(value) for value in serp["url"].dropna() if str(value)})
            for index, url in enumerate(urls, 1):
                text = extract_passage(
                    loader.get_html(url), max_chars=args.max_document_characters
                )
                if text:
                    page_rows.append({"url": url, "text": text})
                if index % 250 == 0 or index == len(urls):
                    print(
                        f"PAGE_TEXT_PROGRESS={index}/{len(urls)} "
                        f"VALID={len(page_rows)}",
                        flush=True,
                    )

        page_urls = {str(row["url"]) for row in page_rows}
        available = serp[serp["url"].astype(str).isin(page_urls)].drop_duplicates(
            ["keyword", "url"]
        )
        counts = available.groupby("keyword")["url"].nunique()
        eligible_keywords = set(
            counts[counts >= args.minimum_documents].index.astype(str)
        )
        prompts = _read_jsonl(audit_root / "compliant-candidates.jsonl")
        axis = _read_jsonl(audit_root / "final-axis-map.jsonl")
        axis_by_id = {str(row["candidate_id"]): row for row in axis}
        eligible_prompts: list[tuple[str, str, dict[str, object]]] = []
        for row in prompts:
            candidate_id = str(row["candidate_id"])
            keyword = str(row.get("keyword", ""))
            if keyword in eligible_keywords and candidate_id in axis_by_id:
                digest = hashlib.sha256(
                    f"{args.master_seed}\0{candidate_id}".encode()
                ).hexdigest()
                eligible_prompts.append((digest, candidate_id, row))
        eligible_prompts.sort()
        selected = [item[2] for item in eligible_prompts[: args.pilot_size]]
        if len(selected) != args.pilot_size:
            raise ValueError(
                f"only {len(selected)} eligible prompts are available; "
                f"requested {args.pilot_size}"
            )

        selected_ids = {str(row["candidate_id"]) for row in selected}
        selected_keywords = {str(row["keyword"]) for row in selected}
        selected_axis = [axis_by_id[str(row["candidate_id"])] for row in selected]
        selected_serp = available[
            available["keyword"].astype(str).isin(selected_keywords)
        ]
        selected_serp = selected_serp.sort_values(["keyword", "position"])
        selected_serp = selected_serp.groupby("keyword", sort=False).head(args.pool)
        selected_urls = set(selected_serp["url"].astype(str))
        selected_pages = [row for row in page_rows if str(row["url"]) in selected_urls]
        if len(selected_ids) != args.pilot_size:
            raise ValueError("selected pilot contains duplicate candidate IDs")

        _write_jsonl_atomic(prompt_output, selected)
        _write_jsonl_atomic(axis_output, selected_axis)
        _write_jsonl_atomic(page_output, selected_pages)
        _write_jsonl_atomic(serp_output, selected_serp.to_dict(orient="records"))
        manifest = {
            "format_version": "acl-arr-pilot-inputs-v1",
            "pilot_size": args.pilot_size,
            "master_seed": args.master_seed,
            "engine": args.engine,
            "pool": args.pool,
            "minimum_documents": args.minimum_documents,
            "max_document_characters": args.max_document_characters,
            "html_run": html_run,
            "eligible_keyword_count": len(eligible_keywords),
            "selected_keyword_count": len(selected_keywords),
            "selected_serp_row_count": len(selected_serp),
            "selected_page_count": len(selected_pages),
        }
        temporary_manifest = manifest_output.with_suffix(".tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_output)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"HTML_RUN={html_run}")
    print(f"PILOT_PROMPTS={len(selected)}")
    print(f"PILOT_KEYWORDS={len(selected_keywords)}")
    print(f"PILOT_SERP_ROWS={len(selected_serp)}")
    print(f"PILOT_PAGE_ROWS={len(selected_pages)}")
    print(f"MANIFEST={manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
