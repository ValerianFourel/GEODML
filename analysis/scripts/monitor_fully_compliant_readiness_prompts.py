#!/usr/bin/env python3
"""Report fully compliant prompt counts from immutable verified rounds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.audit_fully_compliant_readiness_prompts import (
    audit_fully_compliant_prompts,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verified_rounds(pipeline_root: str | Path) -> list[Path]:
    root = Path(pipeline_root).resolve()
    rounds: list[tuple[int, Path]] = []
    for summary in root.glob("round-*/verified_round_summary.json"):
        try:
            index = int(summary.parent.name.removeprefix("round-"))
        except ValueError:
            continue
        rounds.append((index, summary.parent))
    return [path for _, path in sorted(rounds)]


def audit_new_verified_rounds(
    pipeline_root: str | Path,
    *,
    already_audited: Iterable[str] = (),
) -> list[dict[str, object]]:
    seen = set(already_audited)
    payloads: list[dict[str, object]] = []
    for round_root in verified_rounds(pipeline_root):
        resolved = str(round_root.resolve())
        if resolved in seen:
            continue
        report = audit_fully_compliant_prompts(
            round_root,
            maximum_failure_examples=5,
        )
        compliant = int(report["fully_compliant_prompt_count"])
        payloads.append(
            {
                "format_version": "readiness-live-fully-compliant-count-v1",
                "observed_at": _now(),
                "pipeline_root": str(Path(pipeline_root).resolve()),
                "round_root": resolved,
                "round_name": round_root.name,
                "audit_passed": bool(report["audit_passed"]),
                "fully_compliant_prompts": compliant,
                "ready_to_export": int(report["ready_to_export_count"]),
                "missing_from_30330": max(0, 30330 - compliant),
                "validated_candidates": int(report["validation_count"]),
                "independently_accepted_candidates": int(
                    report["independently_accepted_candidate_count"]
                ),
                "complete_30330_population": bool(
                    report["complete_30330_population_passed"]
                ),
                "failed_prompts": int(report["failed_prompt_count"]),
                "failed_global_checks": report["failed_global_checks"],
            }
        )
    return payloads


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_seen(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        round_root = str(row.get("round_root", ""))
        if round_root:
            seen.add(round_root)
    return seen


def _print_payload(payload: Mapping[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True), flush=True)
        return
    print(
        f"LIVE_AUDIT={'PASS' if payload['audit_passed'] else 'FAIL'} "
        f"round={payload['round_name']} "
        f"fully_compliant={payload['fully_compliant_prompts']} "
        f"ready_to_export={payload['ready_to_export']} "
        f"missing_from_30330={payload['missing_from_30330']} "
        f"validated_candidates={payload['validated_candidates']} "
        f"complete={'YES' if payload['complete_30330_population'] else 'NO'}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--history-file")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.poll_seconds <= 0:
        print("--poll-seconds must be positive", file=sys.stderr)
        return 2
    pipeline_root = Path(args.pipeline_root).resolve()
    if not pipeline_root.is_dir():
        print(f"pipeline root does not exist: {pipeline_root}", file=sys.stderr)
        return 2
    history = Path(args.history_file).resolve() if args.history_file else None
    seen = _load_seen(history)
    waiting_reported = False

    while True:
        try:
            payloads = audit_new_verified_rounds(
                pipeline_root,
                already_audited=seen,
            )
        except (OSError, ValueError) as exc:
            print(f"LIVE_AUDIT=WAIT error={exc}", file=sys.stderr, flush=True)
            payloads = []
        for payload in payloads:
            _print_payload(payload, as_json=args.json)
            if history is not None:
                _append_jsonl(history, payload)
            seen.add(str(payload["round_root"]))
        if not payloads and not seen and not waiting_reported:
            print("LIVE_AUDIT=WAIT verified_rounds=0", flush=True)
            waiting_reported = True
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
