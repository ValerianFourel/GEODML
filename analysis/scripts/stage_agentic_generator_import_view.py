#!/usr/bin/env python3
"""Copy completed generator metadata into a disposable claim-import view."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from analysis.interpretability.pipeline.inference_wave import _atomic_json


def stage(source: Path, output: Path) -> int:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite import view: {output}")
    for name in ("config.json", "run_manifest.json"):
        if not (source / name).is_file():
            raise ValueError(f"source output lacks {name}")
    output.mkdir(parents=True)
    shutil.copy2(source / "config.json", output / "config.json")
    shutil.copy2(source / "run_manifest.json", output / "run_manifest.json")
    count = 0
    for result_path in sorted((source / "results").glob("*.json")):
        result = json.loads(result_path.read_text())
        trace_path = Path(result.get("trace", ""))
        diagnostic_path = source / "diagnostics" / result_path.name
        if not trace_path.is_file() or not diagnostic_path.is_file():
            raise ValueError(
                f"completed source cell lacks trace or diagnostics: {result_path}"
            )
        target_trace = output / "traces" / result_path.name
        target_trace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, target_trace)
        target_diagnostic = output / "diagnostics" / result_path.name
        target_diagnostic.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(diagnostic_path, target_diagnostic)
        provenance = source / "provenance" / result_path.name
        if provenance.is_file():
            target = output / "provenance" / result_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(provenance, target)
        result["trace"] = str(target_trace.resolve())
        (output / "results").mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "results" / result_path.name, result)
        count += 1
    if count == 0:
        raise ValueError("source output has no completed results")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"STAGED_COMPLETED_CELLS={stage(args.source, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
