"""Update a CPU-only axis-to-permutation report from verified saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.interpretability.pipeline.axis_permutation_inputs import load_study
from analysis.interpretability.pipeline.axis_permutation_report import (
    analysis_settings,
    build_report,
    save_report,
)


def _verify_preparation(config, base, loaded_sources):
    preparation = config.get("preparation")
    if preparation is None:
        return
    if preparation.get("format_version") != "axis-permutation-report-preparation-v1":
        raise ValueError("unsupported report preparation format")
    loaded = {row["path"]: row["sha256"] for row in loaded_sources}
    for artifact in preparation["verified_artifacts"]:
        path = (base / artifact["path"]).resolve()
        actual = loaded.get(str(path))
        if actual is None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError(f"prepared artifact hash mismatch: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Frozen analysis settings and source paths (JSON).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dedicated report directory, separate from inference outputs.")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    analysis_settings(config)
    sources = config.get("sources")
    if not isinstance(sources, dict) or not any(sources.values()):
        raise ValueError("config.sources must name at least one saved plan or generator source")
    if set(sources) - {"direct", "agentic", "judges"}:
        raise ValueError("unknown source kind")
    study = load_study(sources, config_path.parent)
    _verify_preparation(config, config_path.parent, study.get("sources", []))
    output = args.output_dir.resolve()
    input_parents = {Path(source["path"]).resolve().parent for source in study.get("sources", [])}
    if output in input_parents:
        raise ValueError("use a dedicated report directory, not a source artifact directory")
    report = build_report(study, config)
    if "preparation" in config:
        report["preparation"] = config["preparation"]
    destination = save_report(report, output)
    print("REPORT=" + str(destination / "report.md"))
    print("LATEST=" + str(output / "latest.json"))
    print("GENERATION=" + json.dumps(report["generation"], sort_keys=True))
    print("JUDGING=" + json.dumps(report["judging"], sort_keys=True))
    print("FIT_STATUS=" + json.dumps(report["fit_status_counts"], sort_keys=True))
    print(f"INPUT_ISSUES={len(report['issues'])}")
    print("SCIENTIFIC_RESULT=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
