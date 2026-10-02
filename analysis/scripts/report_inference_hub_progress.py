#!/usr/bin/env python3
"""Count unique published inference outcomes at one private Hub revision.

This reads bundle manifests and checks their identities and registry joins. It
does not download result shards or establish row-level/scientific acceptance.
Registered-hour totals cover those hours, not the whole frozen population.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore, REGISTRY_PATH
from analysis.interpretability.pipeline.agentic_hours import digest
from analysis.interpretability.pipeline.agentic_task_ledger import identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.publish_qwen_results import INDEX_PATH, INDEX_VERSION

MODELS = ("qwen38", "llama4", "nemotron")


def collect(exchange: Exchange, revision: str | None = None) -> dict:
    revision = revision or exchange.store.head()
    issues = []
    scopes = {model: set() for model in MODELS}
    planned = {model: set() for model in MODELS}
    observations = defaultdict(dict)
    labels = defaultdict(set)
    conflicts = set()
    bundles = {}
    registry_claims = defaultdict(set)

    def issue(source, error):
        issues.append({"source": source, "error": str(error)})

    def document(path, version):
        try:
            raw = exchange.store.read(path, revision)
            if raw is None:
                raise ValueError("missing at the selected revision")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            if value.get("format_version") != version:
                raise ValueError("unsupported format_version")
            return value
        except (ValueError, TypeError, KeyError, OSError) as error:
            issue(path, error)
            return None

    def manifest(bundle):
        if bundle not in bundles:
            try:
                value = exchange.manifest(bundle, revision=revision)
                if not isinstance(value.get("outcomes"), dict) or not isinstance(value.get("files"), dict):
                    raise ValueError("bundle outcomes and files must be objects")
                valid = {}
                for fp, event in value["outcomes"].items():
                    if not isinstance(event, dict):
                        raise ValueError("outcome must be an object")
                    identity = ClaimIdentity(**event["identity"])
                    if identity_fingerprint(identity) != fp:
                        raise ValueError("outcome identity hash mismatch")
                    if event.get("state") not in {"completed", "terminal_failed"}:
                        raise ValueError("nonterminal outcome in bundle")
                    refs = event.get("record_references", [])
                    if not isinstance(refs, list) or (event["state"] == "completed" and not refs):
                        raise ValueError("completed outcome lacks record references")
                    if event["state"] == "terminal_failed" and (not event.get("owner_id") or not event.get("generation")):
                        raise ValueError("terminal failure lacks producer evidence")
                    for ref in refs:
                        if not isinstance(ref, dict):
                            raise ValueError("record reference must be an object")
                        stem = f"data/{ref['table']}/part-{ref['writer_id']}-{ref['shard_sequence']:06d}"
                        if any(stem + suffix not in value["files"] for suffix in (".jsonl", ".manifest.json")):
                            raise ValueError("record reference shard absent from bundle")
                    valid[fp] = event
                bundles[bundle] = valid
            except (ValueError, TypeError, KeyError, OSError) as error:
                issue(str(bundle), error)
                bundles[bundle] = None
        return bundles[bundle]

    def remember(model, fp, event):
        labels[fp].add(model)
        signature = digest({"state": event["state"], "record_references": event.get("record_references", [])})
        observations[fp][signature] = event["state"]
        if len(observations[fp]) > 1 or len(labels[fp]) > 1:
            conflicts.add(fp)

    index = document(INDEX_PATH, INDEX_VERSION)
    if index is not None:
        if index.get("model") != "qwen38" or not isinstance(index.get("bundles"), list):
            issue(INDEX_PATH, "invalid Qwen index structure")
            index = None
        else:
            scopes["qwen38"].add("qwen-results index")
            for entry in index["bundles"]:
                try:
                    if not isinstance(entry, dict):
                        raise ValueError("index bundle entry must be an object")
                    outcomes = manifest(entry["bundle"])
                    if outcomes is None:
                        continue
                    for fp, event in outcomes.items():
                        remember("qwen38", fp, event)
                    for state, field in (("completed", "completed"), ("terminal_failed", "failed")):
                        if sum(e["state"] == state for e in outcomes.values()) != entry[field]:
                            issue(entry["bundle"], "index counts differ from bundle outcomes")
                except (ValueError, TypeError, KeyError) as error:
                    issue(INDEX_PATH, error)

    registry = document(REGISTRY_PATH, "geodml-hours-v1")
    if registry is not None:
        if not isinstance(registry.get("hours"), dict):
            issue(REGISTRY_PATH, "invalid hours structure")
            registry = None
        else:
            for hour_id, hour in registry["hours"].items():
                try:
                    if not isinstance(hour, dict):
                        raise ValueError("registry hour must be an object")
                    model = hour["model"]
                    if model not in MODELS:
                        raise ValueError("unsupported model in registry")
                    for key in ("task_fingerprints", "completed", "failed", "checkpoints"):
                        if not isinstance(hour.get(key), list) or any(not isinstance(v, str) for v in hour[key]):
                            raise ValueError(f"registry {key} must be a list of strings")
                    scopes[model].add("registered hours")
                    selected = set(hour["task_fingerprints"])
                    planned[model].update(selected)
                    for fp in selected:
                        labels[fp].add(model)
                        if len(labels[fp]) > 1:
                            conflicts.add(fp)
                    claimed = {state: set(hour[field]) for state, field in
                               (("completed", "completed"), ("terminal_failed", "failed"))}
                    if any(not fps <= selected for fps in claimed.values()):
                        raise ValueError("registry outcomes outside the hour plan")
                    available = defaultdict(set)
                    for bundle in hour["checkpoints"]:
                        outcomes = manifest(bundle)
                        if outcomes is None:
                            continue
                        for fp in selected.intersection(outcomes):
                            event = outcomes[fp]
                            available[event["state"]].add(fp)
                            remember(model, fp, event)
                    for state, fps in claimed.items():
                        for fp in fps:
                            registry_claims[fp].add(state)
                        missing = fps - available[state]
                        if missing:
                            issue(hour_id, f"{len(missing)} {state} registry outcomes lack matching checkpoint evidence")
                    # A checkpoint terminal result must agree with its registry row.
                    for state, fps in available.items():
                        unrecorded = fps - claimed[state]
                        if unrecorded:
                            conflicts.update(unrecorded)
                            issue(hour_id, f"{len(unrecorded)} checkpoint outcomes disagree with the registry")
                except (ValueError, TypeError, KeyError) as error:
                    issue(str(hour_id), error)

    for fp, states in registry_claims.items():
        if len(states) > 1 or (observations.get(fp) and set(observations[fp].values()) != states):
            conflicts.add(fp)
    if conflicts:
        issue("outcomes", f"{len(conflicts)} conflicting fingerprints excluded from completed/failed counts")

    successful = {fp for fp, values in observations.items() if set(values.values()) == {"completed"}} - conflicts
    failed = {fp for fp, values in observations.items() if set(values.values()) == {"terminal_failed"}} - conflicts
    models = {}
    for model in MODELS:
        members = {fp for fp, names in labels.items() if model in names}
        present = bool(scopes[model])
        expected = planned[model] if "registered hours" in scopes[model] else None
        models[model] = {
            "status": ("partial" if issues else "counted") if present else "unavailable",
            "sources": sorted(scopes[model]),
            "completed": len(successful & members) if present else None,
            "terminal_failed": len(failed & members) if present else None,
            "conflicts": len(conflicts & members) if present else None,
            "expected_total": None,
            "registry_expected": len(expected) if expected is not None else None,
            "registry_unresolved": len(expected - successful - failed) if expected is not None else None,
        }
    generators = {fp for fp, names in labels.items() if names & {"llama4", "qwen38"}}
    both_generators = all(scopes[model] for model in ("llama4", "qwen38"))
    generator_evidence = (any(observations.get(fp) for fp in generators)
                          or (index is not None and not index["bundles"])
                          or (registry is not None and not registry["hours"]))
    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(), "revision": revision,
        "status": "partial" if issues else "counted", "models": models,
        "generator_totals": {
            "status": "counted" if both_generators and not issues else "partial",
            "completed": len(generators & successful) if generator_evidence else None,
            "terminal_failed": len(generators & failed) if generator_evidence else None,
            "conflicts": len(generators & conflicts), "expected_total": None,
            "counts_are_lower_bounds": not (both_generators and not issues),
        },
        "bundle_manifests_checked": sum(value is not None for value in bundles.values()),
        "issues": issues,
        "scope": "Published outcome manifests; registered-hour denominator excludes unregistered/deferred work.",
        "verification": "Bundle content hashes, identity hashes and registry joins; result shard bytes/rows not downloaded.",
        "scientific_acceptance_established": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Read-only methods never use Exchange's mutation journal.
    report = collect(Exchange(HubStore(args.repo_id), Path(".")), args.revision)
    report["repo_id"] = args.repo_id
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "counted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
