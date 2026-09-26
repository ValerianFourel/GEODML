"""End-to-end contract for the saved-artifact latent/ranking report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.report_latent_ranking_relationship import (
    build_report,
    write_report,
)


def _fixture(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="policy")
    registration = root / "local-only" / "population-registration-v1"
    registration.mkdir(parents=True)
    coordinate_path = registration / "population-selection-records.jsonl"
    coordinate_rows = []
    writer = FinalDatasetWriter(root, writer_id="fixture")
    ledger = StripedTaskLedger(root / "control" / "task-ledger", stripe_count=4)
    pending = []
    for index in range(8):
        prompt_id = f"prompt-{index}"
        coordinate = index / 7
        coordinate_rows.append({
            "candidate_id": prompt_id,
            "axis_bin": index,
            "assigned_axis_1_0_1": coordinate,
            "observed_axis_1_percentile_0_1": coordinate,
        })
        writer.append(
            "keyword_memberships",
            {
                "prompt_id": prompt_id,
                "keyword_ids": ["keyword"],
                "primary_keyword_id": "keyword",
                "primary_priority_rank": 0,
            },
            transaction_id=f"membership-{index}",
        )
        for condition in ("natural", "shuffled"):
            task_id = f"task-{index}-{condition}"
            identity = ClaimIdentity(
                task_id=task_id,
                model_id="qwen/model",
                model_revision="a" * 40,
                protocol="agentic-generator-shared-v1",
                request_sha256=hashlib.sha256(task_id.encode()).hexdigest(),
            )
            writer.append(
                "task_definitions",
                {
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "model": "qwen38",
                    "method": "Parallel-Expansion-v1",
                    "engine": "duckduckgo",
                    "condition": condition,
                    "claim_identity": asdict(identity),
                    "dependency_fingerprints": [],
                    "runnable_task": {"cell_id": task_id},
                },
                transaction_id=f"definition-{task_id}",
            )
            claim = ledger.claim(identity, owner_id="fixture").claim
            target = f"https://target.example/{index}"
            other = f"https://other.example/{index}"
            ranking = (
                [target, other]
                if condition == "natural" and coordinate >= 0.5
                else [other, target]
            )
            reference = writer.append(
                "generations",
                {
                    "cell_id": task_id,
                    "ranking": ranking,
                    "condition_audit": {
                        "target_url": target,
                        "target_observed": True,
                    },
                },
                transaction_id=task_id,
            )
            pending.append((claim, reference))
    writer.seal()
    for claim, reference in pending:
        ledger.transition(claim, state="completed", record_references=[reference])
    raw = b"".join(
        json.dumps(row, sort_keys=True).encode() + b"\n" for row in coordinate_rows
    )
    coordinate_path.write_bytes(raw)
    manifest_path = registration / "manifest.json"
    manifest_path.write_text(json.dumps({
        "files": {
            coordinate_path.name: {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        }
    }))
    return root, coordinate_path, manifest_path


def test_report_joins_verified_rankings_and_separates_observed_axis(tmp_path):
    root, coordinates, manifest = _fixture(tmp_path)
    report = build_report(
        root,
        coordinates,
        coordinate_manifest=manifest,
        models=["qwen38"],
        axis_construct="fixture-readiness",
        permutations=49,
        seed=7,
        stripe_count=4,
    )

    assert report["observation_count"] == 16
    assert report["natural_shuffled_pair_count"] == 8
    assert report["accounting"]["verified_completed_tasks"] == {"qwen38": 16}
    target = next(
        row for row in report["task_associations"]
        if row["coordinate_role"] == "observed_latent"
        and row["condition"] == "natural"
        and row["outcome"] == "target_reciprocal_rank_if_seen"
    )
    order = next(
        row for row in report["permutation_associations"]
        if row["coordinate_role"] == "observed_latent"
        and row["outcome"] == "natural_vs_shuffled_top1_change"
    )
    assert target["spearman_rho"] > 0.8
    assert order["spearman_rho"] > 0.8
    assert target["permutable_keywords"] == 1
    first_bin = report["permutation_bin_summaries"][0]
    last_bin = report["permutation_bin_summaries"][-1]
    assert first_bin["mean_natural_vs_shuffled_top1_change"] == 0.0
    assert last_bin["mean_natural_vs_shuffled_top1_change"] == 1.0
    assert report["scientific_result"] is False

    output = write_report(report, tmp_path / "report")
    assert json.loads((output / "report.json").read_text()) == report
    assert "Ranking change along the observed latent axis" in (
        output / "report.md"
    ).read_text()
    assert (output / "associations.csv").read_text().count("observed_latent") > 1
    bins = (output / "permutation-bin-summary.csv").read_text()
    assert "mean_natural_vs_shuffled_top1_change" in bins
