#!/usr/bin/env python3
"""Generate the Milestone 3A policy-clause request or candidate pilot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.policy_clause_bank import (  # noqa: E402
    FakePolicyClauseProvider,
    RepositoryLocalPolicyClauseProvider,
    default_generation_parameters,
)
from interpretability.pipeline.policy_clause_pilot import (  # noqa: E402
    write_policy_clause_pilot,
)

DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "output" / "policy_clause_pilot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "generate"), default="dry-run")
    parser.add_argument("--provider", choices=("fake", "local"), default="fake")
    parser.add_argument(
        "--model",
        default=os.getenv("POLICY_GENERATOR_MODEL", "fake-policy-provider-v1"),
    )
    parser.add_argument("--precision", choices=("full", "4bit"), default="full")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--number-style-seeds", type=int, default=8)
    parser.add_argument("--first-style-seed", type=int, default=0)
    parser.add_argument("--number-b-values", type=int, default=8)
    parser.add_argument("--master-seed", type=int, default=20260810)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--prompt-space-version", default="hybrid-pilot-v1")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-clause-chars", type=int, default=420)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    provider = None
    if args.mode == "generate":
        if args.provider == "fake":
            provider = FakePolicyClauseProvider()
        else:
            provider = RepositoryLocalPolicyClauseProvider.from_model(
                args.model, precision=args.precision
            )
    parameters = default_generation_parameters(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        precision=args.precision,
    )
    try:
        artifacts = write_policy_clause_pilot(
            args.output_dir,
            mode=args.mode,
            provider=provider,
            generator_model=args.model,
            number_style_seeds=args.number_style_seeds,
            first_style_seed=args.first_style_seed,
            number_bias_values=args.number_b_values,
            master_seed=args.master_seed,
            include_anchors=args.include_anchors,
            top_n=args.top_n,
            prompt_space_version=args.prompt_space_version,
            generation_parameters=parameters,
            max_clause_chars=args.max_clause_chars,
            overwrite=args.overwrite,
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Requests: {len(artifacts.requests)}")
    print(f"Request manifest: {artifacts.requests_path}")
    if artifacts.candidates_path is not None:
        print(f"Accepted candidates: {len(artifacts.candidates)}")
        print(f"Candidate clauses: {artifacts.candidates_path}")
        print(f"Candidate prompts: {artifacts.full_prompts_path}")
        print(f"Pilot report: {artifacts.report_path}")
    else:
        print("Dry run complete; no provider was invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
