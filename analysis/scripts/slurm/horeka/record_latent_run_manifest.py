#!/usr/bin/env python3
"""Atomically record provenance for a HoreKa latent-prompt pilot job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile


CONFIGURATION_ENV = {
    "data_root": "GEODML_DATA_ROOT",
    "generator_model": "PROMPT_GENERATOR_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_device": "EMBEDDING_DEVICE",
    "precision": "PROMPT_GENERATOR_PRECISION",
    "engine": "LATENT_PROMPT_ENGINE",
    "candidate_pool": "LATENT_PROMPT_POOL",
    "top_n": "LATENT_PROMPT_TOP_N",
    "max_keywords": "LATENT_PROMPT_MAX_KEYWORDS",
    "target_grid": "LATENT_PROMPT_TARGET_GRID",
    "number_style_seeds": "LATENT_PROMPT_STYLE_SEEDS",
    "first_style_seed": "LATENT_PROMPT_FIRST_STYLE_SEED",
    "number_candidates": "LATENT_PROMPT_CANDIDATES",
    "master_seed": "LATENT_PROMPT_MASTER_SEED",
    "output_dir": "LATENT_PROMPT_PILOT_OUTPUT",
}

SLURM_ENV = {
    "job_id": "SLURM_JOB_ID",
    "job_name": "SLURM_JOB_NAME",
    "account": "SLURM_JOB_ACCOUNT",
    "partition": "SLURM_JOB_PARTITION",
    "node_list": "SLURM_JOB_NODELIST",
    "cpus_per_task": "SLURM_CPUS_PER_TASK",
    "gpus": "SLURM_JOB_GPUS",
    "gres": "SLURM_JOB_GRES",
    "submit_dir": "SLURM_SUBMIT_DIR",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", choices=("started", "complete", "failed"), required=True)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()

    repository = Path(os.environ.get("SLURM_SUBMIT_DIR", ".")).resolve()
    payload: dict[str, object] = {
        "manifest_version": "horeka-latent-prompt-run-v1",
        "git_commit_sha": _git_commit(repository),
        "status": args.status,
        "exit_code": args.exit_code,
        "started_at": os.environ.get("LATENT_PROMPT_RUN_STARTED_AT"),
        "ended_at": _utc_now() if args.status != "started" else None,
        "configuration": {
            key: os.environ.get(environment_name)
            for key, environment_name in CONFIGURATION_ENV.items()
        },
        "environment": {
            "venv": os.environ.get("GEODML_VENV"),
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "modules": os.environ.get("HOREKA_MODULES"),
        },
        "slurm": {
            key: os.environ.get(environment_name)
            for key, environment_name in SLURM_ENV.items()
        },
        "logs": {
            "stdout": f"logs/geodml-latent-prompts-{os.environ.get('SLURM_JOB_ID', 'unknown')}.out",
            "stderr": f"logs/geodml-latent-prompts-{os.environ.get('SLURM_JOB_ID', 'unknown')}.err",
        },
    }
    _atomic_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
