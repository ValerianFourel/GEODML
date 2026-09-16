#!/usr/bin/env bash
# Submit with separately approved walltime and resources. No allocation defaults.
set -euo pipefail
: "${GEODML_EXECUTION_REPOSITORY:?Use the absolute pinned repository path}"
export GEODML_JUDGE_EXECUTION_MODE=throughput
exec bash "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/slurm/jupiter/run_nemotron_judge_pilot.sbatch"
