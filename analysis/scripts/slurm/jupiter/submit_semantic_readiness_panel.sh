#!/bin/bash
# Submit the three predeclared, distinct-family Phase-2 judge slots on JUPITER.

set -euo pipefail
umask 077

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

: "${READINESS_RUN:?Set READINESS_RUN}"
: "${READINESS_JUDGE_TASKS:?Set READINESS_JUDGE_TASKS}"
: "${READINESS_JUDGE_TASKS_SHA256:?Set READINESS_JUDGE_TASKS_SHA256}"
: "${READINESS_EXPECTED_TASKS_PER_SLOT:?Set READINESS_EXPECTED_TASKS_PER_SLOT}"
: "${GEODML_EXPECTED_COMMIT:?Set GEODML_EXPECTED_COMMIT}"
: "${PRIMARY_JUDGE_FAMILY:?Set PRIMARY_JUDGE_FAMILY}"
: "${PRIMARY_JUDGE_MODEL:?Set PRIMARY_JUDGE_MODEL}"
: "${PRIMARY_JUDGE_REVISION:?Set PRIMARY_JUDGE_REVISION}"
: "${REPLICATE_A_JUDGE_FAMILY:?Set REPLICATE_A_JUDGE_FAMILY}"
: "${REPLICATE_A_JUDGE_MODEL:?Set REPLICATE_A_JUDGE_MODEL}"
: "${REPLICATE_A_JUDGE_REVISION:?Set REPLICATE_A_JUDGE_REVISION}"
: "${REPLICATE_B_JUDGE_FAMILY:?Set REPLICATE_B_JUDGE_FAMILY}"
: "${REPLICATE_B_JUDGE_MODEL:?Set REPLICATE_B_JUDGE_MODEL}"
: "${REPLICATE_B_JUDGE_REVISION:?Set REPLICATE_B_JUDGE_REVISION}"

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "Commit mismatch: expected $GEODML_EXPECTED_COMMIT, found $actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to submit from a dirty worktree" >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$READINESS_JUDGE_TASKS" ]]; then
    echo "Missing task bank: $READINESS_JUDGE_TASKS" >&2
    exit 2
fi
actual_task_sha256="$(sha256sum "$READINESS_JUDGE_TASKS" | awk '{print $1}')"
if [[ "$actual_task_sha256" != "$READINESS_JUDGE_TASKS_SHA256" ]]; then
    echo "Task-bank hash mismatch: expected $READINESS_JUDGE_TASKS_SHA256, found $actual_task_sha256" >&2
    exit 2
fi

models=(
    "$PRIMARY_JUDGE_MODEL"
    "$REPLICATE_A_JUDGE_MODEL"
    "$REPLICATE_B_JUDGE_MODEL"
)
families=(
    "$PRIMARY_JUDGE_FAMILY"
    "$REPLICATE_A_JUDGE_FAMILY"
    "$REPLICATE_B_JUDGE_FAMILY"
)
normalized_families=(
    "${PRIMARY_JUDGE_FAMILY,,}"
    "${REPLICATE_A_JUDGE_FAMILY,,}"
    "${REPLICATE_B_JUDGE_FAMILY,,}"
)
revisions=(
    "$PRIMARY_JUDGE_REVISION"
    "$REPLICATE_A_JUDGE_REVISION"
    "$REPLICATE_B_JUDGE_REVISION"
)
if [[ "${models[0]}" == "${models[1]}" || "${models[0]}" == "${models[2]}" || "${models[1]}" == "${models[2]}" ]]; then
    echo "The three judge slots must use distinct model snapshots" >&2
    exit 2
fi
if [[ "${normalized_families[0]}" == "${normalized_families[1]}" || "${normalized_families[0]}" == "${normalized_families[2]}" || "${normalized_families[1]}" == "${normalized_families[2]}" ]]; then
    echo "The three judge slots must use distinct model families" >&2
    exit 2
fi
for model in "${models[@]}"; do
    if [[ ! -f "$model/config.json" ]]; then
        echo "Missing model snapshot: $model" >&2
        exit 2
    fi
done

mkdir -p "$READINESS_RUN/logs" "$READINESS_RUN/submissions"
panel_tsv="$READINESS_RUN/submissions/judge-panel.tsv"
candidate_panel="$(mktemp "$READINESS_RUN/submissions/.judge-panel.XXXXXX")"
trap 'rm -f "$candidate_panel"' EXIT
printf '# git_commit_sha=%s\n' "$GEODML_EXPECTED_COMMIT" > "$candidate_panel"
printf '# task_file_sha256=%s\n' "$READINESS_JUDGE_TASKS_SHA256" >> "$candidate_panel"
printf '# expected_tasks_per_slot=%s\n' "$READINESS_EXPECTED_TASKS_PER_SLOT" >> "$candidate_panel"
printf 'judge_slot\tjudge_tag\tmodel_family\tmodel\trevision\n' >> "$candidate_panel"
printf 'primary-frontier\tprimary-frontier\t%s\t%s\t%s\n' "${families[0]}" "${models[0]}" "${revisions[0]}" >> "$candidate_panel"
printf 'replicate-frontier-a\treplicate-frontier-a\t%s\t%s\t%s\n' "${families[1]}" "${models[1]}" "${revisions[1]}" >> "$candidate_panel"
printf 'replicate-frontier-b\treplicate-frontier-b\t%s\t%s\t%s\n' "${families[2]}" "${models[2]}" "${revisions[2]}" >> "$candidate_panel"
if [[ -f "$panel_tsv" ]] && ! cmp -s "$candidate_panel" "$panel_tsv"; then
    echo "Refusing to change the frozen judge panel: $panel_tsv" >&2
    exit 2
fi
if [[ ! -f "$panel_tsv" ]]; then
    mv "$candidate_panel" "$panel_tsv"
fi

submit_one() {
    local slot="$1"
    local tag="$2"
    local family="$3"
    local model="$4"
    local revision="$5"
    sbatch \
        --account="${JUPITER_ACCOUNT:-scifi}" \
        --job-name="readiness-${tag}" \
        --output="$READINESS_RUN/logs/${tag}-%j.out" \
        --error="$READINESS_RUN/logs/${tag}-%j.err" \
        --export="ALL,READINESS_RUN=$READINESS_RUN,READINESS_JUDGE_TASKS=$READINESS_JUDGE_TASKS,READINESS_JUDGE_TASKS_SHA256=$READINESS_JUDGE_TASKS_SHA256,READINESS_EXPECTED_TASKS_PER_SLOT=$READINESS_EXPECTED_TASKS_PER_SLOT,GEODML_EXPECTED_COMMIT=$GEODML_EXPECTED_COMMIT,JUDGE_SLOT=$slot,JUDGE_TAG=$tag,JUDGE_MODEL_FAMILY=$family,JUDGE_MODEL=$model,JUDGE_MODEL_REVISION=$revision" \
        analysis/scripts/slurm/jupiter/run_semantic_readiness_judge.sbatch
}

submit_one primary-frontier primary-frontier "${families[0]}" "${models[0]}" "${revisions[0]}"
submit_one replicate-frontier-a replicate-frontier-a "${families[1]}" "${models[1]}" "${revisions[1]}"
submit_one replicate-frontier-b replicate-frontier-b "${families[2]}" "${models[2]}" "${revisions[2]}"

echo "Frozen panel: $panel_tsv"
echo "Monitor with: squeue -u $USER"
