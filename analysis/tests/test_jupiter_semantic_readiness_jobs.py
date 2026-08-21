"""Static contracts for JUPITER Phase-2 readiness judge jobs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JUPITER_ROOT = REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter"
SBATCH = JUPITER_ROOT / "run_semantic_readiness_judge.sbatch"
SUBMIT = JUPITER_ROOT / "submit_semantic_readiness_panel.sh"
BEHAVIORAL_DEBUG_QUEUE = JUPITER_ROOT / "run_readiness_behavioral_debug_queue.sh"
EXPANDED_LLAMA_QUEUE = JUPITER_ROOT / "run_readiness_expanded_llama_queue.sbatch"
PROJECT_SETUP = JUPITER_ROOT / "geodml_project_setup.sh"
INCREMENTAL_FOUR_JUDGE_QUEUE = (
    JUPITER_ROOT / "run_readiness_incremental_four_judge_queue.sbatch"
)
INCREMENTAL_FOUR_JUDGE_SUBMITTER = (
    JUPITER_ROOT / "submit_readiness_incremental_four_judge.sh"
)
ABSTENTION_20K_PREPARER = (
    JUPITER_ROOT / "prepare_readiness_20k_abstention_four_judge.sh"
)
ABSTENTION_20K_SLICE = JUPITER_ROOT / "run_readiness_20k_abstention_slice.sbatch"
INCREMENTAL_AUDIT = JUPITER_ROOT / "audit_readiness_incremental_four_judge.sh"
ABSTENTION_20K_AUDIT = JUPITER_ROOT / "audit_readiness_20k_abstention.sh"
READINESS_HF_EMBEDDINGS = JUPITER_ROOT / "run_readiness_hf_embedding_views.sh"
READINESS_PROMPT_ROUND1 = JUPITER_ROOT / "run_readiness_prompt_round1.sh"


class JupiterSemanticReadinessJobTests(unittest.TestCase):
    def test_shell_scripts_are_valid_bash(self) -> None:
        for script in (
            SBATCH,
            SUBMIT,
            BEHAVIORAL_DEBUG_QUEUE,
            EXPANDED_LLAMA_QUEUE,
            PROJECT_SETUP,
            INCREMENTAL_FOUR_JUDGE_QUEUE,
            INCREMENTAL_FOUR_JUDGE_SUBMITTER,
            ABSTENTION_20K_PREPARER,
            ABSTENTION_20K_SLICE,
            INCREMENTAL_AUDIT,
            ABSTENTION_20K_AUDIT,
            READINESS_HF_EMBEDDINGS,
            READINESS_PROMPT_ROUND1,
        ):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_prompt_round1_runner_is_resumable_and_fail_closed(self) -> None:
        script = READINESS_PROMPT_ROUND1.read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertNotIn("srun", script)
        self.assertIn("GEODML_EXPECTED_COMMIT", script)
        self.assertIn("clean exact-commit worktree", script)
        self.assertIn("clear_inherited_python_runtime", script)
        self.assertIn('[[ "$path_entry" == "$inherited_venv_bin" ]]', script)
        self.assertIn("partial or conflicting immutable projection directory", script)
        self.assertIn("QWEN PROJECTION: COMPLETE; SKIPPING", script)
        self.assertIn("MISTRAL PROJECTION: COMPLETE; SKIPPING", script)
        self.assertIn("validator does not cover the exact candidate set", script)
        self.assertIn("projection candidate set differs", script)
        self.assertIn("compare-projections", script)
        self.assertIn("spatial-select", script)
        self.assertIn("generator_target_continuity_qwen_view", script)
        self.assertIn("scale_to_30000_gate_passed", script)
        self.assertIn("do not scale to 30,000 yet", script)
        self.assertIn("do not define", script)
        self.assertIn("randomized policy variable B", script)

    def test_hf_embedding_runner_never_allocates_and_is_resumable(self) -> None:
        script = READINESS_HF_EMBEDDINGS.read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertNotIn("srun", script)
        self.assertIn("exactly one visible GPU", script)
        self.assertIn("restricted-local/prompts.jsonl", script)
        self.assertIn("qwen3-8b-mntp-unsup-simcse", script)
        self.assertIn("qwen3-8b-mntp-supervised", script)
        self.assertIn("qwen3-8b-llm2vec-gen", script)
        self.assertIn("--shard-size", script)
        self.assertIn("--git-commit-sha", script)

    def test_project_setup_is_source_only_and_exports_portable_roots(self) -> None:
        direct = subprocess.run(
            ["bash", str(PROJECT_SETUP)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(direct.returncode, 2)
        self.assertIn("must be sourced", direct.stderr)

        setup = PROJECT_SETUP.read_text(encoding="utf-8")
        self.assertIn('jutil env activate -p "$project_name"', setup)
        self.assertIn('$PROJECT/$USER/geodml', setup)
        self.assertIn('$FSCRATCH/$USER/geodml', setup)
        self.assertIn('git -C "$GEODML_REPOSITORY"', setup)
        self.assertNotIn("git fetch", setup)
        self.assertNotIn("module load", setup)
        self.assertNotIn("sbatch", setup)

        sourced = subprocess.run(
            [
                "bash",
                "-c",
                """
jutil() {
    export PROJECT=/project
    export FSCRATCH=/scratch
}
export USER=tester
export GEODML_REPOSITORY="$2"
source "$1"
printf '%s\n' \
    "$JUPITER_PROJECT" \
    "$GEODML_PROJECT_ROOT" \
    "$GEODML_CACHE_ROOT" \
    "$PWD"
""",
                "bash",
                str(PROJECT_SETUP),
                str(REPOSITORY_ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(sourced.returncode, 0, sourced.stderr)
        self.assertIn("geodml environment ready", sourced.stdout)
        self.assertIn("scifi\n/project/tester/geodml", sourced.stdout)
        self.assertIn("/scratch/tester/geodml", sourced.stdout)
        self.assertTrue(sourced.stdout.rstrip().endswith(str(REPOSITORY_ROOT)))

    def test_incremental_queue_labels_one_new_bank_with_four_canonical_judges(self) -> None:
        queue = INCREMENTAL_FOUR_JUDGE_QUEUE.read_text(encoding="utf-8")

        self.assertIn("READINESS_INCREMENTAL_TASKS_SHA256", queue)
        self.assertIn("judge slots do not cover identical prompt items", queue)
        self.assertIn("run_data_parallel full gemma4-31b-primary", queue)
        self.assertIn("run_data_parallel full qwen3-32b-replicate-a", queue)
        self.assertIn("run_data_parallel full ministral3-8b-replicate-b", queue)
        self.assertIn('run_llama full ""', queue)
        self.assertIn("replicate-frontier-c", queue)
        self.assertNotIn("Qwen3-8B", queue)
        self.assertNotIn("primary-sensitivity", queue)
        self.assertIn("--run-purpose \"$purpose\"", queue)
        self.assertIn("--resume", queue)
        self.assertLess(
            queue.index('echo "queue complete:'),
            queue.index('find "$queue_root" -type f'),
        )

        submitter = INCREMENTAL_FOUR_JUDGE_SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("READINESS_CANDIDATE_MAXIMUM_PER_SOURCE:-3200", submitter)
        self.assertIn("READINESS_MINIMUM_NEW_PROMPTS:-10000", submitter)
        self.assertIn("nested sample lost", submitter)
        self.assertIn("incremental corpus overlaps an already annotated text", submitter)
        self.assertIn("insufficient development coverage", submitter)
        self.assertIn("insufficient confirmation coverage", submitter)
        self.assertIn("FOUR-SLOT COVERAGE: PASS", submitter)
        self.assertIn("run_readiness_incremental_four_judge_queue.sbatch", submitter)
        self.assertIn("ALL INPUTS VALIDATED AND INCREMENTAL JOB SUBMITTED: PASS", submitter)

    def test_20k_abstention_preparation_never_allocates_and_freezes_v2(self) -> None:
        preparer = ABSTENTION_20K_PREPARER.read_text(encoding="utf-8")

        self.assertIn("semantic_readiness_expanded_corpus.jsonl", preparer)
        self.assertIn("READINESS_MINIMUM_COMBINED_PROMPTS:-20000", preparer)
        self.assertIn("decision-readiness-ordinal-abstention-v2", preparer)
        self.assertIn("judge slots do not cover identical prompt items", preparer)
        self.assertIn("DONT-KNOW CONTRACT: PASS", preparer)
        self.assertIn("No Slurm allocation was submitted", preparer)
        self.assertNotIn("sbatch", preparer)

    def test_20k_abstention_slice_is_short_resumable_and_has_no_default_time(self) -> None:
        queue = ABSTENTION_20K_SLICE.read_text(encoding="utf-8")

        self.assertNotIn("#SBATCH --time", queue)
        self.assertIn("READINESS_APPROVED_WALLTIME", queue)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", queue)
        self.assertIn("READINESS_SLICE_BUDGET_SECONDS", queue)
        self.assertIn("timeout --signal=TERM", queue)
        self.assertIn("wait_for_idle_gpus", queue)
        self.assertIn("GPU processes did not exit", queue)
        self.assertIn("task_cache", queue)
        self.assertIn("slice checkpointed", queue)
        self.assertIn("--resume", queue)
        self.assertIn("--resume-extra-attempts", queue)
        self.assertIn("resume-extra-attempts.txt", queue)
        self.assertIn("READINESS_ALLOW_FAILED_TASKS", queue)
        self.assertIn("allow-failed-tasks.txt", queue)
        self.assertIn("skipping terminal", queue)
        self.assertIn("skipping complete", queue)
        self.assertIn("run_data_parallel full qwen3-32b-replicate-a", queue)
        self.assertIn("run_data_parallel full ministral3-8b-replicate-b", queue)
        self.assertIn("run_data_parallel full gemma4-31b-primary", queue)
        self.assertIn('run_llama full ""', queue)
        self.assertNotIn("Qwen3-8B", queue)
        self.assertNotIn("primary-sensitivity", queue)
        self.assertLess(
            queue.index("run_data_parallel smoke qwen3-32b-replicate-a"),
            queue.index("run_data_parallel full qwen3-32b-replicate-a"),
        )
        self.assertLess(
            queue.index("run_data_parallel full qwen3-32b-replicate-a"),
            queue.index("run_data_parallel smoke ministral3-8b-replicate-b"),
        )

        audit = INCREMENTAL_AUDIT.read_text(encoding="utf-8")
        self.assertNotIn("sbatch", audit)
        self.assertIn("OBSERVED STAGE DURATIONS", audit)
        self.assertIn("TASK AND CACHE COUNTS", audit)
        self.assertIn("sacct", audit)

        v2_audit = ABSTENTION_20K_AUDIT.read_text(encoding="utf-8")
        self.assertNotIn("sbatch", v2_audit)
        self.assertNotIn("semantic_readiness_dataset", v2_audit)
        self.assertIn("json.JSONDecoder", v2_audit)
        self.assertIn("V2 ANSWER TYPES", v2_audit)

    def test_behavioral_debug_queue_smokes_before_full_runs(self) -> None:
        queue = BEHAVIORAL_DEBUG_QUEUE.read_text(encoding="utf-8")

        self.assertIn('actual_commit="$(git rev-parse HEAD)"', queue)
        self.assertIn("task-bank hash mismatch", queue)
        self.assertIn('visible_gpus" != "4"', queue)
        self.assertIn("run_stage smoke", queue)
        self.assertIn("run_stage full", queue)
        self.assertIn("Qwen3-32B", queue)
        self.assertIn("--disable-thinking", queue)
        self.assertIn("Ministral-3-8B-Instruct-2512-BF16", queue)
        self.assertIn("gemma-4-31B-it", queue)
        self.assertIn("--run-purpose debug", queue)
        self.assertIn("--resume", queue)
        self.assertIn("READINESS_DEBUG_QUEUE_ROOT", queue)
        self.assertIn("slurm-job-$SLURM_JOB_ID.txt", queue)
        self.assertIn("artifact-sha256.txt", queue)

    def test_job_requires_one_complete_gh200_node(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=booster", job)
        self.assertIn("#SBATCH --nodes=1", job)
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn("module load Stages/2026", job)
        self.assertIn("module load PyTorch/2.9.1", job)
        self.assertIn("GEODML_REQUIRED_GPU_COUNT=4", job)
        self.assertIn('SLURM_JOB_PARTITION:-}\" != \"booster', job)
        self.assertIn('gpu_name\" != *GH200*', job)

    def test_job_pins_provenance_and_runs_offline(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")

        self.assertIn("GEODML_EXPECTED_COMMIT", job)
        self.assertIn("READINESS_JUDGE_TASKS_SHA256", job)
        self.assertIn("READINESS_EXPECTED_TASKS_PER_SLOT", job)
        self.assertIn("JUDGE_MODEL_REVISION", job)
        self.assertIn("JUDGE_MODEL_FAMILY", job)
        self.assertIn("HF_HUB_OFFLINE=1", job)
        self.assertIn("TRANSFORMERS_OFFLINE=1", job)
        self.assertIn("--model-revision \"$JUDGE_MODEL_REVISION\"", job)
        self.assertIn("--model-family \"$JUDGE_MODEL_FAMILY\"", job)
        self.assertIn("--run-purpose production", job)
        self.assertIn("--resume", job)
        self.assertIn("READINESS_JUDGE_BATCH_SIZE", job)
        self.assertIn("artifact-sha256.txt", job)
        self.assertIn('manifest["task_count_for_slot"]', job)

    def test_submitter_requires_three_distinct_slots_and_snapshots(self) -> None:
        submitter = SUBMIT.read_text(encoding="utf-8")

        self.assertIn("PRIMARY_JUDGE_MODEL", submitter)
        self.assertIn("PRIMARY_JUDGE_FAMILY", submitter)
        self.assertIn("REPLICATE_A_JUDGE_MODEL", submitter)
        self.assertIn("REPLICATE_B_JUDGE_MODEL", submitter)
        self.assertIn("must use distinct model snapshots", submitter)
        self.assertIn("must use distinct model families", submitter)
        self.assertIn("Task-bank hash mismatch", submitter)
        self.assertIn("Refusing to change the frozen judge panel", submitter)
        self.assertIn("primary-frontier", submitter)
        self.assertIn("replicate-frontier-a", submitter)
        self.assertIn("replicate-frontier-b", submitter)
        self.assertEqual(submitter.count("\nsubmit_one "), 3)

    def test_expanded_queue_uses_four_distinct_families_and_two_task_banks(self) -> None:
        queue = EXPANDED_LLAMA_QUEUE.read_text(encoding="utf-8")

        self.assertIn("READINESS_TRANSFER_TASKS_SHA256", queue)
        self.assertIn("READINESS_EXPANDED_TASKS_SHA256", queue)
        self.assertIn("replicate-frontier-c", queue)
        self.assertIn("llama3.3-70b-replicate-c", queue)
        self.assertIn("--model-family llama", queue)
        self.assertIn("--batch-size \"${LLAMA_BATCH_SIZE:-16}\"", queue)
        self.assertIn("GEODML_DEVICE_MAP=balanced", queue)
        self.assertIn("run_data_parallel full qwen3-8b-primary-sensitivity", queue)
        self.assertIn("run_data_parallel full qwen3-32b", queue)
        self.assertIn("run_data_parallel full ministral3-8b", queue)
        self.assertIn("run_data_parallel full gemma4-31b", queue)
        self.assertIn("--run-purpose \"$purpose\"", queue)
        self.assertIn("--resume", queue)
        self.assertIn("artifact-sha256.txt", queue)


if __name__ == "__main__":
    unittest.main()
