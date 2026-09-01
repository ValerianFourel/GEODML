"""Static contracts for JUPITER Phase-2 readiness judge jobs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
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
READINESS_30K_PILOT = JUPITER_ROOT / "run_readiness_30k_four_gpu_pilot.sh"
READINESS_GENERATOR_INSTALLER = (
    JUPITER_ROOT / "install_readiness_generator_runtime.sh"
)
READINESS_30K_END_TO_END = JUPITER_ROOT / "run_readiness_30k_end_to_end.sh"
READINESS_JUPITER_RUNTIME = JUPITER_ROOT / "readiness_jupiter_runtime.sh"
READINESS_30K_PIPELINE_STAGE = (
    JUPITER_ROOT / "run_readiness_30k_pipeline_stage.sh"
)
READINESS_30K_SEARCH_TRIGGER_V2 = (
    JUPITER_ROOT / "run_readiness_30k_search_trigger_v2.sh"
)
READINESS_30K_SEARCH_TRIGGER_V2_HIGH_AXIS = (
    JUPITER_ROOT / "run_readiness_30k_search_trigger_v2_high_axis.sh"
)
READINESS_30K_AXIS1_STRICT_LOOP = (
    JUPITER_ROOT / "run_readiness_30k_axis1_strict_loop.sh"
)
READINESS_30K_AXIS1_8GPU_RESUME = (
    JUPITER_ROOT / "run_readiness_30k_axis1_8gpu_resume.sbatch"
)
READINESS_30K_AXIS1_PARTITION = (
    JUPITER_ROOT / "run_readiness_30k_axis1_partition.sbatch"
)
READINESS_30K_AXIS1_ONE_NODE_RECOVERY = (
    JUPITER_ROOT / "run_readiness_30k_axis1_one_node_recovery.sbatch"
)
READINESS_30K_AXIS1_ONE_NODE_GLOBAL = (
    JUPITER_ROOT / "run_readiness_30k_axis1_one_node_global.sbatch"
)
READINESS_30K_AXIS1_KEYWORD_SECTION = (
    JUPITER_ROOT / "run_readiness_30k_axis1_keyword_section.sbatch"
)
READINESS_30K_REPARTITION_KEYWORD_SECTIONS = (
    JUPITER_ROOT / "run_readiness_30k_repartition_keyword_sections.sbatch"
)
READINESS_30K_REPARTITION_SUBMITTER = (
    JUPITER_ROOT / "submit_readiness_30k_repartition_keyword_sections.sh"
)
READINESS_30K_PARTITION_FINALIZER = (
    JUPITER_ROOT / "finalize_readiness_30k_partitions.sh"
)
READINESS_AXIS1_CHECKPOINT_AUDIT = (
    JUPITER_ROOT / "run_readiness_axis1_checkpoint_audit.sh"
)


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
            READINESS_30K_PILOT,
            READINESS_GENERATOR_INSTALLER,
            READINESS_30K_END_TO_END,
            READINESS_JUPITER_RUNTIME,
            READINESS_30K_PIPELINE_STAGE,
            READINESS_30K_SEARCH_TRIGGER_V2,
            READINESS_30K_SEARCH_TRIGGER_V2_HIGH_AXIS,
            READINESS_30K_AXIS1_STRICT_LOOP,
            READINESS_30K_AXIS1_8GPU_RESUME,
            READINESS_30K_AXIS1_PARTITION,
            READINESS_30K_AXIS1_ONE_NODE_RECOVERY,
            READINESS_30K_AXIS1_ONE_NODE_GLOBAL,
            READINESS_30K_AXIS1_KEYWORD_SECTION,
            READINESS_30K_REPARTITION_KEYWORD_SECTIONS,
            READINESS_30K_REPARTITION_SUBMITTER,
            READINESS_30K_PARTITION_FINALIZER,
            READINESS_AXIS1_CHECKPOINT_AUDIT,
        ):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_30k_pilot_preflights_models_and_retains_exhausted_tasks(self) -> None:
        script = READINESS_30K_PILOT.read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertLess(
            script.index('cd "$GEODML_REPOSITORY"'),
            script.index("from analysis.interpretability.utils"),
        )
        self.assertIn('expected_transformers = "5.6.2"', script)
        self.assertIn("AutoConfig.from_pretrained", script)
        self.assertIn("AutoModelForMultimodalLM", script)
        self.assertIn("--allow-failed-tasks", script)
        self.assertIn('"failed_task_count": manifest["failed_task_count"]', script)
        self.assertIn("READINESS_APPROVED_WALLTIME", script)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", script)
        self.assertNotIn('printf \'%s\\n\' "01:00:00"', script)

        installer = READINESS_GENERATOR_INSTALLER.read_text(encoding="utf-8")
        self.assertNotIn("salloc", installer)
        self.assertNotIn("sbatch", installer)
        self.assertIn('"transformers==5.6.2"', installer)
        self.assertIn("--system-site-packages", installer)
        self.assertIn("PYTHONNOUSERSITE=1", installer)
        self.assertIn('"python-dotenv==1.1.1"', installer)
        self.assertIn('"huggingface-hub==1.16.1"', installer)

    def test_30k_end_to_end_runner_is_one_resumable_strict_pipeline(self) -> None:
        script = READINESS_30K_END_TO_END.read_text(encoding="utf-8")
        worker = READINESS_30K_PIPELINE_STAGE.read_text(encoding="utf-8")
        projection_verifier = (
            REPOSITORY_ROOT
            / "analysis/scripts/verify_readiness_projection_checkpoint.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertIn("srun --exact --exclusive", script)
        self.assertIn("READINESS_APPROVED_WALLTIME", script)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", script)
        self.assertIn("READINESS_SOURCE_PILOT_ROOT", script)
        self.assertIn("READINESS_MAX_REFINEMENT_ROUNDS", script)
        self.assertIn("READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND", script)
        self.assertIn(
            'READINESS_DISTANCE_TOLERANCE="${READINESS_DISTANCE_TOLERANCE:-0.017}"',
            script,
        )
        self.assertIn(
            'READINESS_TEXT_CONTRACT="${READINESS_TEXT_CONTRACT:-question-v1}"',
            script,
        )
        self.assertIn(
            'READINESS_ACCEPTANCE_CONTRACT="${READINESS_ACCEPTANCE_CONTRACT:-question-v1}"',
            script,
        )
        self.assertIn(
            'expected_tolerance = 0.017 if text_contract == "question-v1" else 0.035',
            script,
        )
        self.assertIn("readiness text and acceptance contracts must use one version", script)
        self.assertIn("strict axis-1 plan preflight: PASS", script)
        self.assertIn('manifest.get("target_design") != "axis-1-quantized-uniform"', script)
        self.assertIn('"keyword_count": 1011', script)
        self.assertIn('"target_count_per_keyword": 30', script)
        self.assertIn('"task_count": 30330', script)
        self.assertIn('math.isclose(increment, 0.001', script)
        self.assertNotIn('READINESS_DISTANCE_TOLERANCE:-0.22', script)
        self.assertIn("prepare_refinement_task_batch", script)
        self.assertIn("partition_readiness_refinement_tasks.py", script)
        self.assertIn("keyword_section_plan_sha256", script)
        partitioner = (
            REPOSITORY_ROOT
            / "analysis/scripts/partition_readiness_refinement_tasks.py"
        ).read_text(encoding="utf-8")
        self.assertIn("REFINEMENT BATCH", partitioner)
        self.assertIn("flock -n 9", script)
        self.assertIn("another controller already owns this pipeline root", script)
        self.assertIn(".strict-selection-attempt-", script)
        self.assertIn("the end-to-end loop requires at least four allocated GPUs", script)
        self.assertIn("the two-generator loop requires an even allocated GPU count", script)
        self.assertIn('generation_shards_per_generator="$((allocated_gpu_count / 2))"', script)
        self.assertNotIn("shard_count=2", script)
        self.assertIn("GENERATION CHECKPOINTED", script)
        self.assertIn("GENERATION CONTINUING", script)
        self.assertIn("allocation_seconds_left", script)
        self.assertIn("SLURM_ARRAY_JOB_ID", script)
        self.assertIn("SLURM_ARRAY_TASK_ID", script)
        self.assertIn("squeue -h -r", script)
        self.assertIn("-o '%A|%a|%L'", script)
        self.assertIn("expected exactly one Slurm time-left value", script)
        self.assertIn("READINESS_FINALIZATION_RESERVE_SECONDS", script)
        self.assertIn('READINESS_GENERATION_SECONDS="$generation_slice_seconds"', script)
        self.assertIn("validate", script)
        self.assertIn("project-qwen", script)
        self.assertIn("project-mistral", script)
        self.assertIn("compare-projections", script)
        self.assertIn("spatial-select", script)
        self.assertIn("--require-both-views-within-tolerance", script)
        self.assertIn("--require-delexicalized-template-uniqueness", script)
        self.assertIn("verified_population_passed", script)
        self.assertIn('pipeline_status="quality-gate-failed"', script)
        self.assertIn("not source_pilot_mode", script)
        self.assertIn('READINESS_VALIDATION_SHARD_COUNT:-4', script)
        self.assertIn("initial_validation_slots", script)
        self.assertIn("launch_validation_shard", script)
        self.assertIn("validation shards do not cover the exact candidate set", script)
        self.assertIn("SOURCE PILOT REFINEMENT", script)
        self.assertIn("interrupt_pipeline", script)
        self.assertIn("cached work is preserved", script)
        self.assertIn(".qwen-attempt-$projection_attempt", script)
        self.assertIn(".mistral-attempt-$projection_attempt", script)
        self.assertIn("READINESS_RECOVERY_PIPELINE_ROOT", script)
        self.assertIn("READINESS_INITIAL_CANDIDATE_ROOT", script)
        self.assertIn("READINESS_INITIAL_CANDIDATE_FILE_LIST", script)
        self.assertIn("READINESS_INITIAL_LOGICAL_ROUND_INDEX", script)
        self.assertIn('mapfile -t initial_candidates < "$initial_candidate_file_list"', script)
        self.assertIn(
            '[[ -n "$recovery_pipeline" && -z "$initial_candidate_file_list" ]]',
            script,
        )
        self.assertIn("READINESS_INITIAL_PROJECTION_ROOT", script)
        self.assertIn("READINESS_INITIAL_VALIDATION_OUTPUT", script)
        self.assertIn("READINESS_VALIDATION_CACHE_ROOT", script)
        self.assertIn("recover_projection_attempt", script)
        self.assertIn("recover_projection_source", script)
        self.assertIn("projection_artifact_matches", script)
        self.assertIn(
            'get("attention_implementation", "eager")', projection_verifier
        )
        self.assertIn("READINESS_BASE_PROJECTION_ROOT", script)
        self.assertIn("READINESS_COORDINATE_ONLY_PROJECTION_REUSE", script)
        self.assertIn("--base-coordinate-projections", worker)
        self.assertIn("READINESS_BASE_VALIDATION_OUTPUT", script)
        self.assertIn("recovered completed projection", script)
        self.assertIn("merge_readiness_validation_caches.py", script)
        self.assertIn("READINESS_VALIDATION_CACHE_SEARCH_ROOTS", script)
        self.assertIn("verify_readiness_projection_checkpoint.py", script)
        self.assertNotIn('manifest["candidate_files"] == identities', script)
        self.assertNotIn("stale current-job Qwen projection", script)
        self.assertIn("the independent validator must differ", script)

        self.assertIn("GEODML_GENERATOR_VENV", worker)
        self.assertIn("QWEN_LLM2VEC_VENV", worker)
        self.assertIn("MISTRAL_LLM2VEC_VENV", worker)
        self.assertIn("PYTHONNOUSERSITE=1", worker)
        self.assertIn("CUDA_VISIBLE_DEVICES is Slurm's per-step GPU isolation", worker)
        self.assertNotIn("VIRTUAL_ENV CUDA_VISIBLE_DEVICES", worker)
        self.assertIn('case "$stage"', worker)
        self.assertIn("validate-candidates", worker)
        self.assertIn("READINESS_VALIDATION_SHARD_COUNT", worker)
        self.assertIn("READINESS_VALIDATION_SHARD_INDEX", worker)
        self.assertIn("READINESS_VALIDATION_SHARD_SALT", worker)
        self.assertIn("READINESS_VALIDATION_SHARD_SALT", script)
        self.assertIn("project-candidates", worker)
        self.assertIn("--base-projections", worker)
        self.assertIn("--attention-implementation", worker)
        self.assertIn("--base-validation", worker)
        self.assertIn("READINESS_VALIDATION_BATCH_SIZE", worker)
        self.assertIn("--text-contract", worker)
        self.assertIn("--acceptance-contract", worker)

        strict_loop = READINESS_30K_AXIS1_STRICT_LOOP.read_text(encoding="utf-8")
        self.assertNotIn("salloc", strict_loop)
        self.assertNotIn("sbatch", strict_loop)
        self.assertIn("READINESS_APPROVED_WALLTIME", strict_loop)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", strict_loop)
        self.assertIn(
            'READINESS_DISTANCE_TOLERANCE="${READINESS_DISTANCE_TOLERANCE:-0.017}"',
            strict_loop,
        )
        self.assertIn("READINESS_TEXT_CONTRACT", strict_loop)
        self.assertIn("READINESS_ACCEPTANCE_CONTRACT", strict_loop)
        self.assertIn(
            'READINESS_VALIDATION_SHARD_COUNT:-$READINESS_ALLOCATED_GPU_COUNT',
            strict_loop,
        )
        self.assertIn('READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024', strict_loop)
        self.assertIn("READINESS_VALIDATION_CACHE_SEARCH_ROOTS", strict_loop)
        self.assertIn("run_readiness_30k_end_to_end.sh", strict_loop)

    def test_search_trigger_v2_runner_is_additive_and_never_allocates(self) -> None:
        script = READINESS_30K_SEARCH_TRIGGER_V2.read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertNotIn("#SBATCH", script)
        self.assertIn("SLURM_JOB_ID", script)
        self.assertIn("READINESS_APPROVED_WALLTIME_SECONDS", script)
        self.assertIn('READINESS_ALLOCATED_GPU_COUNT:-4', script)
        self.assertIn('"maximum_gpu_hours"', script)
        self.assertIn('"scheduler_allocated_cpus"', script)
        self.assertIn('READINESS_TEXT_CONTRACT="search-trigger-v2"', script)
        self.assertIn('READINESS_ACCEPTANCE_CONTRACT="search-trigger-v2"', script)
        self.assertIn('READINESS_DISTANCE_TOLERANCE="0.035"', script)
        self.assertIn('READINESS_COORDINATE_ONLY_PROJECTION_REUSE="1"', script)
        self.assertIn("READINESS_INITIAL_CANDIDATE_FILE_LIST", script)
        self.assertIn("READINESS_INITIAL_PROJECTION_ROOT", script)
        self.assertIn("READINESS_INITIAL_VALIDATION_OUTPUT", script)
        self.assertIn("maximum_candidate_round_index", script)
        self.assertIn("READINESS_FINALIZATION_RESERVE_SECONDS", script)
        self.assertIn('SLURM_MPI_TYPE="none"', script)
        self.assertIn("readiness_jupiter_runtime.sh", script)
        self.assertLess(
            script.index("readiness_bootstrap_jupiter_control_runtime"),
            script.index("approved_walltime_seconds="),
        )
        self.assertIn("SEARCH_TRIGGER_V2_CONTROL_RUNTIME=PASS", script)
        self.assertIn("Prompt embeddings describe generated text", script)

    def test_high_axis_v2_harness_targets_action_end_without_allocating(self) -> None:
        script = READINESS_30K_SEARCH_TRIGGER_V2_HIGH_AXIS.read_text(
            encoding="utf-8"
        )
        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertNotIn("#SBATCH", script)
        self.assertIn("SLURM_JOB_ID", script)
        self.assertIn("readiness_jupiter_runtime.sh", script)
        self.assertIn("readiness_bootstrap_jupiter_control_runtime", script)
        self.assertLess(
            script.index("readiness_bootstrap_jupiter_control_runtime"),
            script.index("counterfactual_root="),
        )
        self.assertIn("HIGH_AXIS_CONTROL_RUNTIME=PASS", script)
        self.assertIn('READINESS_GENERATION_PROFILE="high-axis-action-v1"', script)
        self.assertIn("READINESS_REFINEMENT_MIN_TARGET_AXIS_1", script)
        self.assertIn('READINESS_REFINEMENT_TASK_PRIORITY="descending-axis-1"', script)
        self.assertIn("READINESS_HIGH_AXIS_BASELINE_SELECTED", script)
        self.assertIn("search_trigger_v2_relaxed_tolerance", script)
        self.assertIn("run_readiness_30k_search_trigger_v2.sh", script)
        self.assertIn('exec "$driver"', script)
        self.assertIn("Prompt embeddings diagnose generated text", script)

    def test_shared_jupiter_runtime_clears_inherited_venv_and_loads_stack(self) -> None:
        runtime = READINESS_JUPITER_RUNTIME.read_text(encoding="utf-8")

        self.assertIn("readiness_clear_inherited_python_runtime", runtime)
        self.assertIn("readiness_load_jupiter_stack", runtime)
        self.assertIn("readiness_bootstrap_jupiter_control_runtime", runtime)
        self.assertIn("unset PYTHONHOME PYTHONPATH VIRTUAL_ENV", runtime)
        self.assertIn("module load PyTorch/2.9.1", runtime)
        self.assertIn('jutil env activate -p', runtime)

        end_to_end = READINESS_30K_END_TO_END.read_text(encoding="utf-8")
        self.assertIn("readiness_jupiter_runtime.sh", end_to_end)
        self.assertNotIn("clear_runtime()", end_to_end)
        self.assertNotIn("load_stack()", end_to_end)

    def test_shared_jupiter_runtime_executes_with_clean_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            inherited_bin = temporary / "inherited-venv/bin"
            inherited_bin.mkdir(parents=True)
            inherited_python = inherited_bin / "python3"
            inherited_python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
            inherited_python.chmod(0o755)
            runtime_log = temporary / "runtime.log"

            command = textwrap.dedent(
                """
                set -euo pipefail
                module() {
                    printf 'module %s\n' "$*" >> "$RUNTIME_LOG"
                }
                jutil() {
                    printf 'jutil %s\n' "$*" >> "$RUNTIME_LOG"
                }
                source "$RUNTIME_HELPER"
                readiness_bootstrap_jupiter_control_runtime \
                    "MOCK_JUPITER_RUNTIME=PASS"
                printf 'VIRTUAL_ENV=%s\n' "${VIRTUAL_ENV-unset}"
                printf 'PYTHONHOME=%s\n' "${PYTHONHOME-unset}"
                printf 'PYTHONPATH=%s\n' "${PYTHONPATH-unset}"
                command -v python3
                """
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{inherited_bin}:{environment['PATH']}",
                    "PYTHONHOME": "/invalid/python-home",
                    "PYTHONPATH": "/invalid/python-path",
                    "RUNTIME_HELPER": str(READINESS_JUPITER_RUNTIME),
                    "RUNTIME_LOG": str(runtime_log),
                    "VIRTUAL_ENV": str(inherited_bin.parent),
                }
            )
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-c", command],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MOCK_JUPITER_RUNTIME=PASS", result.stdout)
            self.assertIn("VIRTUAL_ENV=unset", result.stdout)
            self.assertIn("PYTHONHOME=unset", result.stdout)
            self.assertIn("PYTHONPATH=unset", result.stdout)
            self.assertNotIn(str(inherited_python), result.stdout)
            self.assertEqual(
                runtime_log.read_text(encoding="utf-8").splitlines(),
                [
                    "module --force purge",
                    "module load Stages/2026",
                    "module load GCCcore/14.3.0",
                    "module load SciPy-Stack/2025b",
                    "module load git",
                    "module load PyTorch/2.9.1",
                    "jutil env activate -p scifi",
                ],
            )

    def test_axis1_eight_gpu_resume_records_approved_budget_and_reuses_work(self) -> None:
        script = READINESS_30K_AXIS1_8GPU_RESUME.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks=8", script)
        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn('READINESS_APPROVED_WALLTIME="12:00:00"', script)
        self.assertIn('READINESS_ALLOCATED_GPU_COUNT="8"', script)
        self.assertIn('READINESS_VALIDATION_SHARD_COUNT="8"', script)
        self.assertIn('READINESS_VALIDATION_BATCH_SIZE="16"', script)
        self.assertIn('READINESS_EMBEDDING_BATCH_SIZE="32"', script)
        self.assertIn('READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION="eager"', script)
        self.assertIn('READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="1024"', script)
        self.assertIn("maximum 96 GPU-hours", script)
        self.assertIn("READINESS_RECOVERY_PIPELINE_ROOT", script)
        self.assertIn("READINESS_INITIAL_PROJECTION_ROOT", script)
        self.assertIn("READINESS_INITIAL_VALIDATION_OUTPUT", script)
        self.assertIn("READINESS_INITIAL_CANDIDATE_FILE_LIST", script)
        self.assertIn("READINESS_INITIAL_LOGICAL_ROUND_INDEX", script)
        self.assertIn("latest_verified_summary", script)

    def test_axis1_partition_jobs_are_disjoint_and_finalize_the_exact_union(self) -> None:
        script = READINESS_30K_AXIS1_PARTITION.read_text(encoding="utf-8")
        finalizer = READINESS_30K_PARTITION_FINALIZER.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=4", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=12:00:00", script)
        self.assertIn('READINESS_WORK_PARTITION_COUNT="2"', script)
        self.assertIn("SLURM_ARRAY_TASK_ID", script)
        self.assertIn("axis1-30330-two-way-v1", script)
        self.assertIn("partition-$READINESS_WORK_PARTITION_INDEX-latest.txt", script)
        self.assertIn("READINESS_CONTINUATION_SOURCE_PREFIX", script)
        self.assertIn(
            'checkpoint_search_root="${continuation_source_root:-$READINESS_RECOVERY_PIPELINE_ROOT}"',
            script,
        )
        self.assertIn(
            'READINESS_APPROVED_WALLTIME="${READINESS_APPROVED_WALLTIME:-12:00:00}"',
            script,
        )
        self.assertIn(
            'READINESS_FINALIZATION_RESERVE_SECONDS="${READINESS_FINALIZATION_RESERVE_SECONDS:-5400}"',
            script,
        )
        self.assertIn("continuation source and destination partition roots must differ", script)
        self.assertIn(
            'READINESS_INITIAL_CANDIDATE_FILE_LIST="${READINESS_INITIAL_CANDIDATE_FILE_LIST:-$latest_verified_round/candidate-files.txt}"',
            script,
        )
        self.assertIn('READINESS_INITIAL_LOGICAL_ROUND_INDEX="$((10#$latest_verified_round_number))"', script)
        self.assertIn('f"producer-{index}.ready.json"', script)
        self.assertIn("finalize_readiness_30k_partitions.sh", script)

        self.assertIn("merge_readiness_partition_checkpoints.py", finalizer)
        self.assertIn("compare-projections", finalizer)
        self.assertIn("spatial-select", finalizer)
        self.assertIn("--require-both-views-within-tolerance", finalizer)
        self.assertIn("--require-delexicalized-template-uniqueness", finalizer)
        self.assertIn("final-latest.txt", finalizer)
        self.assertIn("PARTITION FINALIZER COMPLETE", finalizer)
        self.assertIn("verified_round_summary.json", script)
        self.assertIn("SLURM_JOB_NUM_NODES", script)
        self.assertIn("run_readiness_30k_axis1_strict_loop.sh", script)

    def test_axis1_one_node_recovery_is_bounded_and_traceable(self) -> None:
        script = READINESS_30K_AXIS1_ONE_NODE_RECOVERY.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=4", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --time=01:00:00", script)
        self.assertIn('READINESS_APPROVED_WALLTIME="01:00:00"', script)
        self.assertIn("maximum four GPU-hours", script)
        self.assertIn("module load GCCcore/14.3.0", script)
        self.assertIn("batch job 1484067 failed during repository preflight", script)
        self.assertIn("10-20 minutes", script)
        self.assertIn("40-50 minutes of safety margin", script)
        self.assertIn("approximately 0.7-1.3 GPU-hours", script)
        self.assertIn("GEODML_RECOVERY_ORCHESTRATOR_REPOSITORY", script)

        self.assertIn(
            'orchestrator_repository="$(realpath "$GEODML_RECOVERY_ORCHESTRATOR_REPOSITORY")"',
            script,
        )
        self.assertNotIn('dirname "${BASH_SOURCE[0]}"', script)
        self.assertIn("quarantine_corrupt_readiness_validation.py", script)
        self.assertIn("validation-shard-2.jsonl", script)
        self.assertIn("corrupt-source-job-1481430", script)
        self.assertIn('science_commit="f77b16f453a9421218d44a4d2e896cb7eb5fb589"', script)
        self.assertIn("GEODML_RECOVERY_ORCHESTRATOR_COMMIT", script)
        self.assertIn('READINESS_FINALIZATION_RESERVE_SECONDS="900"', script)
        self.assertIn('READINESS_MAX_REFINEMENT_ROUNDS="1000"', script)
        self.assertIn('READINESS_STOP_AFTER_PHYSICAL_ROUND="19"', script)
        self.assertIn("READINESS_END_TO_END_RUNNER", script)
        self.assertIn(
            '"$orchestrator_repository/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"',
            script,
        )
        self.assertIn("recover_partition 0", script)
        self.assertIn("recover_partition 1", script)
        self.assertIn("round-19", script)
        self.assertIn("finalize_readiness_30k_partitions.sh", script)
        self.assertIn("recovery-job-$SLURM_JOB_ID.json", script)

        strict_loop = READINESS_30K_AXIS1_STRICT_LOOP.read_text(encoding="utf-8")
        end_to_end = READINESS_30K_END_TO_END.read_text(encoding="utf-8")
        self.assertIn("READINESS_END_TO_END_RUNNER", strict_loop)
        self.assertIn("READINESS_STOP_AFTER_PHYSICAL_ROUND", end_to_end)
        self.assertIn("OPERATIONAL CHECKPOINT", end_to_end)

    def test_axis1_one_node_global_continues_the_fused_checkpoint(self) -> None:
        script = READINESS_30K_AXIS1_ONE_NODE_GLOBAL.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertNotIn("#SBATCH --time=", script)
        self.assertIn("Wall time is intentionally omitted", script)
        self.assertIn("READINESS_APPROVED_WALLTIME", script)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", script)
        self.assertIn("READINESS_GLOBAL_CHECKPOINT_ROOT", script)
        self.assertIn("merged/candidates.jsonl", script)
        self.assertIn("merged/projections", script)
        self.assertIn("merged/validation.jsonl", script)
        self.assertIn('READINESS_WORK_PARTITION_COUNT="1"', script)
        self.assertIn("empty-validation-cache-source", script)
        self.assertIn("geodml-readiness-global-latest.txt", script)
        self.assertIn("run_readiness_30k_axis1_strict_loop.sh", script)

    def test_axis1_keyword_section_is_independent_and_uses_frozen_plan(self) -> None:
        script = READINESS_30K_AXIS1_KEYWORD_SECTION.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=1", script)
        self.assertIn("#SBATCH --ntasks=4", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertNotIn("#SBATCH --time=", script)
        self.assertIn("Wall time is supplied by sbatch", script)
        self.assertIn("READINESS_APPROVED_WALLTIME", script)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", script)
        self.assertIn("READINESS_KEYWORD_SECTION_PLAN", script)
        self.assertIn("READINESS_TEN_SECTION_RUN_ROOT", script)
        self.assertIn("--verify-plan", script)
        self.assertIn("READINESS_WORK_PARTITION_COUNT", script)
        self.assertIn("READINESS_WORK_PARTITION_INDEX", script)
        self.assertIn("READINESS_WORK_PARTITION_SALT", script)
        self.assertIn('section_name="section-$READINESS_WORK_PARTITION_INDEX-of-$READINESS_WORK_PARTITION_COUNT"', script)
        self.assertIn("empty-validation-cache-source", script)
        self.assertIn("READINESS_INITIAL_CANDIDATE_FILE_LIST", script)
        self.assertIn("READINESS_INITIAL_LOGICAL_ROUND_INDEX", script)
        self.assertIn("readiness-keyword-section-checkpoint-v1", script)
        self.assertNotIn("finalize_readiness_30k_partitions.sh", script)
        self.assertIn("run_readiness_30k_axis1_strict_loop.sh", script)

    def test_keyword_sections_are_globally_repartitioned_before_resubmission(self) -> None:
        merge_job = READINESS_30K_REPARTITION_KEYWORD_SECTIONS.read_text(
            encoding="utf-8"
        )
        submitter = READINESS_30K_REPARTITION_SUBMITTER.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --nodes=1", merge_job)
        self.assertIn("#SBATCH --ntasks=1", merge_job)
        self.assertIn("#SBATCH --cpus-per-task=32", merge_job)
        self.assertIn("#SBATCH --mem=128G", merge_job)
        self.assertIn("#SBATCH --gres=none", merge_job)
        self.assertNotIn("#SBATCH --time=", merge_job)
        self.assertIn("READINESS_APPROVED_WALLTIME", merge_job)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", merge_job)
        self.assertIn("expected exactly ten source section roots", merge_job)
        self.assertIn("merge_readiness_partition_checkpoints.py", merge_job)
        self.assertIn("compare-projections", merge_job)
        self.assertIn("spatial-select", merge_job)
        self.assertIn("audit_fully_compliant_readiness_prompts.py", merge_job)
        self.assertIn("prepare_readiness_keyword_sections.py", merge_job)
        self.assertIn("--section-count 10", merge_job)
        self.assertIn("readiness-30k-ten-section-repartition-run-v1", merge_job)

        self.assertIn("READINESS_MERGE_APPROVED_WALLTIME", submitter)
        self.assertIn("READINESS_SECTION_APPROVED_WALLTIME", submitter)
        self.assertIn("--dependency=\"afterok:$merge_job_id\"", submitter)
        self.assertIn("for index in {0..9}", submitter)
        self.assertIn("maximum_gpu_hours\": 320", submitter)
        self.assertIn("run_readiness_30k_repartition_keyword_sections.sbatch", submitter)
        self.assertIn("run_readiness_30k_axis1_keyword_section.sbatch", submitter)
        self.assertIn("submission_manifest.json", submitter)

    def test_axis1_checkpoint_audit_uses_all_four_gpus_and_preserves_semantics(self) -> None:
        script = READINESS_AXIS1_CHECKPOINT_AUDIT.read_text(encoding="utf-8")

        self.assertNotIn("salloc", script)
        self.assertNotIn("sbatch", script)
        self.assertIn("requires exactly four allocated GPUs", script)
        self.assertEqual(script.count("--gres=gpu:1"), 4)
        self.assertIn("READINESS_APPROVED_WALLTIME", script)
        self.assertIn("READINESS_ALLOCATION_ESTIMATE", script)
        self.assertIn("project-qwen", script)
        self.assertIn("project-mistral", script)
        self.assertIn("READINESS_VALIDATION_SHARD_INDEX=0", script)
        self.assertIn("READINESS_VALIDATION_SHARD_INDEX=1", script)
        self.assertIn("compare-projections", script)
        self.assertIn("audit_readiness_axis1_continuity.py", script)
        self.assertIn("--primary-tolerance-steps 0.5", script)
        self.assertIn("validator cache continues", script)

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
