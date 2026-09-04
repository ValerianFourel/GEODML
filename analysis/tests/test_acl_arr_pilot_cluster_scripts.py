from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JUPITER_ROOT = REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter"
DOWNLOADER = REPOSITORY_ROOT / "analysis/scripts/download_acl_arr_pilot_models.py"
SHELL_SCRIPTS = (
    JUPITER_ROOT / "prepare_acl_arr_pilot_environment.sh",
    JUPITER_ROOT / "run_acl_arr_pilot_model_setup.sh",
    JUPITER_ROOT / "launch_acl_arr_pilot_model_setup_tmux.sh",
    JUPITER_ROOT / "check_acl_arr_pilot_model_setup.sh",
    JUPITER_ROOT / "prepare_acl_arr_pilot_plan.sh",
    JUPITER_ROOT / "launch_acl_arr_document_pilot_tmux.sh",
    JUPITER_ROOT / "check_acl_arr_document_pilot.sh",
)
MODEL_IDS = (
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-Small-4-119B-2603",
    "Qwen/Qwen3.8-27B",
)


class AclArrPilotClusterScriptTests(unittest.TestCase):
    def test_all_cluster_scripts_have_valid_bash_syntax(self) -> None:
        for script in SHELL_SCRIPTS:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_setup_runs_in_tmux_without_a_slurm_allocation(self) -> None:
        source = (JUPITER_ROOT / "launch_acl_arr_pilot_model_setup_tmux.sh").read_text()
        self.assertIn("tmux new-session -d", source)
        self.assertNotIn("salloc", source)
        self.assertNotIn("sbatch", source)

    def test_environment_stage_writes_a_sourceable_commit_scoped_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            (repository / "tracked.txt").write_text("test\n")
            subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-qm", "test"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip()
            audit = root / "audit"
            audit.mkdir()
            (audit / "compliant-candidates.jsonl").write_text("{}\n")
            (audit / "final-axis-map.jsonl").write_text("{}\n")
            data_root = root / "data-root"
            snapshot = data_root / "data/serp/phase0_top20_searxng.parquet"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"parquet-test")
            environment_file = root / "pilot.env"
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(root),
                    "USER": "test-user",
                    "GEODML_REPOSITORY": str(repository),
                    "GEODML_PROJECT_ROOT": str(root / "project"),
                    "GEODML_CACHE_ROOT": str(root / "cache"),
                    "AUDIT_ROOT": str(audit),
                    "ACL_ARR_SEARCH_SNAPSHOT": str(snapshot),
                    "ACL_ARR_ENVIRONMENT_FILE": str(environment_file),
                    "ACL_ARR_RUN_ROOT": str(root / "stale-run-root"),
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(JUPITER_ROOT / "prepare_acl_arr_pilot_environment.sh"),
                    commit,
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ACL_ARR_ENVIRONMENT=PASS", result.stdout)
            expected_run_root = root / "project/runs/acl-arr-document-pilot" / (
                f"pilot-128-{commit[:7]}"
            )
            sourced = subprocess.check_output(
                [
                    "bash",
                    "-c",
                    'source "$1"; printf "%s\\n%s" "$GEODML_EXPECTED_COMMIT" "$ACL_ARR_RUN_ROOT"',
                    "_",
                    str(environment_file),
                ],
                text=True,
            )
            self.assertEqual(sourced, f"{commit}\n{expected_run_root}")

    def test_allocation_launcher_freezes_the_approved_budget(self) -> None:
        source = (JUPITER_ROOT / "launch_acl_arr_document_pilot_tmux.sh").read_text()
        self.assertEqual(source.count("salloc "), 1)
        self.assertIn("--nodes=1", source)
        self.assertIn("--gres=gpu:4", source)
        self.assertIn("--cpus-per-task=32", source)
        self.assertIn("--mem=512G", source)
        self.assertIn("--time=03:00:00", source)
        self.assertIn("maximum 12 GH200 GPU-hours", source)
        self.assertIn("allocation-requested-once.txt", source)

    def test_verify_only_checks_all_four_local_snapshots(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = []
            locks = []
            for index, model_id in enumerate(MODEL_IDS, start=1):
                revision = f"{index:x}" * 40
                snapshot = root / f"snapshot-{index}"
                snapshot.mkdir()
                models.append({"model_id": model_id, "model_revision": revision})
                locks.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "snapshot": str(snapshot),
                    }
                )
            (root / "models.json").write_text(json.dumps({"models": models}))
            (root / "model-snapshots.json").write_text(
                json.dumps({"models": locks})
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DOWNLOADER),
                    "--run-root",
                    str(root),
                    "--verify-only",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MODEL_PANEL_VERIFICATION=PASS models=4", result.stdout)

    def test_verify_only_rejects_a_missing_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = []
            locks = []
            for index, model_id in enumerate(MODEL_IDS, start=1):
                revision = f"{index:x}" * 40
                snapshot = root / f"snapshot-{index}"
                if index != 4:
                    snapshot.mkdir()
                models.append({"model_id": model_id, "model_revision": revision})
                locks.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "snapshot": str(snapshot),
                    }
                )
            (root / "models.json").write_text(json.dumps({"models": models}))
            (root / "model-snapshots.json").write_text(
                json.dumps({"models": locks})
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(DOWNLOADER),
                    "--run-root",
                    str(root),
                    "--verify-only",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing snapshot", result.stderr)


if __name__ == "__main__":
    unittest.main()
