import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from analysis.scripts import search_vllm_stage as stage


def gpu_inventory():
    return tuple(
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GH200 120GB",
            "memory_total_mib": 97871,
        }
        for index in range(4)
    )


def profile(**overrides):
    values = {
        "stage": "mistral-primary",
        "model_id": "mistralai/Mistral-Small-4-119B-2603",
        "model_revision": "a" * 40,
        "vllm_executable": "/runtime/bin/vllm",
        "vllm_version": "0.28.0",
        "vllm_help": "--data-parallel-size --language-model-only",
        "visible_gpus": gpu_inventory(),
        "cuda_visible_devices": "0,1,2,3",
        "expected_gpu_name_pattern": "GH200",
        "host": "127.0.0.1",
        "port": 8010,
        "data_parallel_size": 1,
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_model_len": 41472,
        "gpu_memory_utilization": 0.90,
        "request_concurrency": 8,
        "language_model_only": True,
        "tokenizer_mode": "mistral",
        "attention_backend": "FLASH_ATTN_MLA",
        "config_format": "mistral",
        "load_format": "mistral",
        "structured_outputs_config": {"backend": "xgrammar"},
    }
    values.update(overrides)
    return stage.build_profile(**values)


class SearchVllmStageTests(unittest.TestCase):
    def test_compatibility_defaults_keep_one_public_tp4_server(self):
        record = profile()
        serving = record["serving"]
        self.assertEqual(serving["public_base_url"], "http://127.0.0.1:8010/v1")
        self.assertEqual(serving["data_parallel_size"], 1)
        self.assertEqual(serving["tensor_parallel_size"], 4)
        self.assertEqual(serving["request_concurrency"], 8)
        self.assertNotIn("--data-parallel-size", record["server_argv"])
        self.assertEqual(record["server_argv"].count("--tensor-parallel-size"), 1)

    def test_opt_in_dp_requires_help_support_and_exactly_four_gpus(self):
        record = profile(data_parallel_size=2, tensor_parallel_size=2)
        flag = record["server_argv"].index("--data-parallel-size")
        self.assertEqual(record["server_argv"][flag + 1], "2")
        with self.assertRaisesRegex(ValueError, "lacks --data-parallel-size"):
            profile(data_parallel_size=2, tensor_parallel_size=2,
                    vllm_help="--tensor-parallel-size")
        with self.assertRaisesRegex(ValueError, "DP x TP must equal four"):
            profile(data_parallel_size=1, tensor_parallel_size=2)
        with self.assertRaisesRegex(ValueError, "exactly four visible GPUs"):
            profile(visible_gpus=gpu_inventory()[:3])
        with self.assertRaisesRegex(ValueError, "exactly four assigned GPUs"):
            profile(cuda_visible_devices="0,1")

    def test_profile_is_complete_and_hash_covers_canonical_server_argv(self):
        record = profile()
        self.assertEqual(
            set(record),
            {
                "format_version", "stage", "model", "runtime", "server_argv",
                "serving", "features", "visible_gpu_assignment", "cache_policy",
                "profile_sha256",
            },
        )
        self.assertEqual(record["model"], {
            "model_id": "mistralai/Mistral-Small-4-119B-2603",
            "model_revision": "a" * 40,
        })
        self.assertEqual(record["runtime"]["vllm_version"], "0.28.0")
        changed = profile(request_concurrency=9)
        self.assertNotEqual(record["profile_sha256"], changed["profile_sha256"])
        changed = profile(gpu_memory_utilization=0.89)
        self.assertNotEqual(record["profile_sha256"], changed["profile_sha256"])

    def test_eager_and_nccl_fallback_flags_are_profiled(self):
        help_text = (
            "--data-parallel-size --language-model-only "
            "--enforce-eager --disable-custom-all-reduce"
        )
        record = profile(
            vllm_help=help_text,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.assertIn("--enforce-eager", record["server_argv"])
        self.assertIn("--disable-custom-all-reduce", record["server_argv"])
        self.assertNotEqual(record["profile_sha256"], profile()["profile_sha256"])
        self.assertEqual(stage.verify_profile(record), record)

        with self.assertRaisesRegex(ValueError, "lacks --enforce-eager"):
            profile(enforce_eager=True)
        with self.assertRaisesRegex(ValueError, "lacks --disable-custom-all-reduce"):
            profile(disable_custom_all_reduce=True)

    def test_profile_hash_excludes_invocation_gpu_and_cache_facts(self):
        first = profile()
        changed_gpus = tuple(
            {**gpu, "uuid": f"OTHER-{gpu['index']}"} for gpu in gpu_inventory()
        )
        second = profile(
            visible_gpus=changed_gpus,
            cuda_visible_devices="OTHER-3,OTHER-2,OTHER-1,OTHER-0",
        )
        self.assertEqual(first["profile_sha256"], second["profile_sha256"])
        serialized = json.dumps(first)
        self.assertNotIn("GPU-0", serialized)
        self.assertNotIn("cuda_visible_devices", first["visible_gpu_assignment"])
        self.assertNotIn("/fscratch", serialized)
        self.assertEqual(first["visible_gpu_assignment"]["required_gpu_count"], 4)

    def test_expected_gpu_family_and_structured_output_shape_fail_closed(self):
        wrong = list(gpu_inventory())
        wrong[2] = {**wrong[2], "name": "NVIDIA H100 80GB"}
        with self.assertRaisesRegex(ValueError, "expected GPU name pattern"):
            profile(visible_gpus=wrong)
        with self.assertRaisesRegex(ValueError, "valid regular expression"):
            profile(expected_gpu_name_pattern="[")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            profile(structured_outputs_config=["xgrammar"])
        self.assertEqual(
            profile()["visible_gpu_assignment"]["expected_gpu_name_pattern"],
            "GH200",
        )

    def test_cuda_assignment_is_required_and_matches_all_indexes_or_all_uuids(self):
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES is required"):
            profile(cuda_visible_devices=None)
        with self.assertRaisesRegex(ValueError, "indexes or UUIDs"):
            profile(cuda_visible_devices="0,1,2,GPU-3")
        with self.assertRaisesRegex(ValueError, "indexes or UUIDs"):
            profile(cuda_visible_devices="4,5,6,7")
        uuid_assignment = profile(
            cuda_visible_devices="GPU-3,GPU-2,GPU-1,GPU-0"
        )
        self.assertEqual(uuid_assignment["visible_gpu_assignment"]["required_gpu_count"], 4)

    def test_sidecar_is_create_or_verify_and_mismatch_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scientific-output.serving-profile.json"
            first = profile()
            self.assertEqual(
                stage.create_or_verify_profile(path, first), first["profile_sha256"]
            )
            before = path.read_bytes()
            self.assertEqual(stage.create_or_verify_profile(path, profile()), first["profile_sha256"])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "serving profile changed"):
                stage.create_or_verify_profile(path, profile(request_concurrency=9))
            self.assertEqual(path.read_bytes(), before)

    def test_tampered_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            stage.create_or_verify_profile(path, profile())
            value = json.loads(path.read_text())
            value["serving"]["port"] = 9000
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                stage.load_profile(path)

    def test_rehashed_malformed_server_argv_fails_cross_field_validation(self):
        for mutate in (
            lambda value: value["server_argv"].__setitem__(2, "other/model"),
            lambda value: value["server_argv"].__setitem__(
                value["server_argv"].index("--port") + 1, "9000"
            ),
            lambda value: value["serving"].__setitem__("tensor_parallel_size", 2),
        ):
            malformed = copy.deepcopy(profile())
            mutate(malformed)
            malformed["profile_sha256"] = stage._profile_hash(malformed)
            with self.assertRaisesRegex(ValueError, "profile|argv|topology"):
                stage.verify_profile(malformed)

    def test_all_six_caches_use_job_stage_and_full_profile_hash(self):
        record = profile()
        environment = stage.cache_environment(
            record,
            job_id="1725979",
            cache_base="/fscratch/geodml/compile-cache",
            hostname="jpbo-001-04",
        )
        self.assertEqual(set(environment), set(stage.CACHE_VARIABLES))
        expected = (
            "/fscratch/geodml/compile-cache/job1725979/jpbo-001-04/mistral-primary/"
            + record["profile_sha256"]
        )
        self.assertTrue(
            all(
                str(Path(value)).startswith(expected + "/")
                for value in environment.values()
            )
        )
        self.assertEqual(len(set(environment.values())), 6)
        self.assertIn("FLASHINFER_WORKSPACE_BASE", environment)
        self.assertIn("TRTLLM_DG_CACHE_DIR", environment)
        other_node = stage.cache_environment(
            record,
            job_id="1725979",
            cache_base="/fscratch/geodml/compile-cache",
            hostname="jpbo-001-05",
        )
        self.assertTrue(set(environment.values()).isdisjoint(other_node.values()))

    def test_port_collision_fails_before_server_launch(self):
        class BusySocket:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def bind(self, _address):
                raise OSError("fixture port collision")

        with self.assertRaisesRegex(ValueError, "already in use"):
            stage.ensure_port_available(
                "127.0.0.1", 8010, socket_factory=BusySocket
            )

    def test_second_cooperative_port_owner_fails_while_first_holds_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            record = profile()
            cache_base = str(Path(directory) / "cache")
            with stage.port_ownership(record, cache_base=cache_base):
                with self.assertRaisesRegex(ValueError, "already owned"):
                    with stage.port_ownership(record, cache_base=cache_base):
                        self.fail("second owner acquired the same serving port")

    def test_dp_scientific_use_requires_matching_benchmark_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approval = Path(directory) / "approval.json"
            dp1 = profile()
            self.assertIsNone(stage.require_benchmark_approval(dp1, None))
            dp2 = profile(data_parallel_size=2, tensor_parallel_size=2)
            with self.assertRaisesRegex(ValueError, "benchmark approval"):
                stage.require_benchmark_approval(dp2, None)
            value = {
                "format_version": stage.APPROVAL_FORMAT_VERSION,
                "profile_sha256": dp2["profile_sha256"],
                "benchmark_result_sha256": "b" * 64,
                "approved_for_scientific_use": True,
            }
            approval.write_text(json.dumps(value))
            self.assertEqual(
                stage.require_benchmark_approval(dp2, approval),
                value,
            )
            value["profile_sha256"] = "c" * 64
            approval.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "profile hash"):
                stage.require_benchmark_approval(dp2, approval)
            profile_path = root / "profile.json"
            stage.create_or_verify_profile(profile_path, dp2)
            with self.assertRaisesRegex(ValueError, "benchmark approval"):
                stage.run_stage(
                    profile_path,
                    root / "server.log",
                    ["true"],
                    cache_base=root / "cache",
                    startup_timeout_seconds=1,
                    benchmark_approval_path=None,
                )

    def test_runtime_and_readiness_fail_closed(self):
        record = profile()
        runtime = record["runtime"]
        with mock.patch.object(
            stage,
            "inspect_vllm",
            return_value=(
                runtime["vllm_executable"],
                "different-version",
                "--data-parallel-size --language-model-only",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                stage.verify_runtime(record)

        class Process:
            def poll(self):
                return None

        class Response:
            def __init__(self, model_id):
                self.model_id = model_id

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({"data": [{"id": self.model_id}]}).encode()

        with mock.patch.object(stage, "urlopen", return_value=Response(record["model"]["model_id"])):
            stage.wait_until_ready(
                Process(),
                base_url=record["serving"]["public_base_url"],
                model_id=record["model"]["model_id"],
                timeout_seconds=1,
            )
        with mock.patch.object(stage, "urlopen", return_value=Response("other/model")), \
                mock.patch.object(stage.time, "monotonic", side_effect=[0.0, 0.0, 2.0, 2.0]), \
                mock.patch.object(stage.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                stage.wait_until_ready(
                    Process(),
                    base_url=record["serving"]["public_base_url"],
                    model_id=record["model"]["model_id"],
                    timeout_seconds=1,
                )

    def test_server_death_interrupts_controller_and_cleanup_signals_orphan_group(self):
        controller = mock.Mock()
        controller.poll.return_value = None
        server = mock.Mock()
        server.poll.return_value = 17
        with mock.patch.object(stage, "_terminate_group") as terminate:
            with self.assertRaisesRegex(RuntimeError, "server exited"):
                stage.wait_for_controller(controller, server, poll_seconds=0)
        terminate.assert_called_once_with(controller)

        exited_parent = mock.Mock(pid=456)
        exited_parent.poll.return_value = 0
        with mock.patch.object(stage, "_group_exists", side_effect=[True, False], create=True), \
                mock.patch.object(stage.os, "killpg") as killpg:
            stage._terminate_group(exited_parent, grace_seconds=0)
        killpg.assert_any_call(456, stage.signal.SIGTERM)

    def test_signal_receipt_names_the_active_lifecycle_phase(self):
        startup = {"server_status": "starting", "controller_status": "not_started"}
        stage.record_stage_signal(startup, stage.signal.SIGTERM, controller_started=False)
        self.assertEqual(startup["server_status"], "signaled")
        self.assertEqual(startup["controller_status"], "not_started")
        running = {"server_status": "ready", "controller_status": "running"}
        stage.record_stage_signal(running, stage.signal.SIGINT, controller_started=True)
        self.assertEqual(running["server_status"], "ready")
        self.assertEqual(running["controller_status"], "signaled")

    def test_runtime_json_records_invocation_facts_adjacent_to_server_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = profile()
            cache_paths = stage.cache_environment(
                record,
                job_id="1725979",
                cache_base=str(root / "cache"),
                hostname="jpbo-001-04",
            )
            runtime = stage.build_runtime_record(
                profile_path=root / "profile.json",
                profile=record,
                visible_gpus=gpu_inventory(),
                cuda_visible_devices="0,1,2,3",
                cache_paths=cache_paths,
                hostname="jpbo-001-04",
                job_id="1725979",
                step_id="1725979.5",
                benchmark_approval={
                    "path": str(root / "approval.json"),
                    "sha256": "b" * 64,
                    "profile_sha256": record["profile_sha256"],
                    "benchmark_result_sha256": "c" * 64,
                },
            )
            server_log = root / "server.log"
            path = stage.write_runtime_record(server_log, runtime)
            self.assertEqual(path, root / "server.log.runtime.json")
            saved = json.loads(path.read_text())
            self.assertEqual(saved["profile_sha256"], record["profile_sha256"])
            self.assertEqual(saved["visible_gpu_inventory"], list(gpu_inventory()))
            self.assertEqual(saved["cuda_visible_devices"], "0,1,2,3")
            self.assertEqual(saved["hostname"], "jpbo-001-04")
            self.assertEqual(saved["slurm_job_id"], "1725979")
            self.assertEqual(saved["slurm_step_id"], "1725979.5")
            self.assertEqual(saved["cache_paths"], cache_paths)
            self.assertEqual(saved["benchmark_approval"]["sha256"], "b" * 64)
            self.assertEqual(saved["controller_status"], "not_started")
            self.assertFalse(list(root.glob("server.log.runtime.json.*")))

    def test_runtime_binding_covers_static_facts_and_validates_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = profile()
            profile_path = root / "profile.json"
            stage.create_or_verify_profile(profile_path, record)
            caches = stage.cache_environment(
                record,
                job_id="1725979",
                cache_base=root / "cache",
                hostname="jpbo-001-04",
            )
            runtime = stage.build_runtime_record(
                profile_path=profile_path,
                profile=record,
                visible_gpus=gpu_inventory(),
                cuda_visible_devices="0,1,2,3",
                cache_paths=caches,
                hostname="jpbo-001-04",
                job_id="1725979",
                step_id="1725979.5",
                benchmark_approval=None,
            )
            runtime.update(
                server_status="terminated",
                server_exit_code=0,
                controller_status="exited",
                controller_exit_code=0,
                finished_at_epoch_seconds=time.time(),
            )
            runtime_path = stage.write_runtime_record(root / "server.log", runtime)
            binding = stage.load_runtime_binding(
                runtime_path,
                expected_profile_sha256=record["profile_sha256"],
                require_approval=False,
            )
            self.assertEqual(binding["visible_gpu_inventory"], list(gpu_inventory()))
            self.assertEqual(binding["hostname"], "jpbo-001-04")
            self.assertEqual(binding["slurm_step_id"], "1725979.5")
            self.assertEqual(binding["cache_paths"], caches)
            self.assertEqual(
                binding["started_at_epoch_seconds"],
                runtime["started_at_epoch_seconds"],
            )
            for key in (
                "visible_gpu_inventory",
                "cache_paths",
                "controller_status",
                "finished_at_epoch_seconds",
            ):
                malformed = dict(runtime)
                malformed.pop(key)
                stage.write_runtime_record(root / "server.log", malformed)
                with self.assertRaisesRegex(ValueError, "runtime record"):
                    stage.load_runtime_binding(
                        runtime_path,
                        expected_profile_sha256=record["profile_sha256"],
                        require_approval=False,
                    )


if __name__ == "__main__":
    unittest.main()
