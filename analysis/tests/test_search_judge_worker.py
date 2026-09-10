import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class JudgeWorkerTests(unittest.TestCase):
    def test_workers_use_shared_stage_profile_and_lifecycle(self):
        stage_helper = 'analysis/scripts/search_vllm_stage.py'
        for relative in (
            'scripts/slurm/jupiter/run_search_mistral_primary.sh',
            'scripts/slurm/jupiter/run_search_quote_judge.sh',
        ):
            worker = Path(__file__).resolve().parents[1] / relative
            source = worker.read_text()
            self.assertIn(stage_helper, source)
            self.assertIn(' prepare ', source)
            self.assertIn(' run ', source)

    def test_mistral_defaults_to_dp1_tp4_concurrency8_port8010(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_mistral_primary.sh'
        source = worker.read_text()
        self.assertIn('geodml_dp="${SEARCH_PRIMARY_DATA_PARALLEL_SIZE:-1}"', source)
        self.assertIn('geodml_tp="${SEARCH_PRIMARY_TENSOR_PARALLEL_SIZE:-4}"', source)
        self.assertIn('geodml_concurrency="${SEARCH_PRIMARY_REQUEST_CONCURRENCY:-8}"', source)
        self.assertIn('geodml_port="${SEARCH_PRIMARY_PORT:-8010}"', source)
        self.assertIn(
            'geodml_max_model_len="${SEARCH_PRIMARY_MAX_MODEL_LEN:-41472}"',
            source,
        )
        self.assertIn(
            'geodml_output="${SEARCH_PRIMARY_OUTPUT:-$SEARCH_PILOT_ROOT/primary-schema-fix-0b3ce8acb5d6/model-config-c860fb2fb61da06a8443}"',
            source,
        )
        self.assertNotIn('SEARCH_VLLM_', source)
        self.assertIn('--expected-gpu-name-pattern GH200', source)
        self.assertIn('SEARCH_PRIMARY_BENCHMARK_APPROVAL_PATH', source)
        self.assertIn('--benchmark-approval "$geodml_approval_path"', source)
        self.assertIn('--serving-profile "$geodml_profile"', source)
        self.assertIn(
            'geodml_answer_max_tokens="${SEARCH_PRIMARY_ANSWER_MAX_TOKENS:-}"',
            source,
        )
        self.assertIn(
            'geodml_args+=(--answer-max-tokens "$geodml_answer_max_tokens")',
            source,
        )
        self.assertIn('ANSWER_MAX_TOKENS=%s', source)
        self.assertIn('check_search_mistral_context.py', source)
        self.assertIn('--max-model-len "$geodml_max_model_len"', source)

    def test_judge_defaults_to_dp1_tp4_concurrency8_port8010(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_quote_judge.sh'
        source = worker.read_text()
        self.assertIn('geodml_dp="${SEARCH_JUDGE_DATA_PARALLEL_SIZE:-1}"', source)
        self.assertIn('geodml_tp="${SEARCH_JUDGE_TENSOR_PARALLEL_SIZE:-4}"', source)
        self.assertIn('geodml_concurrency="${SEARCH_JUDGE_REQUEST_CONCURRENCY:-8}"', source)
        self.assertIn('geodml_port="${SEARCH_JUDGE_PORT:-8010}"', source)
        self.assertNotIn('SEARCH_VLLM_', source)
        self.assertIn('--expected-gpu-name-pattern GH200', source)
        self.assertIn('SEARCH_JUDGE_BENCHMARK_APPROVAL_PATH', source)
        self.assertIn('--benchmark-approval "$geodml_approval_path"', source)
        self.assertIn('--serving-profile "$geodml_profile"', source)

    def test_mistral_startup_timeout_is_configurable_with_safe_default(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_mistral_primary.sh'
        source = worker.read_text()
        self.assertIn(
            'geodml_startup_timeout_seconds="${SEARCH_PRIMARY_STARTUP_TIMEOUT_SECONDS:-1800}"',
            source,
        )
        self.assertIn(
            '--startup-timeout-seconds "$geodml_startup_timeout_seconds"', source
        )

    def test_mistral_complete_skips_loading(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_mistral_primary.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            (root / 'bin/activate').write_text('true\n')
            startup = root / 'startup'
            startup.write_text('''
module() { return 0; }
git() { if [[ "$1" == rev-parse ]]; then echo fixture; fi; }
python3() {
  if [[ "$*" == *--preflight-only* ]]; then return 0; fi
  echo UNEXPECTED_LOADING >&2
  return 99
}
''')
            result = subprocess.run(['bash', str(worker)], capture_output=True, text=True,
                env=dict(os.environ, BASH_ENV=str(startup), ACL_ARR_VENV=str(root),
                         GEODML_EXECUTION_COMMIT='fixture', SLURM_JOB_ID='fixture',
                         GEODML_EXPECTED_JOB_ID='fixture', SEARCH_PILOT_ROOT=str(root / 'pilot')))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('ALREADY_COMPLETE', result.stdout)
            self.assertNotIn('UNEXPECTED_LOADING', result.stderr)
            self.assertFalse((root / 'pilot').exists())

    def test_mistral_only_blocks_historical_unbound_partial_results(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_mistral_primary.sh'
        source = worker.read_text()
        self.assertIn('historical partial results have no serving profile provenance', source)
        self.assertIn('manifest["serving_profile"]["path"]', source)
        self.assertNotIn('partial results exist; preserve them for retry review', source)

    def test_complete_skips_engine_and_invalid_preflight_fails(self):
        worker = Path(__file__).resolve().parents[1] / 'scripts/slurm/jupiter/run_search_quote_judge.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            (root / 'bin/activate').write_text('true\n')
            startup = root / 'startup'
            startup.write_text('''
module() { return 0; }
git() { if [[ "$1" == rev-parse ]]; then echo fixture; fi; }
python3() {
  if [[ "$*" == *--preflight-only* ]]; then return "$CHECK_STATUS"; fi
  echo UNEXPECTED_MODEL_PREFLIGHT >&2
  return 99
}
''')
            environment = dict(os.environ, BASH_ENV=str(startup), ACL_ARR_VENV=str(root),
                GEODML_EXECUTION_COMMIT='fixture', SLURM_JOB_ID='fixture',
                GEODML_EXPECTED_JOB_ID='fixture', SEARCH_BUNDLE_DIR='fixture',
                SEARCH_PRIMARY_OUTPUT='fixture', SEARCH_JUDGE_OUTPUT=str(root / 'output'),
                SEARCH_JUDGE_MODEL='fixture', SEARCH_JUDGE_REVISION='fixture', ACL_ARR_RUN_ROOT='fixture')
            for status in (0, 2):
                result = subprocess.run(['bash', str(worker)], env=dict(environment, CHECK_STATUS=str(status)),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertNotIn('UNEXPECTED_MODEL_PREFLIGHT', result.stderr)
                self.assertFalse((root / 'output').exists())
                if status == 0:
                    self.assertIn('ALREADY_COMPLETE', result.stdout)
            (root / 'output').mkdir()
            (root / 'output/run_manifest.json').write_text('{}')
            result = subprocess.run(['bash', str(worker)], env=dict(environment, CHECK_STATUS='3'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('partial run requires', result.stderr)
            self.assertNotIn('JUDGE_WORKER_FAILED', result.stderr)


if __name__ == '__main__':
    unittest.main()
