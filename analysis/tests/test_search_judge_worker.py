"""CPU-only checks of worker branching; these do not test serving compatibility."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class JudgeWorkerTests(unittest.TestCase):
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
